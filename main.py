"""Smart CCTV — main monitoring loop.

This is the conductor of the whole system. It owns the camera, runs the
per-frame pipeline, and decides when a face is family, when it is a
stranger, and when the alarm should sound.

The story of one frame
----------------------
Every loop iteration reads a single frame from the camera and walks it
through the pipeline, cheapest checks first so a quiet scene costs almost
nothing:

    1. Motion gate   (motion.py)   — did anything move? If not, skip the
                                     expensive steps but still show the
                                     frame, so the video never freezes.
    2. Enhance       (enhance.py)  — fix brightness/contrast so faces are
                                     easier to see.
    3. Detect        (faces.py)    — find faces (HOG, cheap) on a
                                     downscaled frame.
    4. Recognize     (faces.py)    — compare each face against the family
                                     database.
    5. Track         (tracking.py) — smooth boxes over time and decide
                                     identity by majority vote.
    6. Decide        (this file)   — family → green name; unknown → red
                                     banner, countdown, snapshot, siren.

While an unknown face lingers, yolo.py may explain the scene (animal →
suppress the alarm, human → shorten the delay). storage.py records every
event, and hud.py draws all on-screen overlays.

Design notes
------------
- All tunables live in config.py; PERFORMANCE_MODE picks a hardware tier.
- The camera driver queue is capped at one frame so the view stays live.
- The loop is wrapped in try/finally so Ctrl+C or an error still releases
  the camera, silences the siren, and closes the window.

Keyboard: ``q`` quits, ``s`` silences the siren, ``a`` (or clicking the
``+ ADD FAMILY`` button) opens the in-app family enrollment flow
(cctv/enroll.py).
"""

import os
import time

import cv2

from config import (
    FACE_TOLERANCE,
    UNKNOWN_CONFIRMATIONS,
    UNKNOWN_DELAY_SECONDS,
    NIGHT_UNKNOWN_DELAY_SECONDS,
    SNAPSHOT_INTERVAL,
    SIGHTING_LOG_INTERVAL,
    DETECTION_SCALE,
    ENABLE_CNN_FALLBACK,
    TRACKING_SKIP_FRAMES,
    YOLO_SKIP_FRAMES,
    MOTION_ENABLED,
    MOTION_THRESHOLD,
    MOTION_MIN_AREA,
    MOTION_SCALE,
    MOTION_BG_ALPHA,
    FAMILY_DIR,
    SNAPSHOT_DIR,
    LOG_DIR,
    ANIMAL_DETECTION_ENABLED,
    UNKNOWN_HUMAN_DELAY_SECONDS,
    SHOW_FPS,
    SIREN_RETRIGGER_COOLDOWN,
    SIREN_DAY_DURATION,
    SIREN_NIGHT_DURATION,
    NIGHT_START_HOUR,
)

from cctv.enhance import enhance_frame
from cctv.faces import (
    load_family_database,
    recognize_face,
    detect_faces_enhanced,
)
from cctv.hud import (
    draw_face_boxes,
    draw_countdown,
    draw_family_text,
    draw_mode,
    draw_status,
    draw_fps,
    draw_unknown_alert,
    draw_add_button,
)
from cctv.enroll import run_enrollment
from cctv.motion import MotionDetector
from cctv.tracking import match_tracks
from cctv.siren import Siren
from cctv.yolo import ObjectDetector
from cctv.storage import (
    initialize_database,
    log_event,
    logger,
    enforce_retention,
)
from cctv import hardware
from cctv.timeutil import (
    is_night_mode,
    siren_duration,
    nepal_now,
)


# Create needed folders if missing
os.makedirs(FAMILY_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def main():

    # Set up database, siren, detector and load known faces at startup
    initialize_database()
    enforce_retention()

    siren = Siren()
    detector = ObjectDetector(enabled=ANIMAL_DETECTION_ENABLED)
    motion = MotionDetector(
        threshold=MOTION_THRESHOLD,
        min_area=MOTION_MIN_AREA,
        scale=MOTION_SCALE,
        bg_alpha=MOTION_BG_ALPHA,
    )

    known_encodings, known_names = load_family_database()

    print("\n--------------------------------")
    print("SMART CCTV")
    print("--------------------------------")

    print(
        f"Family face samples: "
        f"{len(known_encodings)}"
    )

    print(
        f"Recognition tolerance: "
        f"{FACE_TOLERANCE}"
    )

    print(
        f"Unknown delay: "
        f"{UNKNOWN_DELAY_SECONDS}s"
    )

    print(
        f"Security modes (Nepal time): "
        f"day siren {SIREN_DAY_DURATION}s, "
        f"night mode from {NIGHT_START_HOUR:02d}:00 "
        f"siren {SIREN_NIGHT_DURATION}s"
    )

    print("--------------------------------\n")

    # Open the camera through the hardware abstraction layer so the same
    # code runs on a PC today and on a Raspberry Pi 5 / ESP32-CAM later.
    camera = hardware.open_camera()
    print(f"Camera: {hardware.describe()}")

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    # Counters to track unknown detection
    unknown_count = 0
    unknown_start = None

    last_snapshot = 0
    last_sighting = {}  # person -> timestamp of last FAMILY_SIGHTING log

    tracked_faces = {}
    frame_counter = 0

    # FPS measurement: smoothed frames-per-second for the HUD overlay
    fps = 0.0
    last_frame_time = time.time()

    # Last YOLO result, reused between throttled detection runs
    animal_seen, human_seen = False, False

    reconnect_attempts = 0

    running = True

    # ── Add-family button state ──
    # The mouse callback records the cursor and any left-click; the main
    # loop hit-tests the click against the button rect drawn each frame.
    mouse_pos = [0, 0]
    click_pos = None  # set by the callback, consumed by the loop
    button_rect = None

    def on_mouse(event, x, y, flags, param):
        nonlocal click_pos
        mouse_pos[0], mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos = (x, y)

    # The window must exist before a mouse callback can be attached, so
    # create it explicitly (imshow alone would create it too late).
    cv2.namedWindow("Smart CCTV Security")
    cv2.setMouseCallback("Smart CCTV Security", on_mouse)

    def _open_enrollment():
        """Pause security, run the in-app add-family flow, reload the DB."""
        nonlocal known_encodings, known_names
        nonlocal unknown_count, unknown_start

        # Pause the alarm side so registration is calm and safe
        siren.stop()
        unknown_count = 0
        unknown_start = None

        try:
            enrolled = run_enrollment(camera)
        except Exception as err:
            print(f"[ENROLL] Error during enrollment: {err}")
            enrolled = False

        if enrolled:
            # Pick up the new photos immediately without a restart
            known_encodings, known_names = load_family_database()
            print(
                f"Family database reloaded: "
                f"{len(known_encodings)} samples."
            )

        # Avoid a motion spike / stale tracks when monitoring resumes
        motion.reset()
        tracked_faces.clear()

    # Main camera loop — wrapped so Ctrl+C or an unexpected error still
    # releases the camera, silences the siren, and closes the window.
    try:
        while running:

            ret, frame = camera.read()

            if not ret:

                # Reconnect with exponential backoff instead of a fixed sleep
                wait = min(30, 2 ** reconnect_attempts)
                print(
                    f"WARNING: Camera frame unavailable. "
                    f"Retrying in {wait}s (attempt {reconnect_attempts + 1})."
                )

                camera.release()
                time.sleep(wait)

                camera = hardware.open_camera()

                reconnect_attempts += 1

                if camera.isOpened():
                    reconnect_attempts = 0
                    motion.reset()  # avoid a motion spike after reconnect

                continue

            # ── FPS measurement (smoothed) ──
            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst

            # ── Motion gate ──
            # When nothing moves, skip the expensive pipeline — but the frame
            # is still displayed below, so the video never freezes. Tracks are
            # aged out gradually (via match_tracks) so stale boxes fade away.
            has_motion = not MOTION_ENABLED or motion.has_motion(frame)

            if has_motion:

                # ── Image enhancement ──
                enhanced = enhance_frame(frame)

                # ── Downscale for recognition speed ──
                small_frame = cv2.resize(
                    enhanced,
                    (0, 0),
                    fx=DETECTION_SCALE,
                    fy=DETECTION_SCALE
                )

                rgb_frame = cv2.cvtColor(
                    small_frame,
                    cv2.COLOR_BGR2RGB
                )

            else:
                rgb_frame = None

            # Full-resolution frame for CNN fallback. Only converted when the
            # fallback is enabled AND something is moving — otherwise this is
            # wasted work on every idle frame.
            rgb_full = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if (ENABLE_CNN_FALLBACK and has_motion) else None
            )

            # Find all faces and their encodings in this frame.
            # Skip detection on idle and non-detection frames, rely on track
            # persistence in between.
            if has_motion and frame_counter % TRACKING_SKIP_FRAMES == 0:
                locations, encodings = detect_faces_enhanced(
                    rgb_frame, rgb_full
                )
            else:
                locations, encodings = [], []

            # ── Recognize each detected face ──
            raw_faces = []  # (location, name, confidence)

            for encoding, location in zip(encodings, locations):
                name, distance, confidence = recognize_face(
                    encoding,
                    known_encodings,
                    known_names
                )
                raw_faces.append((location, name, confidence))

            # Update tracks with smoothing and frame-skip. Skip the call
            # entirely when there is nothing to match and no live tracks —
            # a pure no-op on idle frames.
            if raw_faces or tracked_faces:
                tracked_faces = match_tracks(
                    raw_faces, tracked_faces, frame_counter
                )

            # Build final classification from majority vote
            recognized_people = []
            unknown_faces = []
            displayed_faces = []  # (fullres_location, label, color, confidence)

            for tid, track in tracked_faces.items():
                final_name = track.majority_name
                conf = track.avg_confidence

                # Use smoothed location, convert to full-res coords
                top, right, bottom, left = track.smoothed
                scale = DETECTION_SCALE
                ftop, fright, fbottom, fleft = (
                    int(top / scale), int(right / scale),
                    int(bottom / scale), int(left / scale)
                )

                if final_name == "UNKNOWN":
                    unknown_faces.append(track.last_seen)
                    displayed_faces.append(
                        ((fleft, ftop, fright, fbottom),
                         "UNKNOWN", (0, 0, 255), conf)
                    )
                else:
                    recognized_people.append(final_name)
                    conf_pct = int(conf * 100)
                    displayed_faces.append(
                        ((fleft, ftop, fright, fbottom),
                         f"{final_name} {conf_pct}%", (0, 255, 0), conf)
                    )

            # Draw all tracked faces with smoothed boxes
            draw_face_boxes(frame, displayed_faces)

            # Prominent red banner whenever an unknown face is on screen
            if unknown_faces:
                draw_unknown_alert(frame)

            frame_counter += 1

            # Handle unknown faces: start timer, save snapshots, raise alarm

            # Object detection explains the scene while an unknown face lingers:
            # animals suppress the siren, a confirmed human shortens the delay.
            # Throttled to every YOLO_SKIP_FRAMES; last result reused in between.
            if unknown_faces and ANIMAL_DETECTION_ENABLED:
                if frame_counter % YOLO_SKIP_FRAMES == 0:
                    animal_seen, human_seen = detector.detect(frame)
            else:
                animal_seen, human_seen = False, False

            # Shorter confirmation delay in night security mode,
            # fastest delay when YOLO confirms a human
            if is_night_mode():
                active_delay = NIGHT_UNKNOWN_DELAY_SECONDS
            elif human_seen:
                active_delay = UNKNOWN_HUMAN_DELAY_SECONDS
            else:
                active_delay = UNKNOWN_DELAY_SECONDS

            if unknown_faces:

                unknown_count += 1

                # Only count an unknown after several consecutive frames

                if unknown_count >= UNKNOWN_CONFIRMATIONS:

                    if unknown_start is None:

                        unknown_start = time.time()

                        print(
                            "\nUNKNOWN PERSON CONFIRMED"
                        )

                        logger.warning(
                            "Unknown person confirmed"
                        )

                        log_event(
                            "UNKNOWN_CONFIRMED"
                        )

                    elapsed = (
                        time.time()
                        -
                        unknown_start
                    )

                    remaining = max(
                        0,
                        active_delay
                        -
                        elapsed
                    )

                    # Save a snapshot every few seconds

                    if (
                        time.time()
                        -
                        last_snapshot
                        >= SNAPSHOT_INTERVAL
                    ):

                        timestamp = nepal_now().strftime(
                            "%Y%m%d_%H%M%S"
                        )

                        path = os.path.join(
                            SNAPSHOT_DIR,
                            f"unknown_{timestamp}.jpg"
                        )

                        cv2.imwrite(
                            path,
                            frame
                        )

                        log_event(
                            "UNKNOWN_SNAPSHOT",
                            "UNKNOWN",
                            path
                        )

                        last_snapshot = time.time()

                    # Show countdown on screen
                    draw_countdown(frame, remaining)

                    # Trigger the siren once the delay has passed,
                    # unless an animal (and no human) explains the scene.
                    #
                    # The siren sounds in BOTH modes now (Nepal time):
                    #   • daytime        → SIREN_DAY_DURATION   (2 min)
                    #   • night security → SIREN_NIGHT_DURATION (5 min)
                    # It auto-stops after that duration (Siren timer), and
                    # a cooldown prevents an immediate re-trigger loop.

                    animal_only = animal_seen and not human_seen

                    # Cooldown is anchored to when the siren last STOPPED
                    # (not when it was triggered), so it cannot restart the
                    # instant a run finishes. 0.0 = never stopped = ready.
                    cooled_down = (
                        siren.last_stop == 0.0
                        or time.time() - siren.last_stop
                        >= SIREN_RETRIGGER_COOLDOWN
                    )

                    if (
                        elapsed >= active_delay
                        and not animal_only
                        and not siren.is_active
                        and cooled_down
                    ):

                        duration = siren_duration()
                        siren.start(duration=duration)

                        log_event(
                            "SIREN_TRIGGERED",
                            "UNKNOWN",
                            f"mode={'NIGHT' if is_night_mode() else 'DAY'},"
                            f" auto-stop={duration}s"
                        )

            else:

                unknown_count = 0
                unknown_start = None

            # Show the names of recognized family members

            if recognized_people:

                unique_people = sorted(
                    set(recognized_people)
                )

                draw_family_text(frame, unique_people)

                # Log family member sightings (rate-limited per person)
                now = time.time()
                for person in unique_people:
                    if now - last_sighting.get(person, 0) >= SIGHTING_LOG_INTERVAL:
                        log_event("FAMILY_SIGHTING", person=person)
                        last_sighting[person] = now

            # Show whether night security mode is active (Nepal time)

            draw_mode(frame, is_night_mode())

            # Show overall system status (alarm on or ok)

            draw_status(frame, siren.is_active)

            # Show the live FPS counter

            if SHOW_FPS:
                draw_fps(frame, fps)

            # Clickable '+ ADD FAMILY' button (bottom right). Hovering
            # brightens it; the rect is hit-tested against mouse clicks.
            mx, my = mouse_pos
            hover = False
            if button_rect is not None:
                bx0, by0, bx1, by1 = button_rect
                hover = bx0 <= mx <= bx1 and by0 <= my <= by1
            button_rect = draw_add_button(frame, hover)

            # Show the current camera frame

            cv2.imshow(
                "Smart CCTV Security",
                frame
            )

            # Keyboard controls: q quits, s stops the siren,
            # a opens the add-family-member flow

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                running = False

            elif key == ord("s"):

                siren.stop()

            elif key == ord("a"):

                click_pos = None  # ignore any stale click
                _open_enrollment()

            # Mouse click on the '+ ADD FAMILY' button
            elif click_pos is not None:
                cx, cy = click_pos
                click_pos = None
                bx0, by0, bx1, by1 = button_rect
                if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                    _open_enrollment()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Shut down cleanly when the loop ends

        siren.stop()

        camera.release()

        cv2.destroyAllWindows()

        print("\nSmart CCTV stopped safely.")


if __name__ == "__main__":
    main()
