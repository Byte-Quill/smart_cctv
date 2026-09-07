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

Keyboard: ``q`` quits, ``s`` silences the siren, ``t`` plays a short
siren self-test (safe, auto-stops), ``a`` (or clicking the
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
    STARTUP_SIREN_TEST,
    STARTUP_SIREN_TEST_DURATION,
    NIGHT_START_HOUR,
    MIRROR_DISPLAY,
    VAULT_KEY_SOURCE,
    VAULT_VERIFY_ON_STARTUP,
)

from cctv.enhance import enhance_frame
from cctv.faces import (
    load_family_database,
    load_family_from_vault,
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
from cctv.vault import FaceVault
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

    # ── Encrypted face vault ──
    # Open the vault (prompting for a passphrase if configured), verify
    # its HMAC tamper-evidence chain, and fail CLOSED if an attacker
    # edited or deleted biometric records — a compromised vault must
    # never silently continue.
    vault_cipher = None
    if VAULT_KEY_SOURCE == "passphrase":
        import getpass
        from cctv.crypto import VaultCipher

        while vault_cipher is None:
            try:
                passphrase = getpass.getpass("Vault passphrase: ")
                vault_cipher = VaultCipher.from_passphrase(passphrase)
            except Exception as err:
                print(f"Invalid passphrase ({err}); try again.")

    vault = FaceVault(cipher=vault_cipher)

    if VAULT_VERIFY_ON_STARTUP:
        problems = vault.verify_integrity()
        if problems:
            print("\n" + "!" * 60)
            print("VAULT INTEGRITY FAILURE — biometric data was tampered with")
            print("!" * 60)
            for problem in problems:
                print(f"  • {problem}")
            print(
                "\nRefusing to run with a compromised vault (fail-closed).\n"
                "Restore logs/vault.db from a trusted backup, or delete\n"
                "it and re-register your family."
            )
            vault.close()
            return
        print("Vault integrity verified — no tampering detected.")

    vault.enforce_unknown_retention()
    vault.cap_unknown_per_person()

    siren = Siren()

    # Audible power-on self-test so a presenter can confirm the alarm
    # works. It auto-silences after a few seconds and never affects real
    # alarm logic (not an activation, no cooldown side effects).
    if STARTUP_SIREN_TEST:
        siren.self_test(duration=STARTUP_SIREN_TEST_DURATION)

    detector = ObjectDetector(enabled=ANIMAL_DETECTION_ENABLED)
    motion = MotionDetector(
        threshold=MOTION_THRESHOLD,
        min_area=MOTION_MIN_AREA,
        scale=MOTION_SCALE,
        bg_alpha=MOTION_BG_ALPHA,
    )

    known_encodings, known_names, vault = load_family_from_vault(
        cipher=vault_cipher
    )

    # Unknown-face templates recorded since the last run, so repeat
    # intruders can be flagged even before the family DB is consulted.
    unknown_encodings, unknown_infos = vault.load_unknown()
    print(
        f"Unknown-face records in vault: {len(unknown_encodings)}"
    )

    print(
        "\n--------------------------------\n"
        "SMART CCTV\n"
        "--------------------------------\n"
        f"Family face samples: {len(known_encodings)}\n"
        f"Recognition tolerance: {FACE_TOLERANCE}\n"
        f"Unknown delay: {UNKNOWN_DELAY_SECONDS}s\n"
        f"Security modes (Nepal time): "
        f"day siren {SIREN_DAY_DURATION}s, "
        f"night mode from {NIGHT_START_HOUR:02d}:00 "
        f"siren {SIREN_NIGHT_DURATION}s\n"
        "--------------------------------\n"
    )

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

    def _in_button(pos, rect) -> bool:
        """True when *pos* lies inside the button *rect* (x0, y0, x1, y1)."""
        if pos is None or rect is None:
            return False
        px, py = pos
        x0, y0, x1, y1 = rect
        return x0 <= px <= x1 and y0 <= py <= y1

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
            enrolled = run_enrollment(camera, vault=vault)
        except Exception as err:
            print(f"[ENROLL] Error during enrollment: {err}")
            enrolled = False

        if enrolled:
            # Pick up the new templates immediately without a restart
            known_encodings, known_names, _ = load_family_from_vault(
                cipher=vault_cipher
            )
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

            # Mirror the view so it behaves like a mirror — raising your
            # right hand shows on the right side of the screen. Every
            # downstream step (detection, zones, overlays) works on the
            # flipped frame, so what you see is exactly what is processed.
            if MIRROR_DISPLAY:
                frame = cv2.flip(frame, 1)

            # ── FPS measurement (smoothed) ──
            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now
            if dt > 0:
                # Standard EMA; seeding from 0.0 is fine — the first
                # reading simply gets 10% weight and it converges fast.
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

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

                # Every confirmed stranger's template is sealed into the
                # encrypted unknown-faces table, so repeat intruders are
                # recognised and their sighting count grows.
                if name == "UNKNOWN" and unknown_count >= UNKNOWN_CONFIRMATIONS:
                    try:
                        vault.record_unknown(
                            encoding,
                            metadata={
                                "mode": (
                                    "NIGHT" if is_night_mode() else "DAY"
                                ),
                            },
                        )
                    except Exception as err:
                        print(f"Vault write failed: {err}")

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
                fleft, ftop, fright, fbottom = (
                    int(c / DETECTION_SCALE)
                    for c in (left, top, right, bottom)
                )

                if final_name == "UNKNOWN":
                    unknown_faces.append(track.last_seen)
                    # No label on the box: the red 'UNKNOWN PERSON
                    # DETECTED' banner is the single unknown indicator.
                    displayed_faces.append(
                        ((fleft, ftop, fright, fbottom),
                         "", (0, 0, 255), conf)
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
            night_mode = is_night_mode()
            if night_mode:
                active_delay = NIGHT_UNKNOWN_DELAY_SECONDS
            elif human_seen:
                active_delay = UNKNOWN_HUMAN_DELAY_SECONDS
            else:
                active_delay = UNKNOWN_DELAY_SECONDS

            if unknown_faces:

                unknown_count += 1

                # Only count an unknown after several consecutive frames

                if unknown_count >= UNKNOWN_CONFIRMATIONS:

                    now = time.time()

                    if unknown_start is None:

                        unknown_start = now

                        print("\nUNKNOWN PERSON CONFIRMED")
                        logger.warning("Unknown person confirmed")
                        log_event("UNKNOWN_CONFIRMED")

                    elapsed = now - unknown_start
                    remaining = max(0, active_delay - elapsed)

                    # Save a snapshot every few seconds

                    if now - last_snapshot >= SNAPSHOT_INTERVAL:

                        timestamp = nepal_now().strftime("%Y%m%d_%H%M%S")
                        path = os.path.join(
                            SNAPSHOT_DIR,
                            f"unknown_{timestamp}.jpg"
                        )

                        cv2.imwrite(path, frame)

                        log_event("UNKNOWN_SNAPSHOT", "UNKNOWN", path)

                        last_snapshot = now

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
                        or now - siren.last_stop >= SIREN_RETRIGGER_COOLDOWN
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
                            f"mode={'NIGHT' if night_mode else 'DAY'},"
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

            draw_mode(frame, night_mode)

            # Show overall system status (alarm on or ok)

            draw_status(frame, siren.is_active)

            # Show the live FPS counter

            if SHOW_FPS:
                draw_fps(frame, fps)

            # Clickable '+ ADD FAMILY' button (bottom right). Hovering
            # brightens it; the rect is hit-tested against mouse clicks.
            hover = _in_button(tuple(mouse_pos), button_rect)
            button_rect = draw_add_button(frame, hover)

            # Show the current camera frame

            cv2.imshow(
                "Smart CCTV Security",
                frame
            )

            # Keyboard controls: q quits, s stops the siren,
            # t plays a short siren self-test,
            # a opens the add-family-member flow

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                running = False

            elif key == ord("s"):

                siren.stop()

            elif key == ord("t"):

                # Audible self-test on demand — safe: auto-stops after
                # STARTUP_SIREN_TEST_DURATION and never affects real
                # alarm state or the re-trigger cooldown.
                siren.self_test(duration=STARTUP_SIREN_TEST_DURATION)

            elif key == ord("a"):

                click_pos = None  # ignore any stale click
                _open_enrollment()

            # Mouse click on the '+ ADD FAMILY' button
            elif click_pos is not None:
                if _in_button(click_pos, button_rect):
                    _open_enrollment()
                click_pos = None

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Shut down cleanly when the loop ends

        siren.stop()

        camera.release()

        cv2.destroyAllWindows()

        vault.close()

        print("\nSmart CCTV stopped safely.")


if __name__ == "__main__":
    main()
