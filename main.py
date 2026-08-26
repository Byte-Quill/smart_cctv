"""Smart CCTV — main monitoring loop.

Pipeline: enhance -> detect -> recognize -> track -> alarm/log.
All building blocks live in the cctv/ package.
"""

import os
import time

import cv2

from datetime import datetime

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    FACE_TOLERANCE,
    UNKNOWN_CONFIRMATIONS,
    UNKNOWN_DELAY_SECONDS,
    NIGHT_UNKNOWN_DELAY_SECONDS,
    ALLOWED_START_HOUR,
    ALLOWED_END_HOUR,
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
)
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


# Create needed folders if missing
os.makedirs(FAMILY_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# True only during the allowed hours of the day
def is_allowed_time():

    hour = datetime.now().hour

    return (
        ALLOWED_START_HOUR
        <= hour
        <
        ALLOWED_END_HOUR
    )


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
        f"Allowed time: "
        f"{ALLOWED_START_HOUR:02d}:00 - "
        f"{ALLOWED_END_HOUR:02d}:00"
    )

    print("--------------------------------\n")

    # Open the camera and set its resolution
    camera = cv2.VideoCapture(CAMERA_INDEX)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    # Keep the driver queue at a single frame so the display never lags
    # behind real time while the pipeline is busy.
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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

    # Main camera loop
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

            camera = cv2.VideoCapture(CAMERA_INDEX)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
        # fallback is enabled — otherwise this is wasted work every frame.
        rgb_full = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ENABLE_CNN_FALLBACK else None
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

        # Update tracks with smoothing and frame-skip
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

        # Shorter delay outside allowed hours (night mode),
        # fastest delay when YOLO confirms a human
        if not is_allowed_time():
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

                    timestamp = datetime.now().strftime(
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
                # unless an animal (and no human) explains the scene

                animal_only = animal_seen and not human_seen

                if (
                    elapsed
                    >= active_delay
                ) and is_allowed_time() and not animal_only:

                    siren.start()

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

        # Show whether we're inside the allowed hours

        draw_mode(frame, is_allowed_time())

        # Show overall system status (alarm on or ok)

        draw_status(frame, siren.is_active)

        # Show the live FPS counter

        if SHOW_FPS:
            draw_fps(frame, fps)

        # Show the current camera frame

        cv2.imshow(
            "Smart CCTV Security",
            frame
        )

        # Keyboard controls: q quits, s stops the siren

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            running = False

        elif key == ord("s"):

            siren.stop()

    # Shut down cleanly when the loop ends

    siren.stop()

    camera.release()

    cv2.destroyAllWindows()

    print("\nSmart CCTV stopped safely.")


if __name__ == "__main__":
    main()
