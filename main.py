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
    FACE_TOLERANCE,
    UNKNOWN_CONFIRMATIONS,
    UNKNOWN_DELAY_SECONDS,
    NIGHT_UNKNOWN_DELAY_SECONDS,
    ALLOWED_START_HOUR,
    ALLOWED_END_HOUR,
    SNAPSHOT_INTERVAL,
    DETECTION_SCALE,
    TRACKING_SKIP_FRAMES,
    FAMILY_DIR,
    SNAPSHOT_DIR,
    LOG_DIR,
    ANIMAL_DETECTION_ENABLED,
    UNKNOWN_HUMAN_DELAY_SECONDS,
)

from cctv.enhance import enhance_frame
from cctv.faces import (
    load_family_database,
    recognize_face,
    detect_faces_enhanced,
)
from cctv.tracking import match_tracks
from cctv.siren import Siren
from cctv.yolo import ObjectDetector
from cctv.storage import initialize_database, log_event, logger


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

    siren = Siren()
    detector = ObjectDetector(enabled=ANIMAL_DETECTION_ENABLED)

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

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    # Counters to track unknown detection
    unknown_count = 0
    unknown_start = None

    last_snapshot = 0

    tracked_faces = {}
    frame_counter = 0

    running = True

    # Main camera loop
    while running:

        ret, frame = camera.read()

        if not ret:

            print("WARNING: Camera frame unavailable.")

            time.sleep(0.5)

            continue

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

        # Full-resolution frame for CNN fallback
        rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all faces and their encodings in this frame.
        # Skip detection on non-detection frames, rely on track persistence.
        if frame_counter % TRACKING_SKIP_FRAMES == 0:
            locations, encodings = detect_faces_enhanced(
                rgb_frame, rgb_full
            )
        else:
            locations, encodings = [], []

        # ── Recognize each detected face ──
        raw_faces = []  # (location, name, confidence, encoding)

        for encoding, location in zip(encodings, locations):
            name, distance, confidence = recognize_face(
                encoding,
                known_encodings,
                known_names
            )
            raw_faces.append((location, name, confidence, encoding))

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
        for ((lx, ty, rx, by), label, color, conf) in displayed_faces:
            cv2.rectangle(frame, (lx, ty), (rx, by), color, 2)
            cv2.putText(frame, label, (lx, max(30, ty - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Confidence bar (green/amber/red)
            bar_len = int((rx - lx) * conf)
            bar_color = (
                (0, 255, 0) if conf > 0.6
                else (0, 255, 255) if conf > 0.3
                else (0, 0, 255)
            )
            cv2.rectangle(
                frame, (lx, by + 6), (lx + bar_len, by + 14),
                bar_color, -1
            )

        frame_counter += 1

        # Handle unknown faces: start timer, save snapshots, raise alarm

        # Object detection explains the scene while an unknown face lingers:
        # animals suppress the siren, a confirmed human shortens the delay.
        animal_seen, human_seen = False, False
        if unknown_faces and ANIMAL_DETECTION_ENABLED:
            animal_seen, human_seen = detector.detect(frame)

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
                        "\n⚠ UNKNOWN PERSON CONFIRMED"
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

                cv2.putText(
                    frame,
                    f"UNKNOWN - {remaining:.1f}s",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

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

            text = (
                "Family: "
                +
                ", ".join(unique_people)
            )

            cv2.putText(
                frame,
                text,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Log family member sightings
            for person in unique_people:
                log_event("FAMILY_SIGHTING", person=person)

        # Show whether we're inside the allowed hours

        allowed = is_allowed_time()

        time_text = (
            "ALLOWED TIME"
            if allowed
            else "🌙 NIGHT MODE"
        )

        time_color = (
            (0, 255, 0)
            if allowed
            else (200, 150, 0)
        )

        cv2.putText(
            frame,
            time_text,
            (20, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            time_color,
            2
        )

        # Show overall system status (alarm on or ok)

        status = (
            "ALARM ACTIVE"
            if siren.active
            else "SYSTEM OK"
        )

        status_color = (
            (0, 0, 255)
            if siren.active
            else (0, 255, 0)
        )

        cv2.putText(
            frame,
            status,
            (20, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2
        )

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
