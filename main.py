import cv2
import face_recognition
import numpy as np
import os
import time
import sqlite3
import logging

from datetime import datetime

import pygame

from config import (
    CAMERA_INDEX,
    FACE_TOLERANCE,
    REQUIRED_CONFIRMATIONS,
    UNKNOWN_CONFIRMATIONS,
    UNKNOWN_DELAY_SECONDS,
    ALLOWED_START_HOUR,
    ALLOWED_END_HOUR,
    SNAPSHOT_INTERVAL,
    FRAME_SCALE,
    FAMILY_DIR,
    SNAPSHOT_DIR,
    LOG_DIR,
    SIREN_FILE
)


# Create needed folders if missing
os.makedirs(FAMILY_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# Set up file logging for security events
logging.basicConfig(
    filename=os.path.join(
        LOG_DIR,
        "security.log"
    ),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("SmartCCTV")


# SQLite database that stores all events
DATABASE = os.path.join(
    LOG_DIR,
    "events.db"
)


# Create the events table if it does not exist yet
def initialize_database():

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            person TEXT,
            snapshot TEXT
        )
    """)

    connection.commit()
    connection.close()


# Save one event to the database and the log file
def log_event(
    event_type,
    person=None,
    snapshot=None
):

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO events
        (timestamp, event_type, person, snapshot)
        VALUES (?, ?, ?, ?)
        """,
        (
            timestamp,
            event_type,
            person,
            snapshot
        )
    )

    connection.commit()
    connection.close()

    logger.info(
        "%s | %s | %s",
        event_type,
        person,
        snapshot
    )


# Read all registered family photos and build their face encodings
def load_family_database():

    encodings = []
    names = []

    print("\nLoading family database...\n")

    if not os.path.exists(FAMILY_DIR):
        return encodings, names

    for person in sorted(
        os.listdir(FAMILY_DIR)
    ):

        person_path = os.path.join(
            FAMILY_DIR,
            person
        )

        if not os.path.isdir(person_path):
            continue

        print(f"Loading: {person}")

        for filename in sorted(
            os.listdir(person_path)
        ):

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            image_path = os.path.join(
                person_path,
                filename
            )

            try:

                image = face_recognition.load_image_file(
                    image_path
                )

                locations = face_recognition.face_locations(
                    image,
                    model="hog"
                )

                # Photos must have exactly one face to be usable
                if len(locations) != 1:

                    print(
                        f"  SKIPPED {filename}: "
                        f"expected exactly 1 face, "
                        f"found {len(locations)}"
                    )

                    continue

                face_encoding = (
                    face_recognition.face_encodings(
                        image,
                        locations
                    )[0]
                )

                encodings.append(
                    face_encoding
                )

                names.append(
                    person
                )

                print(
                    f"  Loaded {filename}"
                )

            except Exception as error:

                print(
                    f"  Error: {filename}: {error}"
                )

    return encodings, names


# Compare a detected face against all known faces, return name or UNKNOWN
def recognize_face(
    face_encoding,
    known_encodings,
    known_names
):

    if not known_encodings:
        return "UNKNOWN", None

    distances = face_recognition.face_distance(
        known_encodings,
        face_encoding
    )

    # Find the closest known face
    best_index = int(
        np.argmin(distances)
    )

    best_distance = float(
        distances[best_index]
    )

    # Only a match if the distance is small enough
    if best_distance <= FACE_TOLERANCE:

        return (
            known_names[best_index],
            best_distance
        )

    return "UNKNOWN", best_distance


# True only during the allowed hours of the day
def is_allowed_time():

    hour = datetime.now().hour

    return (
        ALLOWED_START_HOUR
        <= hour
        <
        ALLOWED_END_HOUR
    )


# Load the siren sound if it exists
pygame.mixer.init()

siren = None

try:

    siren = pygame.mixer.Sound(
        SIREN_FILE
    )

except Exception as error:

    print(
        "\nWARNING: Could not load siren:"
    )

    print(error)

    logger.warning(
        "Siren could not be loaded: %s",
        error
    )


siren_active = False


# Turn the siren on (loops forever until stopped)
def start_siren():

    global siren_active

    if siren_active:
        return

    print("\n🚨 SIREN ACTIVATED 🚨")

    logger.warning(
        "SIREN ACTIVATED"
    )

    if siren is not None:

        siren.play(-1)

    siren_active = True

    log_event(
        "SIREN_ON"
    )


# Turn the siren off
def stop_siren():

    global siren_active

    if not siren_active:
        return

    print("\nSiren stopped.")

    if siren is not None:
        siren.stop()

    siren_active = False

    logger.info(
        "SIREN STOPPED"
    )

    log_event(
        "SIREN_OFF"
    )


# Set up database and load known faces at startup
initialize_database()

known_encodings, known_names = (
    load_family_database()
)


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
camera = cv2.VideoCapture(
    CAMERA_INDEX
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

if not camera.isOpened():

    raise RuntimeError(
        "Could not open camera."
    )


# Counters to track family and unknown detection
family_candidate = None
family_count = 0

unknown_count = 0
unknown_start = None

last_snapshot = 0

running = True


# Main camera loop
while running:

    ret, frame = camera.read()

    if not ret:

        print(
            "WARNING: Camera frame unavailable."
        )

        time.sleep(0.5)

        continue


    # Shrink the frame so recognition runs faster
    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=FRAME_SCALE,
        fy=FRAME_SCALE
    )

    rgb_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )


    # Find all faces and their encodings in this frame
    locations = face_recognition.face_locations(
        rgb_frame,
        model="hog"
    )

    encodings = face_recognition.face_encodings(
        rgb_frame,
        locations
    )


    recognized_people = []

    unknown_faces = []


    # Classify every detected face
    for encoding, location in zip(
        encodings,
        locations
    ):

        name, distance = recognize_face(
            encoding,
            known_encodings,
            known_names
        )

        if name == "UNKNOWN":

            unknown_faces.append(
                location
            )

        else:

            recognized_people.append(
                name
            )


        # Convert coordinates back to the full-size frame

        top, right, bottom, left = location

        top = int(top / FRAME_SCALE)
        right = int(right / FRAME_SCALE)
        bottom = int(bottom / FRAME_SCALE)
        left = int(left / FRAME_SCALE)


        # Unknown faces are drawn red, family members green
        if name == "UNKNOWN":

            color = (0, 0, 255)

            label = "UNKNOWN"

        else:

            color = (0, 255, 0)

            label = name


        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            color,
            2
        )


        cv2.putText(
            frame,
            label,
            (left, max(30, top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )


    # Handle unknown faces: start timer, save snapshots, raise alarm

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
                UNKNOWN_DELAY_SECONDS
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


            # Trigger the siren once the delay has passed

            if (
                elapsed
                >= UNKNOWN_DELAY_SECONDS
            ):

                start_siren()


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


    # Show whether we're inside the allowed hours

    allowed = is_allowed_time()

    time_text = (
        "ALLOWED TIME"
        if allowed
        else "OUTSIDE ALLOWED TIME"
    )

    time_color = (
        (0, 255, 0)
        if allowed
        else (0, 0, 255)
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
        if siren_active
        else "SYSTEM OK"
    )

    status_color = (
        (0, 0, 255)
        if siren_active
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

        stop_siren()


# Shut down cleanly when the loop ends

stop_siren()

camera.release()

cv2.destroyAllWindows()

print("\nSmart CCTV stopped safely.")