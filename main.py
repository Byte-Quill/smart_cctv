import cv2
import face_recognition
import numpy as np
import os
import time
import sqlite3
import logging

from datetime import datetime
from collections import deque

import pygame

from config import (
    CAMERA_INDEX,
    FACE_TOLERANCE,
    REQUIRED_CONFIRMATIONS,
    UNKNOWN_CONFIRMATIONS,
    UNKNOWN_DELAY_SECONDS,
    NIGHT_UNKNOWN_DELAY_SECONDS,
    ALLOWED_START_HOUR,
    ALLOWED_END_HOUR,
    SNAPSHOT_INTERVAL,
    FRAME_SCALE,
    DETECTION_SCALE,
    MIN_FACE_SIZE,
    ENABLE_CNN_FALLBACK,
    ENSEMBLE_FRAMES,
    TRACKING_SKIP_FRAMES,
    TRACKING_SMOOTH_ALPHA,
    TRACKING_PATIENCE,
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
        return "UNKNOWN", None, 0.0

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

    # Compute confidence: 1.0 = perfect match, 0.0 = at threshold or worse
    confidence = max(0.0, min(1.0, 1.0 - (best_distance / FACE_TOLERANCE)))

    # Only a match if the distance is small enough
    if best_distance <= FACE_TOLERANCE:

        return (
            known_names[best_index],
            best_distance,
            confidence
        )

    return "UNKNOWN", best_distance, confidence


# ── Phase 3: Enhanced face detection ───────────────────────────────────

def detect_faces_enhanced(rgb_frame_small: np.ndarray, rgb_frame_full: np.ndarray):
    """Detect faces using HOG first, then CNN fallback if nothing found.
    Returns (locations, encodings) tuples in the SMALL-frame coordinate system.
    """
    # Try HOG on the small (enhanced) frame
    locations = face_recognition.face_locations(rgb_frame_small, model="hog")

    # Filter tiny detections
    filtered = []
    for (t, r, b, l) in locations:
        face_h = b - t
        if face_h >= MIN_FACE_SIZE:
            filtered.append((t, r, b, l))
    locations = filtered

    # If HOG found nothing AND CNN is enabled, try CNN on the full-res frame
    if len(locations) == 0 and ENABLE_CNN_FALLBACK:
        try:
            locations = face_recognition.face_locations(rgb_frame_full, model="cnn")
            # Scale CNN locations DOWN to small-frame coords
            scale_factor = DETECTION_SCALE  # CNN was run on full-res, need to scale
            scaled = []
            for (t, r, b, l) in locations:
                face_h = b - t
                if face_h >= MIN_FACE_SIZE * int(1 / DETECTION_SCALE):
                    st = int(t * DETECTION_SCALE)
                    sr = int(r * DETECTION_SCALE)
                    sb = int(b * DETECTION_SCALE)
                    sl = int(l * DETECTION_SCALE)
                    scaled.append((st, sr, sb, sl))
            locations = scaled
        except Exception:
            pass  # CNN model may not be available; silently fall back

    encodings = face_recognition.face_encodings(rgb_frame_small, locations)
    return locations, encodings


# ── Phase 4: Temporal ensemble tracking ────────────────────────────────

class FaceHistory:
    """Rolling classification history for one tracked face."""
    def __init__(self, window: int = ENSEMBLE_FRAMES):
        self.window = window
        self.classes: deque = deque(maxlen=window)
        self.confidences: deque = deque(maxlen=window)
        self.encoding_history: deque = deque(maxlen=window)

    def add(self, name: str, confidence: float, encoding: np.ndarray | None):
        self.classes.append(name)
        self.confidences.append(confidence)
        if encoding is not None:
            self.encoding_history.append(encoding)

    @property
    def majority_name(self) -> str:
        """Return the majority-vote name over the window."""
        if not self.classes:
            return "UNKNOWN"
        from collections import Counter
        counts = Counter(self.classes)
        return counts.most_common(1)[0][0]

    @property
    def avg_confidence(self) -> float:
        """Mean confidence over the window."""
        if not self.confidences:
            return 0.0
        return float(np.mean(self.confidences))


class FaceTrack:
    """Track state for one face: smoothed location, classification history, patience."""
    def __init__(self, location: tuple[int, int, int, int], name: str, confidence: float, encoding: np.ndarray | None):
        self.history = FaceHistory()
        self.history.add(name, confidence, encoding)
        # Smoothed location (full-resolution coords)
        self.smoothed = location
        self.patience = TRACKING_PATIENCE  # frames remaining before expiry
        self.last_seen = location

    def update(self, location: tuple[int, int, int, int], name: str, confidence: float, encoding: np.ndarray | None):
        self.history.add(name, confidence, encoding)
        # EMA smoothing on each coordinate
        a = TRACKING_SMOOTH_ALPHA
        self.smoothed = tuple(
            int(a * loc_coord + (1 - a) * smooth_coord)
            for loc_coord, smooth_coord in zip(location, self.smoothed)
        )
        self.last_seen = location
        self.patience = TRACKING_PATIENCE  # reset patience

    def decay_patience(self):
        self.patience -= 1

    @property
    def is_alive(self) -> bool:
        return self.patience > 0

    @property
    def majority_name(self) -> str:
        return self.history.majority_name

    @property
    def avg_confidence(self) -> float:
        return self.history.avg_confidence


TrackDict = dict[int, FaceTrack]


def _centroid(location) -> tuple[int, int]:
    top, right, bottom, left = location
    return ((left + right) // 2, (top + bottom) // 2)


def match_tracks(
    current_faces: list,
    prev_tracks: TrackDict,
    _frame_counter: int
) -> TrackDict:
    """
    Match current-frame face locations to existing tracks by centroid distance.
    - On detection frames: matches detections to tracks
    - On skip frames: decays patience, keeps tracks alive
    Returns updated {track_id: FaceTrack} dict.
    """
    new_tracks: TrackDict = {}

    # On skip frames, just decay all tracks and return
    if _frame_counter % TRACKING_SKIP_FRAMES != 0:
        for tid, track in prev_tracks.items():
            track.decay_patience()
            if track.is_alive:
                new_tracks[tid] = track
        return new_tracks

    matched = set()
    next_id = max(prev_tracks.keys(), default=-1) + 1

    for _loc, name, conf, enc in current_faces:
        best_id = -1
        best_dist = 60  # centroid distance threshold (detection-scale pixels)
        cx, cy = _centroid(_loc)
        for tid, track in prev_tracks.items():
            if tid in matched:
                continue
            tcx, tcy = _centroid(track.last_seen)
            d = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_id = tid
        if best_id >= 0:
            matched.add(best_id)
            track = prev_tracks[best_id]
            track.update(_loc, name, conf, enc)
            new_tracks[best_id] = track
        else:
            new_tracks[next_id] = FaceTrack(_loc, name, conf, enc)
            next_id += 1

    # Keep unmatched tracks alive (patience decay)
    for tid, track in prev_tracks.items():
        if tid not in matched:
            track.decay_patience()
            if track.is_alive:
                new_tracks[tid] = track

    return new_tracks


# Global tracking state
_tracked_faces: TrackDict = {}
_frame_counter = 0


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


# Counters to track unknown detection
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


    # ── Phase 2: Image enhancement ──
    # Auto gamma correction — brighten dark frames, dim overexposed
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    brightness = l.mean()
    # gamma mapping: 0.6-1.4 range, 1.0 at mid brightness ~128
    gamma = 1.0 + (0.4 * (128.0 - brightness) / 128.0)
    gamma = np.clip(gamma, 0.6, 1.4)
    lut = np.array([pow(i / 255.0, gamma) * 255 for i in range(256)]).astype("uint8")
    l = cv2.LUT(l, lut)

    # CLAHE contrast enhancement on luminance
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Light denoising (bilateral filter preserves edges)
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=30, sigmaSpace=30)

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

    # Find all faces and their encodings in this frame
    # Phase 5: Skip detection on non-detection frames
    if _frame_counter % TRACKING_SKIP_FRAMES == 0:
        locations, encodings = detect_faces_enhanced(rgb_frame, rgb_full)
    else:
        locations, encodings = [], []  # rely on track persistence


    # ── Phase 5: Track with smoothing and frame-skip ──
    raw_faces = []  # (location, name, confidence, encoding)

    for encoding, location in zip(encodings, locations):
        name, distance, confidence = recognize_face(
            encoding,
            known_encodings,
            known_names
        )
        raw_faces.append((location, name, confidence, encoding))

    # Update tracks
    prev = _tracked_faces
    _tracked_faces = match_tracks(raw_faces, prev, _frame_counter)

    # Build final classification from majority vote
    recognized_people = []
    unknown_faces = []
    displayed_faces = []  # (fullres_location, label, color, confidence)

    for tid, track in _tracked_faces.items():
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
            displayed_faces.append(((fleft, ftop, fright, fbottom), "UNKNOWN", (0, 0, 255), conf))
        else:
            recognized_people.append(final_name)
            conf_pct = int(conf * 100)
            displayed_faces.append(((fleft, ftop, fright, fbottom), f"{final_name} {conf_pct}%", (0, 255, 0), conf))

    # Draw all tracked faces with smoothed boxes
    for ((lx, ty, rx, by), label, color, conf) in displayed_faces:
        cv2.rectangle(frame, (lx, ty), (rx, by), color, 2)
        cv2.putText(frame, label, (lx, max(30, ty - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Confidence bar (green/amber/red)
        bar_len = int((rx - lx) * conf)
        bar_color = (0, 255, 0) if conf > 0.6 else (0, 255, 255) if conf > 0.3 else (0, 0, 255)
        cv2.rectangle(frame, (lx, by + 6), (lx + bar_len, by + 14), bar_color, -1)

    _frame_counter += 1


    # Handle unknown faces: start timer, save snapshots, raise alarm

    # Use shorter delay during OUTSIDE allowed hours (night mode)
    active_delay = NIGHT_UNKNOWN_DELAY_SECONDS if not is_allowed_time() else UNKNOWN_DELAY_SECONDS

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


            # Trigger the siren once the delay has passed

            if (
                elapsed
                >= active_delay
            ) && is_allowed_time():

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

        # Log family member sightings (rate-limited)
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