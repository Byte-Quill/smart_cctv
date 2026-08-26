"""Face detection, encoding, and recognition."""

import os

import cv2
import face_recognition
import numpy as np

from config import (
    FAMILY_DIR,
    FACE_TOLERANCE,
    MATCH_MARGIN,
    MIN_FACE_SIZE,
    ENABLE_CNN_FALLBACK,
    DETECTION_SCALE,
    REGISTRATION_JITTERS,
    RECOGNITION_JITTERS,
    BLUR_THRESHOLD,
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
                        locations,
                        num_jitters=REGISTRATION_JITTERS
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

        # Ambiguity guard: if a second family member is almost equally
        # close, the match is a coin flip — refuse to guess and treat
        # the face as UNKNOWN instead of naming the wrong person.
        if len(distances) > 1:
            sorted_d = np.sort(distances)
            if float(sorted_d[1] - sorted_d[0]) < MATCH_MARGIN:
                return "UNKNOWN", best_distance, confidence

        return (
            known_names[best_index],
            best_distance,
            confidence
        )

    return "UNKNOWN", best_distance, confidence


# Enhanced face detection: HOG first, then CNN fallback if nothing found.
# Returns (locations, encodings) in the SMALL-frame coordinate system.
def detect_faces_enhanced(rgb_frame_small, rgb_frame_full):

    # Try HOG on the small (enhanced) frame
    locations = face_recognition.face_locations(
        rgb_frame_small, model="hog"
    )

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
            locations = face_recognition.face_locations(
                rgb_frame_full, model="cnn"
            )
            # Scale CNN locations DOWN to small-frame coords
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

    # Quality gate: blurry faces produce unstable encodings that cause
    # mis-recognition, so drop them before encoding. Blur is measured on
    # the small frame's grayscale (cheap) with the registration threshold
    # scaled to the detection resolution.
    if locations:
        gray_small = cv2.cvtColor(rgb_frame_small, cv2.COLOR_RGB2GRAY)
        min_blur = BLUR_THRESHOLD * (DETECTION_SCALE ** 2)
        sharp = []
        for (t, r, b, l) in locations:
            roi = gray_small[t:b, l:r]
            if roi.size == 0:
                continue
            if cv2.Laplacian(roi, cv2.CV_64F).var() >= min_blur:
                sharp.append((t, r, b, l))
        locations = sharp

    encodings = face_recognition.face_encodings(
        rgb_frame_small, locations, num_jitters=RECOGNITION_JITTERS
    )
    return locations, encodings
