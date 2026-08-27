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


# Minimum FULL-RES face height accepted from the CNN fallback. The int()
# truncation is deliberate (e.g. scale 0.4 -> int(2.5) = 2): it expresses
# the HOG-size gate in full-resolution pixels.
_CNN_MIN_FACE_SIZE = MIN_FACE_SIZE * int(1 / DETECTION_SCALE)


def _drop_tiny(locations):
    """Keep only faces at least MIN_FACE_SIZE px tall (detection scale)."""
    return [
        (t, r, b, l) for (t, r, b, l) in locations
        if b - t >= MIN_FACE_SIZE
    ]


def _cnn_fallback(rgb_frame_full):
    """Run CNN on the full-res frame; return locations in small-frame coords.

    Returns [] when the CNN model is unavailable.
    """
    try:
        locations = face_recognition.face_locations(
            rgb_frame_full, model="cnn"
        )
    except Exception:
        return []  # CNN model may not be available; silently fall back

    # Scale CNN locations DOWN to small-frame coords
    return [
        (
            int(t * DETECTION_SCALE),
            int(r * DETECTION_SCALE),
            int(b * DETECTION_SCALE),
            int(l * DETECTION_SCALE),
        )
        for (t, r, b, l) in locations
        if b - t >= _CNN_MIN_FACE_SIZE
    ]


def _drop_blurry(rgb_frame_small, locations):
    """Drop blurry faces: their encodings jitter and cause mis-recognition.

    Blur is measured on the small frame's grayscale (cheap) with the
    registration threshold scaled to the detection resolution.
    """
    gray_small = cv2.cvtColor(rgb_frame_small, cv2.COLOR_RGB2GRAY)
    min_blur = BLUR_THRESHOLD * (DETECTION_SCALE ** 2)
    sharp = []
    for (t, r, b, l) in locations:
        roi = gray_small[t:b, l:r]
        if roi.size == 0:
            continue
        if cv2.Laplacian(roi, cv2.CV_64F).var() >= min_blur:
            sharp.append((t, r, b, l))
    return sharp


# Enhanced face detection: HOG first, then CNN fallback if nothing found.
# Returns (locations, encodings) in the SMALL-frame coordinate system.
def detect_faces_enhanced(rgb_frame_small, rgb_frame_full):

    # Try HOG on the small (enhanced) frame
    locations = _drop_tiny(
        face_recognition.face_locations(rgb_frame_small, model="hog")
    )

    # If HOG found nothing AND CNN is enabled, try CNN on the full-res frame
    if not locations and ENABLE_CNN_FALLBACK:
        locations = _cnn_fallback(rgb_frame_full)

    # Quality gate before encoding
    if locations:
        locations = _drop_blurry(rgb_frame_small, locations)

    encodings = face_recognition.face_encodings(
        rgb_frame_small, locations, num_jitters=RECOGNITION_JITTERS
    )
    return locations, encodings
