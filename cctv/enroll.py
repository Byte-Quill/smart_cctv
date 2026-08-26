"""In-app family enrollment: add a member without leaving the live view.

Opened from the main window — click the "+ ADD FAMILY" button (bottom
right) or press ``a``. This module then takes over the camera and walks
the user through two screens, all inside the same window:

1. NAME — type the person's name on the keyboard. Enter confirms,
   Backspace edits, Esc cancels.

2. CAPTURE — fully automatic face collection. The user simply looks at
   the camera; while the face is sharp, well lit and big enough, a
   photo is saved automatically every ENROLL_CAPTURE_INTERVAL seconds.
   Pose variety comes from a duplicate-pose check: a photo too similar
   to one already taken is rejected with a "turn your face" hint, so
   the saved set naturally covers different angles. Detection runs on a
   downscaled frame and the expensive encoding is computed only at the
   moment of capture, so the live video stays smooth.

The quality gates are exactly the ones ``register.py`` uses
(cctv/quality.py), so both doors build equally strong reference sets.
Photos land in ``family/<Name>/`` like the CLI tool; ``main.py`` reloads
the family database when enrollment finishes.
"""

import os
import time

import cv2
import face_recognition
import numpy as np

from config import (
    FAMILY_DIR,
    TARGET_PHOTOS,
    ENROLL_CAPTURE_INTERVAL,
    ENROLL_DETECT_SCALE,
    BLUR_THRESHOLD,
)

from cctv.quality import (
    estimate_blur,
    brightness_ok,
    face_large_enough,
    compute_encoding,
    load_existing_encodings,
    is_duplicate_pose,
)
from cctv.timeutil import nepal_now


WINDOW_NAME = "Smart CCTV Security"

_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_GRAY = (160, 160, 160)
_AMBER = (0, 255, 255)

_NAME_CHARS_MAX = 24
_THUMB_SIZE = 56


# ----------------------------------------------------------------------
# Small pure helpers (unit-tested in tests/test_enroll.py)
# ----------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    """Keep only letters, digits, spaces, dashes and underscores."""
    return "".join(
        c for c in name if c.isalnum() or c in (" ", "_", "-")
    ).strip()


def zone_index(x: int, frame_width: int) -> int:
    """Map a pixel column to a zone: 0 = left, 1 = center, 2 = right."""
    third = frame_width / 3.0
    if x < third:
        return 0
    if x < 2 * third:
        return 1
    return 2


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------

def _dim(frame, keep: float = 0.35):
    """Return a darkened copy of *frame* for overlay screens."""
    black = np.zeros_like(frame)
    return cv2.addWeighted(frame, keep, black, 1.0 - keep, 0)


def _center_text(display, text, y, scale, color, thickness=2):
    """Draw *text* horizontally centered at height *y*."""
    (w, _), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x = (display.shape[1] - w) // 2
    cv2.putText(
        display, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness
    )


def _thumbnail(frame, face_box, size: int = _THUMB_SIZE):
    """Crop the captured face and shrink it to a square preview."""
    top, right, bottom, left = face_box
    crop = frame[max(0, top):bottom, max(0, left):right]
    if crop.size == 0:
        crop = frame[:size, :size]
    return cv2.resize(crop, (size, size))


# ----------------------------------------------------------------------
# Screen 1 — name entry
# ----------------------------------------------------------------------

def run_name_entry(camera):
    """Keyboard name entry. Returns the sanitized name, or None if cancelled."""
    typed = []
    error = ""

    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue

        display = _dim(frame)
        h, w = display.shape[:2]
        cx = w // 2

        _center_text(display, "ADD FAMILY MEMBER", h // 2 - 90, 1.0, _AMBER)
        _center_text(
            display, "Type the person's name", h // 2 - 45, 0.6, _GRAY, 1
        )

        # Input box with the typed name and a blinking cursor
        name_so_far = "".join(typed)
        cursor = "_" if (time.time() % 1.0) < 0.6 else " "
        shown = name_so_far + cursor
        (nw, _), _ = cv2.getTextSize(
            shown, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
        )
        box_w = max(nw + 40, min(320, w - 40))
        x0, x1 = cx - box_w // 2, cx + box_w // 2
        y0, y1 = h // 2 - 25, h // 2 + 25
        cv2.rectangle(display, (x0, y0), (x1, y1), (90, 90, 90), -1)
        cv2.rectangle(display, (x0, y0), (x1, y1), _WHITE, 1)
        cv2.putText(
            display, shown, (cx - nw // 2, h // 2 + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, _WHITE, 2
        )

        if error:
            _center_text(display, error, y1 + 30, 0.55, _RED, 1)

        _center_text(
            display,
            "ENTER = confirm    BACKSPACE = delete    ESC = cancel",
            h - 25, 0.55, _GRAY, 1
        )

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # Esc → cancel
            return None

        if key in (10, 13):  # Enter → confirm
            safe = sanitize_name("".join(typed))
            if safe:
                return safe
            error = "Please type a name (letters, numbers, spaces, - _)"

        elif key == 8:  # Backspace
            if typed:
                typed.pop()
            error = ""

        elif 32 <= key <= 126 and len(typed) < _NAME_CHARS_MAX:
            ch = chr(key)
            if ch.isalnum() or ch in (" ", "-", "_"):
                typed.append(ch)
                error = ""


# ----------------------------------------------------------------------
# Screen 2 — automatic capture (no zones, just look at the camera)
# ----------------------------------------------------------------------

def run_capture(camera, safe_name: str) -> int:
    """Automatic face capture. Returns the number of photos saved.

    The user only has to look at the camera and slowly turn their face.
    While the face passes the quality gates, one photo is saved every
    ENROLL_CAPTURE_INTERVAL seconds. A duplicate-pose check rejects
    photos too similar to ones already kept, which naturally collects
    the varied angles recognition needs.

    Performance: detection runs on a frame downscaled by
    ENROLL_DETECT_SCALE, and the expensive 128-d encoding is computed
    ONLY at the moment a capture fires — never per frame — so the live
    video stays smooth even on low-power hardware.
    """
    folder = os.path.join(FAMILY_DIR, safe_name)
    os.makedirs(folder, exist_ok=True)

    known_encs = load_existing_encodings(folder)  # re-registration support
    session_encs = []
    thumbnails = []

    captured = 0
    last_capture = 0.0  # time of last save OR rejected duplicate
    status = "Look at the camera"
    status_color = _WHITE

    scale = ENROLL_DETECT_SCALE
    # Blur is measured on the downscaled frame, so scale the threshold
    # the same way cctv/faces.py does for the live pipeline.
    min_blur = BLUR_THRESHOLD * (scale ** 2)

    print(
        f"[ENROLL] Capturing up to {TARGET_PHOTOS} photos of "
        f"'{safe_name}' (auto-capture every {ENROLL_CAPTURE_INTERVAL}s)."
    )

    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue

        display = frame.copy()
        h, w = display.shape[:2]

        # ── cheap detection on a downscaled frame (keeps video smooth) ──
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_small, model="hog")

        quality_ok = False
        face_box = None  # full-resolution box for display/saving

        if face_locs:
            sizes = [(b - t) for (t, r, b, l) in face_locs]
            best = int(np.argmax(sizes))
            st, sr, sb, sl = face_locs[best]
            top = int(st / scale)
            right = min(int(sr / scale), w)
            bottom = min(int(sb / scale), h)
            left = int(sl / scale)
            face_box = (top, right, bottom, left)

            face_h = bottom - top
            gray_small = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY)
            brightness = gray_small.mean()
            blur_val = estimate_blur(gray_small[st:sb, sl:sr])

            # Quality gate chain (size → light → blur)
            if not face_large_enough(face_h):
                status, status_color = "Come closer (face too small)", _AMBER
            elif not brightness_ok(brightness):
                status, status_color = "Fix the lighting", _AMBER
            elif blur_val < min_blur:
                status, status_color = "Hold still (blurry)", _AMBER
            else:
                quality_ok = True
                status, status_color = (
                    "Good! Slowly turn your face left and right", _GREEN
                )
        else:
            status, status_color = "No face found — look at the camera", _GRAY

        # ── auto-capture: good face + interval elapsed ──
        # The expensive encoding + duplicate check happen ONLY here, at
        # capture time — not on every frame.
        now = time.time()
        if quality_ok and (now - last_capture) >= ENROLL_CAPTURE_INTERVAL:
            top, right, bottom, left = face_box
            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_roi = rgb_full[top:bottom, left:right]
            try:
                enc = (
                    compute_encoding(face_roi)
                    if face_roi.size > 0 else None
                )
            except Exception:
                enc = None

            if enc is None:
                status, status_color = "Face not clear, adjust", _AMBER
                last_capture = now  # wait a full interval before retrying
            elif is_duplicate_pose(enc, known_encs + session_encs):
                status, status_color = "Same pose — turn your face", _AMBER
                last_capture = now
            else:
                stamp = nepal_now().strftime("%Y%m%d_%H%M%S")
                fname = (
                    f"face_{len(known_encs) + captured + 1:02d}_{stamp}.jpg"
                )
                cv2.imwrite(os.path.join(folder, fname), frame)

                captured += 1
                session_encs.append(enc)
                thumbnails.append(_thumbnail(frame, face_box))
                last_capture = now
                print(
                    f"[ENROLL] captured {captured}/{TARGET_PHOTOS} ({fname})"
                )

        # ── face box + countdown bar to the next photo ──
        if face_box is not None:
            t, r, b, l = face_box
            color = _GREEN if quality_ok else _AMBER
            cv2.rectangle(display, (l, t), (r, b), color, 2)
            if quality_ok:
                pct = min(
                    1.0,
                    (now - last_capture) / ENROLL_CAPTURE_INTERVAL
                )
                bar_w = int((r - l) * pct)
                cv2.rectangle(
                    display, (l, b + 6), (l + bar_w, b + 12), _GREEN, -1
                )

        # ── header: who, progress, live guidance ──
        cv2.putText(
            display, f"Registering: {safe_name}", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, _WHITE, 2
        )
        cv2.putText(
            display, f"Photos: {captured}/{TARGET_PHOTOS}", (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, _GREEN, 2
        )
        (sw, _), _ = cv2.getTextSize(
            status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.putText(
            display, status, (w - sw - 15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
        )

        # ── captured thumbnails row (bottom center) ──
        if thumbnails:
            gap = 8
            total_w = (
                len(thumbnails) * _THUMB_SIZE
                + (len(thumbnails) - 1) * gap
            )
            x = max((w - total_w) // 2, 5)
            y = h - _THUMB_SIZE - 30
            for thumb in thumbnails:
                if x + _THUMB_SIZE > w:
                    break
                cv2.rectangle(
                    display,
                    (x - 2, y - 2),
                    (x + _THUMB_SIZE + 2, y + _THUMB_SIZE + 2),
                    _GREEN, 1
                )
                display[y:y + _THUMB_SIZE, x:x + _THUMB_SIZE] = thumb
                x += _THUMB_SIZE + gap

        cv2.putText(
            display,
            "Just look at the camera — photos are taken automatically",
            (15, h - 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GRAY, 1
        )
        cv2.putText(
            display, "Q / ESC = finish and keep photos", (15, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _GRAY, 1
        )

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or captured >= TARGET_PHOTOS:
            break

    return captured


# ----------------------------------------------------------------------
# Entry point used by main.py
# ----------------------------------------------------------------------

def run_enrollment(camera) -> bool:
    """Run the full in-app enrollment flow on *camera*.

    Returns True when at least one photo was saved (main.py then reloads
    the family database). The caller keeps owning the camera; this flow
    only borrows it until it returns.
    """
    print("\n[ENROLL] Add-family-member mode opened.")

    name = run_name_entry(camera)
    if not name:
        print("[ENROLL] Cancelled at name entry.")
        return False

    captured = run_capture(camera, name)

    if captured > 0:
        print(f"[ENROLL] Registered '{name}' with {captured} photos.")
        return True

    print(f"[ENROLL] No photos captured for '{name}'.")
    return False
