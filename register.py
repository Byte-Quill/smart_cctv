"""Family member registration: guided auto-capture of quality face photos.

This is how the system learns who belongs in the house. Run it, type a
name, and sit in front of the camera — it captures a set of photos that
main.py later loads as that person's reference encodings.

How it works
------------
1. You enter a name. A folder ``family/<Name>/`` is created to hold that
   person's photos (one folder per person = one identity).
2. The camera opens and watches for a face. A photo is only saved when it
   passes every quality gate in cctv/quality.py:
       - face large enough      (MIN_REG_FACE_SIZE)
       - good lighting          (MIN_BRIGHTNESS..MAX_BRIGHTNESS)
       - sharp, not blurry      (BLUR_THRESHOLD)
       - a new pose             (MIN_ENCODING_DISTANCE from prior shots)
3. Captures happen automatically after a few stable frames; you just hold
   still and vary your angle slightly between shots.
4. After TARGET_PHOTOS captures (or pressing Q), the session ends.

Why quality matters
-------------------
Recognition in main.py compares live faces against these photos. Ten
sharp, varied poses give a robust identity; blurry or duplicate shots
weaken it. That is why this tool rejects bad frames instead of saving
them.

Re-running with the same name adds more photos to the existing folder.
Delete the folder to remove a person.
"""

import os
import time

import cv2
import face_recognition
import numpy as np

from config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    FAMILY_DIR,
    AUTO_CAPTURE_STABLE_FRAMES,
    TARGET_PHOTOS,
)

from cctv.quality import (
    estimate_blur,
    brightness_ok,
    blur_ok,
    face_large_enough,
    compute_encoding,
    load_existing_encodings,
    is_duplicate_pose,
)


def main() -> None:

    print("=" * 52)
    print("        SMART CCTV  —  FAMILY REGISTRATION")
    print("=" * 52)

    name = input("\nEnter family member name: ").strip()
    if not name:
        print("ERROR: Name cannot be empty.")
        return

    safe_name = "".join(
        c for c in name if c.isalnum() or c in (" ", "_", "-")
    ).strip()
    if not safe_name:
        print("ERROR: Invalid name (use letters, numbers, spaces, dashes).")
        return

    folder = os.path.join(FAMILY_DIR, safe_name)
    os.makedirs(folder, exist_ok=True)

    # Load any existing photos of this person (re-registration)
    existing_encodings = load_existing_encodings(folder)

    print(f"\nOpening camera (index {CAMERA_INDEX}) …")
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("ERROR: Camera could not be opened.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # ── state ──────────────────────────────────────────────────────────
    captured = 0
    skipped_blurry = 0
    skipped_dup = 0
    skipped_badlight = 0
    stable_frames = 0
    last_encoding = None

    print(f"\n  Target: {TARGET_PHOTOS} photos of {safe_name}")
    print("  Look directly at the camera and hold still.")
    print("  Good captures are taken automatically.")
    print("  Press  Q  to quit early.\n")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("WARNING: frame read failed — retrying …")
            time.sleep(0.5)
            continue

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── detect face (full-res for accuracy) ────────────────────────
        face_locs = face_recognition.face_locations(rgb, model="hog")

        face_encoding = None
        face_box = None
        quality_ok = False
        blur_val = 0.0
        brightness = gray.mean()

        if len(face_locs) > 0:
            # Pick the largest face in the frame
            sizes = [(b - t) for (t, r, b, l) in face_locs]
            best = int(np.argmax(sizes))
            top, right, bottom, left = face_locs[best]
            face_box = (top, right, bottom, left)
            face_h = bottom - top

            # Quality check chain: size → lighting → blur → duplicate pose.
            # A rejection is reported only on the first unstable frame so
            # the console stays quiet while the user adjusts.
            reject_reason = None

            if not face_large_enough(face_h):
                reject_reason = f"face too small ({face_h}px)"
            elif not brightness_ok(brightness):
                reject_reason = f"bad lighting ({brightness:.0f})"
                skipped_badlight += 1
            else:
                blur_val = estimate_blur(gray[top:bottom, left:right])
                if not blur_ok(blur_val):
                    reject_reason = f"blurry ({blur_val:.0f})"
                    skipped_blurry += 1
                else:
                    face_roi = rgb[top:bottom, left:right]
                    enc = (
                        compute_encoding(face_roi)
                        if face_roi.size > 0 else None
                    )
                    if face_roi.size == 0:
                        reject_reason = "empty face region"
                    elif enc is None:
                        reject_reason = "could not encode face"
                    else:
                        # Compare against all known encodings plus the
                        # last captured one.
                        all_encs = existing_encodings
                        if last_encoding is not None:
                            all_encs = all_encs + [last_encoding]

                        if is_duplicate_pose(enc, all_encs):
                            reject_reason = "duplicate pose (already captured)"
                            skipped_dup += 1
                        else:
                            face_encoding = enc
                            quality_ok = True

            if reject_reason is not None and stable_frames == 0:
                print(f"  [SKIP]  {reject_reason}")

        else:
            if stable_frames % 30 == 0:
                print("  [WAIT]  no face detected  ", end="\r", flush=True)

        # ── stable-frame auto-capture ──────────────────────────────────
        if quality_ok:
            stable_frames += 1
            if stable_frames >= AUTO_CAPTURE_STABLE_FRAMES:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                fname = f"face_{captured + 1:02d}_{timestamp}.jpg"
                path = os.path.join(folder, fname)
                cv2.imwrite(path, frame)

                captured += 1
                existing_encodings.append(face_encoding)
                last_encoding = face_encoding
                stable_frames = 0
                print(f"  [OK]  captured {captured}/{TARGET_PHOTOS}  ({fname})")
        else:
            stable_frames = 0

        # ── draw overlay ───────────────────────────────────────────────
        if face_box is not None:
            t, r, b, l = face_box
            box_color = (0, 255, 0) if quality_ok else (0, 0, 255)
            cv2.rectangle(display, (l, t), (r, b), box_color, 2)

            # quality score label
            score_text = f"Sharp:{blur_val:.0f} Bright:{brightness:.0f}"
            cv2.putText(display, score_text, (l, t - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            if quality_ok:
                pct = min(stable_frames / AUTO_CAPTURE_STABLE_FRAMES * 100, 100)
                bar_w = int((r - l) * pct / 100)
                cv2.rectangle(display, (l, b + 6), (l + bar_w, b + 12), (0, 255, 0), -1)

        # ── info text ──────────────────────────────────────────────────
        lines = [
            f"Photos: {captured}/{TARGET_PHOTOS}",
            f"Rejected: blurry={skipped_blurry}  dup={skipped_dup}  light={skipped_badlight}",
            f"{'CAPTURING' if quality_ok else 'waiting'}  |  Q = quit",
        ]
        for i, line in enumerate(lines):
            cv2.putText(display, line, (20, 40 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if quality_ok else (100, 100, 100),
                        2)

        cv2.imshow("Family Registration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print(f"\n  Quit early. Saved {captured} photos.")
            break

        if captured >= TARGET_PHOTOS:
            print(f"\n  [OK]  Successfully registered {safe_name} ({captured} photos).")
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
