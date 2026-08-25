"""Family member registration: guided auto-capture of quality face photos."""

import os
import time

import cv2
import face_recognition
import numpy as np

from config import (
    CAMERA_INDEX,
    FAMILY_DIR,
    AUTO_CAPTURE_STABLE_FRAMES,
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


TARGET_PHOTOS = 10  # photos we want per person


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

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

            # Quality check chain ──
            # 1) minimum size
            if face_large_enough(face_h):
                # 2) brightness
                if brightness_ok(brightness):
                    # 3) blur
                    blur_val = estimate_blur(gray[top:bottom, left:right])
                    if blur_ok(blur_val):
                        # 4) encoding + duplicate check
                        face_roi = rgb[top:bottom, left:right]
                        if face_roi.size > 0:
                            enc = compute_encoding(face_roi)
                            if enc is not None:
                                # compare against all known encodings
                                all_encs = existing_encodings
                                # also compare against the last captured
                                if last_encoding is not None:
                                    all_encs = all_encs + [last_encoding]

                                if not is_duplicate_pose(enc, all_encs):
                                    face_encoding = enc
                                    quality_ok = True
                                else:
                                    # print only first time to reduce noise
                                    if stable_frames == 0:
                                        print("  [SKIP]  duplicate pose (already captured)")
                                    skipped_dup += 1
                            else:
                                if stable_frames == 0:
                                    print("  [SKIP]  could not encode face")
                        else:
                            if stable_frames == 0:
                                print("  [SKIP]  empty face region")
                    else:
                        if stable_frames == 0:
                            print(f"  [SKIP]  blurry ({blur_val:.0f})")
                        skipped_blurry += 1
                else:
                    if stable_frames == 0:
                        print(f"  [SKIP]  bad lighting ({brightness:.0f})")
                    skipped_badlight += 1
            else:
                if stable_frames == 0:
                    print(f"  [SKIP]  face too small ({face_h}px)")

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
                print(f"  ✓  captured {captured}/{TARGET_PHOTOS}  ({fname})")
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
            f"{'● AUTO-CAPTURING' if quality_ok else '○ waiting'}  |  Q = quit",
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
            print(f"\n  ✓  Successfully registered {safe_name} ({captured} photos).")
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
