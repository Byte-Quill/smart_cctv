import cv2
import os
import time
import sys

from config import FAMILY_DIR

# Main registration flow: collect face photos of a family member
def main():

    print("=" * 50)
    print("        SMART CCTV FAMILY REGISTRATION")
    print("=" * 50)

    name = input("\nEnter family member name: ").strip()

    if not name:
        print("ERROR: Name cannot be empty.")
        return

    # Keep only letters, numbers, spaces, underscores, dashes
    safe_name = "".join(
        c for c in name
        if c.isalnum() or c in (" ", "_", "-")
    ).strip()

    if not safe_name:
        print("ERROR: Invalid name.")
        return

    # Create a folder to store this person's photos
    folder = os.path.join(FAMILY_DIR, safe_name)

    os.makedirs(folder, exist_ok=True)

    print("\nOpening camera...")

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Camera could not be opened.")
        return

    # We want 10 photos per person
    captured = 0
    target = 10

    print("\nInstructions:")
    print("Look directly at the camera.")
    print("Move your head slightly between captures.")
    print("Press SPACE to capture.")
    print("Press Q to quit.\n")

    while True:

        ret, frame = camera.read()

        if not ret:
            print("ERROR: Could not read camera.")
            break

        # Show live preview with progress on screen
        display = frame.copy()

        cv2.putText(
            display,
            f"{safe_name} | Photos: {captured}/{target}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Family Registration",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # SPACE key saves the current photo
        if key == 32:

            filename = os.path.join(
                folder,
                f"face_{captured + 1}.jpg"
            )

            success = cv2.imwrite(
                filename,
                frame
            )

            if success:

                captured += 1

                print(
                    f"Captured {captured}/{target}"
                )

                time.sleep(0.3)

            if captured >= target:

                print(
                    f"\nSuccessfully registered {safe_name}."
                )

                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()