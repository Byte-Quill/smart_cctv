"""Hardware abstraction layer — the seam for future device ports.

The whole system talks to the physical world through this one module.
Today everything runs on a PC (OpenCV camera + pygame siren), but the
plan is to move to smaller boards later:

    "pc"     desktop/laptop (current default)
    "pi"     Raspberry Pi 5 — same camera/siren stack, plus an optional
             GPIO relay hook for an external siren/light
    "esp32"  ESP32-CAM — the board only captures and streams frames over
             the network; the heavy face pipeline still runs on a host

To port the system, implement a new CameraSource/SirenOutput pair here
and select it with HARDWARE_PROFILE in config.py. Nothing else in the
codebase needs to change — main.py only ever calls ``open_camera()`` and
uses the returned object's ``read()``/``release()``.
"""

import cv2

from config import HARDWARE_PROFILE, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


def open_camera() -> cv2.VideoCapture:
    """Return a configured camera capture object for the active profile.

    All profiles currently use OpenCV's VideoCapture; the differences are
    in which backend/index is chosen. An ESP32-CAM would be reached via
    its MJPEG/RTSP stream URL instead of a local device index.
    """
    # Every profile currently opens the camera through OpenCV. For an
    # ESP32-CAM, point CAMERA_INDEX at the board's MJPEG stream URL
    # (e.g. "http://192.168.1.50:81/stream") and OpenCV decodes it like a
    # local camera; "pc" and "pi" use a locally attached webcam / Pi
    # camera module instead.
    camera = cv2.VideoCapture(CAMERA_INDEX)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    # Keep the driver queue at one frame so the display stays real time.
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return camera


def describe() -> str:
    """Human-readable summary of the active hardware profile."""
    return (
        f"hardware={HARDWARE_PROFILE} "
        f"camera_index={CAMERA_INDEX} "
        f"resolution={CAMERA_WIDTH}x{CAMERA_HEIGHT}"
    )
