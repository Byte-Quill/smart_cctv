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

import threading
import time

import cv2

from config import HARDWARE_PROFILE, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


class LatestFrameCamera:
    """A camera whose frames are captured on a background thread.

    Why this exists
    ---------------
    ``cv2.VideoCapture.read()`` is synchronous and the driver queue holds
    a single frame: the camera only starts capturing the *next* frame once
    ``read()`` has been called. In the monitoring loop that makes every
    frame period equal to ``camera frame time + our per-frame work``, so
    ~20 ms of enhance/detect/draw turns a 10 FPS camera into ~8 FPS.

    This wrapper pulls frames in a daemon thread so capture runs on its
    own clock, independent of the caller. ``read()`` keeps the exact
    ``cv2.VideoCapture`` blocking semantics (wait for the next fresh
    frame), so every caller — the monitoring loop and the enrollment flow
    alike — is unchanged, but processing time is now hidden inside the
    camera's frame interval instead of added to it.
    """

    #: Consecutive failed reads before the camera is reported as dropped
    #: (read() then returns (False, None) so main.py reconnects).
    _FAILURE_LIMIT = 15
    #: How long __init__ waits for the very first frame.
    _FIRST_FRAME_TIMEOUT = 5.0
    #: How long read() waits for a new frame before re-serving the last one.
    _READ_TIMEOUT = 5.0

    def __init__(self, camera: cv2.VideoCapture):
        self._camera = camera
        # One condition guards the frame slot and wakes readers the moment
        # a new frame lands (no polling, no busy-wait).
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._failures = 0
        self._running = True

        self._thread = threading.Thread(
            target=self._capture_loop, name="camera-capture", daemon=True
        )
        self._thread.start()

        # Wait for the first frame so callers never mistake warm-up for a
        # dropped camera (main.py reconnects when a read fails).
        with self._cond:
            self._cond.wait_for(
                lambda: self._frame is not None,
                timeout=self._FIRST_FRAME_TIMEOUT,
            )

    def _capture_loop(self):
        """Continuously read the camera and keep only the newest frame."""
        while self._running:
            ok, frame = self._camera.read()
            with self._cond:
                if ok and frame is not None:
                    self._frame = frame
                    self._seq += 1
                    self._failures = 0
                    self._cond.notify_all()
                else:
                    self._failures += 1
                    if self._failures >= self._FAILURE_LIMIT:
                        # Tell the caller the camera dropped.
                        self._frame = None
                        self._cond.notify_all()
            if not ok:
                # Avoid a hot spin while the device is unplugged or broken.
                time.sleep(0.01)

    def read(self):
        """Wait for a frame newer than the last one (OpenCV-compatible).

        Returns the *newest* available frame so a slow consumer stays
        real-time instead of playing catch-up. Returns ``(False, None)``
        only when the camera has actually dropped.
        """
        with self._cond:
            if self._frame is None:
                return False, None
            start = self._seq
            # Block until a newer frame is published (or the camera drops).
            self._cond.wait_for(
                lambda: self._seq != start or self._frame is None,
                timeout=self._READ_TIMEOUT,
            )
            if self._frame is None:
                return False, None
            return True, self._frame

    def isOpened(self) -> bool:
        return self._camera.isOpened()

    def get(self, prop_id):
        return self._camera.get(prop_id)

    def set(self, prop_id, value):
        return self._camera.set(prop_id, value)

    def release(self):
        """Stop the capture thread, then release the device."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._camera.release()


def open_camera() -> LatestFrameCamera:
    """Return a configured, threaded camera for the active profile.

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

    # Wrap it so capture runs on its own thread (see LatestFrameCamera):
    # the monitoring loop then runs at the camera's true frame rate.
    return LatestFrameCamera(camera)


def describe() -> str:
    """Human-readable summary of the active hardware profile."""
    return (
        f"hardware={HARDWARE_PROFILE} "
        f"camera_index={CAMERA_INDEX} "
        f"resolution={CAMERA_WIDTH}x{CAMERA_HEIGHT}"
    )
