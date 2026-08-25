"""Motion detection: cheap frame-difference gate before heavy processing."""

import cv2
import numpy as np


class MotionDetector:
    """Detects scene change between consecutive frames.

    Only frames with enough motion pass the gate, so the expensive
    face/YOLO pipeline runs only when something actually moves.
    """

    def __init__(
        self,
        threshold: float = 25.0,
        min_area: float = 0.01,
        scale: float = 0.25,
    ):
        self.threshold = threshold
        self.min_area = min_area
        self.scale = scale
        self._prev_gray = None

    def motion(self, frame: np.ndarray) -> float:
        """Return the fraction of changed pixels (0.0-1.0) vs previous frame."""
        small = cv2.resize(
            frame, (0, 0), fx=self.scale, fy=self.scale
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0

        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        _, thresh = cv2.threshold(
            diff, self.threshold, 255, cv2.THRESH_BINARY
        )
        return float(np.count_nonzero(thresh)) / thresh.size

    def has_motion(self, frame: np.ndarray) -> bool:
        """True when the frame differs enough from the previous one.

        Advances the baseline exactly once per call (unlike calling
        ``motion()`` twice, which would compare against the same frame).
        """
        if self._prev_gray is None:
            self.motion(frame)
            return False

        small = cv2.resize(
            frame, (0, 0), fx=self.scale, fy=self.scale
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        _, thresh = cv2.threshold(
            diff, self.threshold, 255, cv2.THRESH_BINARY
        )
        return float(np.count_nonzero(thresh)) / thresh.size >= self.min_area

    def reset(self):
        self._prev_gray = None
