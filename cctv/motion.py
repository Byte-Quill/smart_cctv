"""Motion detection: cheap frame-difference gate before heavy processing."""

import cv2
import numpy as np


def _preprocess(frame: np.ndarray, scale: float) -> np.ndarray:
    """Downscale, grayscale, and blur a frame once for diffing."""
    small = cv2.resize(
        frame, (0, 0), fx=scale, fy=scale
    )
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (21, 21), 0)


class MotionDetector:
    """Detects scene change against a slowly-updating background model.

    Only frames with enough motion pass the gate, so the expensive
    face/YOLO pipeline runs only when something actually moves.

    Compared with a naive consecutive-frame diff, the background model
    is updated only by a fraction each frame. A person creeping in
    gradually still changes enough pixels to trip the gate, instead of
    silently merging into an ever-advancing baseline.
    """

    def __init__(
        self,
        threshold: float = 25.0,
        min_area: float = 0.01,
        scale: float = 0.25,
        bg_alpha: float = 0.05,
    ):
        self.threshold = threshold
        self.min_area = min_area
        self.scale = scale
        # How much the running background adapts toward each new frame.
        self.bg_alpha = bg_alpha
        self._bg_gray = None

    def _fraction_changed(self, frame: np.ndarray) -> float:
        """Diff one frame against the background and advance the model.

        Returns the fraction of changed pixels (0.0-1.0). The background
        advances exactly once per call, so the first frame initializes it
        and reports no motion.
        """
        gray = _preprocess(frame, self.scale)

        if self._bg_gray is None:
            self._bg_gray = gray
            return 0.0

        diff = cv2.absdiff(self._bg_gray, gray)

        # Blend the background toward the current frame so slow, gradual
        # changes accumulate into detectable signal over time.
        a = self.bg_alpha
        self._bg_gray = cv2.convertScaleAbs(
            (1.0 - a) * self._bg_gray + a * gray
        )

        _, thresh = cv2.threshold(
            diff, self.threshold, 255, cv2.THRESH_BINARY
        )
        return float(np.count_nonzero(thresh)) / thresh.size

    def motion(self, frame: np.ndarray) -> float:
        """Return the fraction of changed pixels (0.0-1.0) vs background."""
        return self._fraction_changed(frame)

    def has_motion(self, frame: np.ndarray) -> bool:
        """True when the frame differs enough from the background."""
        return self._fraction_changed(frame) >= self.min_area

    def reset(self):
        self._bg_gray = None
