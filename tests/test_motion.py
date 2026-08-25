"""Unit tests for cctv.motion background-model gate (no camera or GPU)."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    from cctv.motion import MotionDetector
    HAS_CV = True
except ImportError:
    HAS_CV = False


def solid_frame(value: int, shape=(100, 100, 3)) -> np.ndarray:
    """Return a solid-color BGR frame, uint8."""
    return np.full(shape, value, dtype="uint8")


@unittest.skipUnless(HAS_CV, "OpenCV/numpy not installed; skipping motion tests")
class TestMotionDetector(unittest.TestCase):

    def test_first_frame_has_no_motion(self):
        det = MotionDetector()
        self.assertFalse(det.has_motion(solid_frame(0)))

    def test_identical_frames_have_no_motion(self):
        det = MotionDetector()
        f = solid_frame(0)
        det.has_motion(f)          # initializes background
        self.assertFalse(det.has_motion(f))

    def test_large_jump_detects_motion(self):
        det = MotionDetector()
        det.has_motion(solid_frame(0))
        self.assertTrue(det.has_motion(solid_frame(200)))

    def test_slow_drift_accumulates_over_time(self):
        # A small per-frame change below the instantaneous threshold still
        # trips the gate once the background model lags enough behind.
        det = MotionDetector(threshold=80.0, min_area=0.5)
        for v in range(0, 200, 5):
            _ = det.has_motion(solid_frame(v))
        # Near the end the background lags far behind the bright frame.
        self.assertTrue(det.has_motion(solid_frame(200)))

    def test_constant_scene_stays_quiet(self):
        det = MotionDetector(bg_alpha=0.05)
        f = solid_frame(30)
        det.has_motion(f)
        for _ in range(50):
            self.assertFalse(det.has_motion(f))

    def test_reset_clears_background(self):
        det = MotionDetector()
        det.has_motion(solid_frame(100))
        det.reset()
        # After reset the next frame re-initializes the background.
        self.assertFalse(det.has_motion(solid_frame(100)))


if __name__ == "__main__":
    unittest.main()