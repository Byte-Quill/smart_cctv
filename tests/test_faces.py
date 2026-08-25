"""Unit tests for cctv.faces recognition logic (no camera or GPU required).

face_recognition is imported lazily inside the module, so these tests
exercise the pure distance/confidence math with synthetic encodings.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    from cctv.faces import recognize_face
    HAS_FACE_RECOGNITION = True
except ImportError:
    recognize_face = None
    HAS_FACE_RECOGNITION = False


def make_encoding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=128)


@unittest.skipUnless(
    HAS_FACE_RECOGNITION,
    "face_recognition not installed; skipping recognition tests",
)
class TestRecognizeFace(unittest.TestCase):

    def test_no_known_faces_returns_unknown(self):
        name, distance, confidence = recognize_face(
            make_encoding(1), [], []
        )
        self.assertEqual(name, "UNKNOWN")
        self.assertIsNone(distance)
        self.assertEqual(confidence, 0.0)

    def test_exact_match_returns_name_and_high_confidence(self):
        enc = make_encoding(42)
        name, distance, confidence = recognize_face(
            enc, [enc], ["Alice"]
        )
        self.assertEqual(name, "Alice")
        self.assertAlmostEqual(distance, 0.0, places=6)
        self.assertAlmostEqual(confidence, 1.0, places=6)

    def test_far_encoding_is_unknown(self):
        # Two very different encodings -> distance > tolerance
        enc_a = make_encoding(1)
        enc_b = make_encoding(2)
        name, distance, confidence = recognize_face(
            enc_a, [enc_b], ["Alice"]
        )
        self.assertEqual(name, "UNKNOWN")
        self.assertGreater(distance, 0.45)
        self.assertEqual(confidence, 0.0)

    def test_confidence_scales_with_distance(self):
        enc = make_encoding(7)
        # Same encoding twice: distance 0 -> confidence 1.0
        _, _, c0 = recognize_face(enc, [enc], ["A"])
        # A perturbed encoding: distance > 0 -> confidence < 1.0
        noisy = enc + 0.1
        _, _, c1 = recognize_face(noisy, [enc], ["A"])
        self.assertAlmostEqual(c0, 1.0, places=6)
        self.assertLess(c1, 1.0)
        self.assertGreaterEqual(c1, 0.0)

    def test_closest_of_multiple_known_faces(self):
        enc = make_encoding(3)
        far = make_encoding(4)
        name, _, _ = recognize_face(enc, [far, enc], ["Bob", "Alice"])
        self.assertEqual(name, "Alice")


if __name__ == "__main__":
    unittest.main()
