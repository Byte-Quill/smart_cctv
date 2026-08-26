"""Unit tests for cctv.tracking (no camera or GPU required)."""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cctv.tracking import FaceHistory, FaceTrack, match_tracks


class TestFaceHistory(unittest.TestCase):

    def test_majority_name(self):
        h = FaceHistory(window=5)
        for name in ["Alice", "Alice", "Bob", "Alice", "Alice"]:
            h.add(name, 0.9)
        self.assertEqual(h.majority_name, "Alice")

    def test_majority_rolls_off_window(self):
        h = FaceHistory(window=3)
        for name in ["Bob", "Bob", "Bob", "Alice", "Alice", "Alice"]:
            h.add(name, 0.8)
        # Old Bob votes rolled off; Alice now dominates
        self.assertEqual(h.majority_name, "Alice")

    def test_empty_history_is_unknown(self):
        self.assertEqual(FaceHistory().majority_name, "UNKNOWN")

    def test_avg_confidence(self):
        h = FaceHistory(window=5)
        h.add("X", 0.5)
        h.add("X", 0.7)
        self.assertAlmostEqual(h.avg_confidence, 0.6)

    def test_empty_confidence_is_zero(self):
        self.assertEqual(FaceHistory().avg_confidence, 0.0)


class TestFaceTrack(unittest.TestCase):

    def test_patience_decay_and_expiry(self):
        t = FaceTrack((0, 10, 20, 0), "Alice", 0.9)
        self.assertTrue(t.is_alive)
        for _ in range(10):
            t.decay_patience()
        self.assertFalse(t.is_alive)

    def test_update_resets_patience(self):
        t = FaceTrack((0, 10, 20, 0), "Alice", 0.9)
        for _ in range(3):
            t.decay_patience()
        t.update((0, 10, 20, 0), "Alice", 0.9)
        self.assertTrue(t.is_alive)

    def test_smoothing_moves_toward_new_location(self):
        t = FaceTrack((0, 10, 20, 0), "Alice", 0.9)
        t.update((0, 20, 20, 10), "Alice", 0.9)
        # EMA with alpha 0.6: new > old, so smoothed right edge grew
        self.assertGreater(t.smoothed[1], 10)


class TestMatchTracks(unittest.TestCase):

    def test_new_face_creates_track(self):
        tracks = match_tracks([((0, 10, 20, 0), "Alice", 0.9)], {}, 0)
        self.assertEqual(len(tracks), 1)
        # A brand-new track has only 1 vote, so its identity is not yet
        # confirmed (IDENTITY_MIN_VOTES = 2) — it reports UNKNOWN until
        # a second matching frame arrives.
        self.assertEqual(tracks[0].majority_name, "UNKNOWN")

    def test_identity_confirmed_after_min_votes(self):
        tracks = match_tracks([((0, 10, 20, 0), "Alice", 0.9)], {}, 0)
        tracks = match_tracks([((0, 10, 20, 0), "Alice", 0.9)], tracks, 0)
        self.assertEqual(tracks[0].majority_name, "Alice")

    def test_skip_frame_decays_without_matching(self):
        t = FaceTrack((0, 10, 20, 0), "Alice", 0.9)
        prev = {0: t}
        # frame 1 is a skip frame (TRACKING_SKIP_FRAMES = 2)
        tracks = match_tracks([], prev, 1)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].patience, 4)  # decayed once

    def test_nearby_face_matches_existing_track(self):
        t = FaceTrack((0, 10, 20, 0), "Alice", 0.9)
        prev = {0: t}
        # Same centroid, detection frame (frame 0)
        tracks = match_tracks([((0, 12, 22, 2), "Alice", 0.9)], prev, 0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].patience, 5)  # reset by update

    def test_far_face_creates_new_track(self):
        t = FaceTrack((0, 10, 20, 0), "Alice", 0.9)
        prev = {0: t}
        # Centroid far away (> 60px) -> new track
        tracks = match_tracks([((0, 200, 220, 180), "Bob", 0.9)], prev, 0)
        self.assertEqual(len(tracks), 2)

    def test_unmatched_track_expires(self):
        t = FaceTrack((0, 10, 20, 0), "Alice", 0.9)
        prev = {0: t}
        # Detection frame with no faces -> decay, still alive
        tracks = match_tracks([], prev, 0)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].patience, 4)


if __name__ == "__main__":
    unittest.main()
