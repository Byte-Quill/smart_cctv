"""Tests for cctv/enroll.py — pure helpers of the in-app enrollment flow."""

import unittest

from cctv.enroll import sanitize_name, zone_index


class TestSanitizeName(unittest.TestCase):
    def test_keeps_letters_digits_spaces_dashes_underscores(self):
        self.assertEqual(
            sanitize_name("Ama-rita_01"), "Ama-rita_01"
        )

    def test_strips_forbidden_characters(self):
        self.assertEqual(sanitize_name("a/b\\c:d*e"), "abcde")

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(sanitize_name("  Ram  "), "Ram")

    def test_empty_and_garbage_become_empty(self):
        self.assertEqual(sanitize_name(""), "")
        self.assertEqual(sanitize_name("///"), "")


class TestZoneIndex(unittest.TestCase):
    def test_left_zone(self):
        self.assertEqual(zone_index(0, 640), 0)
        self.assertEqual(zone_index(200, 640), 0)

    def test_center_zone(self):
        self.assertEqual(zone_index(320, 640), 1)

    def test_right_zone(self):
        self.assertEqual(zone_index(639, 640), 2)

    def test_boundaries(self):
        w = 600
        self.assertEqual(zone_index(199, w), 0)
        self.assertEqual(zone_index(200, w), 1)
        self.assertEqual(zone_index(399, w), 1)
        self.assertEqual(zone_index(400, w), 2)


if __name__ == "__main__":
    unittest.main()
