"""Tests for cctv/timeutil.py — Nepal-time day/night security logic."""

import unittest

from datetime import datetime, timedelta, timezone

from cctv import timeutil
from config import (
    NIGHT_START_HOUR,
    NIGHT_END_HOUR,
    SIREN_DAY_DURATION,
    SIREN_NIGHT_DURATION,
    NEPAL_UTC_OFFSET_MINUTES,
)


def npt(hour: int, minute: int = 0) -> datetime:
    """Build a Nepal-time datetime at the given local hour/minute."""
    return datetime(2026, 8, 26, hour, minute, tzinfo=timeutil.NPT)


class TestNepalTimezone(unittest.TestCase):
    def test_offset_is_utc_plus_545(self):
        self.assertEqual(NEPAL_UTC_OFFSET_MINUTES, 345)
        self.assertEqual(
            timeutil.NPT.utcoffset(None),
            timedelta(hours=5, minutes=45),
        )

    def test_nepal_now_is_timezone_aware(self):
        now = timeutil.nepal_now()
        self.assertIsNotNone(now.tzinfo)
        self.assertEqual(now.utcoffset(), timedelta(hours=5, minutes=45))


class TestNightMode(unittest.TestCase):
    def test_daytime_is_not_night(self):
        for hour in (6, 10, 12, 15, 21):
            self.assertFalse(timeutil.is_night_mode(npt(hour)), hour)

    def test_night_hours_are_night(self):
        # 22:00-23:59 and 00:00-05:59 (window wraps past midnight)
        for hour in (22, 23, 0, 1, 3, 5):
            self.assertTrue(timeutil.is_night_mode(npt(hour)), hour)

    def test_boundary_hours(self):
        # Exactly at NIGHT_START_HOUR -> night; at NIGHT_END_HOUR -> day
        self.assertTrue(timeutil.is_night_mode(npt(NIGHT_START_HOUR)))
        self.assertFalse(timeutil.is_night_mode(npt(NIGHT_END_HOUR)))


class TestSirenDuration(unittest.TestCase):
    def test_day_duration(self):
        self.assertEqual(
            timeutil.siren_duration(npt(12)), SIREN_DAY_DURATION
        )

    def test_night_duration(self):
        self.assertEqual(
            timeutil.siren_duration(npt(23)), SIREN_NIGHT_DURATION
        )

    def test_durations_match_spec(self):
        # Day = 2 minutes, night = 5 minutes
        self.assertEqual(SIREN_DAY_DURATION, 120)
        self.assertEqual(SIREN_NIGHT_DURATION, 300)


if __name__ == "__main__":
    unittest.main()
