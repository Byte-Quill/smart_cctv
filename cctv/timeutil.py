"""Nepal Time (NPT) clock — the single source of truth for all time checks.

Nepal uses a fixed UTC+5:45 offset with no daylight saving, so the clock
is a simple timezone conversion that works no matter what timezone the
host machine is set to. Every time-of-day decision in the system (day vs
night security mode, allowed siren hours, log timestamps) goes through
this module so the behaviour is always consistent.

Why a dedicated module?
-----------------------
- The host may run in any timezone (or UTC on a server/Pi), but the
  security rules are written in Nepal local time.
- Centralising the conversion means a future change (e.g. a different
  deployment country) touches exactly one file.
- Tests can exercise the day/night logic without mocking the system clock.
"""

from datetime import datetime, timedelta, timezone

from config import (
    NEPAL_UTC_OFFSET_MINUTES,
    NIGHT_START_HOUR,
    NIGHT_END_HOUR,
)


# Fixed Nepal Time zone: UTC+5:45, no DST.
NPT = timezone(timedelta(minutes=NEPAL_UTC_OFFSET_MINUTES))


def nepal_now() -> datetime:
    """Return the current date/time in Nepal Time (timezone-aware)."""
    return datetime.now(NPT)


def nepal_hour(now: datetime | None = None) -> int:
    """Return the current hour (0-23) in Nepal Time."""
    if now is None:
        now = nepal_now()
    return now.hour


def is_night_mode(now: datetime | None = None) -> bool:
    """True when night security mode is active (NIGHT_START_HOUR..NIGHT_END_HOUR).

    Handles the window that crosses midnight, e.g. 22:00 -> 06:00.
    """
    hour = nepal_hour(now)
    if NIGHT_START_HOUR > NIGHT_END_HOUR:
        # Window wraps past midnight: 22,23,0,1,...,5
        return hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR
    # Normal window (start < end)
    return NIGHT_START_HOUR <= hour < NIGHT_END_HOUR


def siren_duration(now: datetime | None = None) -> int:
    """Return the auto-stop duration (seconds) for the current mode.

    Night security mode uses the longer duration; daytime uses the shorter
    one. Imported here to avoid a circular import at module load.
    """
    from config import SIREN_DAY_DURATION, SIREN_NIGHT_DURATION

    return SIREN_NIGHT_DURATION if is_night_mode(now) else SIREN_DAY_DURATION
