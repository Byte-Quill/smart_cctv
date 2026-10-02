"""Human behavior analysis: turn tracked trajectories into behavior.

cctv/tracking.py answers "where is each face, and who is it?". This
module answers the next question: "what is the person *doing*?" — are
they standing still, walking, running, loitering in one spot, or walking
toward the camera? Each track gets a small, explainable behavior summary
plus a 0..1 risk score that lets the alarm logic respond faster to
genuinely suspicious movement without changing how family members are
treated.

Everything here is pure math on the boxes the tracker already produced,
so it runs in microseconds per frame and needs no camera, model, or GPU.
"""

from config import (
    BEHAVIOR_ENABLED,
    BEHAVIOR_WALK_SPEED,
    BEHAVIOR_RUN_SPEED,
    BEHAVIOR_LOITER_SECONDS,
    BEHAVIOR_LOITER_RADIUS,
    BEHAVIOR_APPROACH_GROWTH,
    BEHAVIOR_RISK_UNKNOWN,
    BEHAVIOR_RISK_LOITER,
    BEHAVIOR_RISK_RUN,
    BEHAVIOR_RISK_APPROACH,
    BEHAVIOR_RISK_NIGHT_MULT,
    BEHAVIOR_ALERT_RISK,
)


# Motion states (exactly one per track)
STATIONARY = "STATIONARY"
WALKING = "WALKING"
RUNNING = "RUNNING"

# Behavior flags (zero or more per track, alongside the motion state)
LOITERING = "LOITERING"
APPROACHING = "APPROACHING"
RETREATING = "RETREATING"


def classify_motion(norm_speed: float) -> str:
    """Map a scale-independent speed (face-heights/second) to a state."""
    if norm_speed >= BEHAVIOR_RUN_SPEED:
        return RUNNING
    if norm_speed >= BEHAVIOR_WALK_SPEED:
        return WALKING
    return STATIONARY


def classify_approach(height_growth: float) -> str:
    """Return APPROACHING / RETREATING / "" from fractional face growth."""
    if height_growth >= BEHAVIOR_APPROACH_GROWTH:
        return APPROACHING
    if height_growth <= -BEHAVIOR_APPROACH_GROWTH:
        return RETREATING
    return ""


def is_loitering(
    dwell_seconds: float, net_displacement: float, box_height: float
) -> bool:
    """True when someone has stayed a while without going anywhere.

    Loitering = present for at least ``BEHAVIOR_LOITER_SECONDS`` while
    their net displacement stays within ``BEHAVIOR_LOITER_RADIUS``
    face-heights. Measuring the radius in face-heights keeps the test
    meaningful whether the person is near or far from the camera.
    """
    if dwell_seconds < BEHAVIOR_LOITER_SECONDS:
        return False
    return net_displacement <= BEHAVIOR_LOITER_RADIUS * max(1.0, box_height)


def risk_score(name: str, motion: str, flags, night_mode: bool = False) -> float:
    """Combine identity, motion and behavior into a 0..1 suspicion score."""
    score = 0.0
    if name == "UNKNOWN":
        score += BEHAVIOR_RISK_UNKNOWN
    if LOITERING in flags:
        score += BEHAVIOR_RISK_LOITER
    if motion == RUNNING:
        score += BEHAVIOR_RISK_RUN
    if APPROACHING in flags:
        score += BEHAVIOR_RISK_APPROACH
    if night_mode:
        score *= BEHAVIOR_RISK_NIGHT_MULT
    return max(0.0, min(1.0, score))


class Behavior:
    """Explainable behavior summary for one tracked person."""

    def __init__(
        self,
        motion: str,
        flags,
        risk: float,
        norm_speed: float = 0.0,
        direction_deg: float = 0.0,
        dwell_seconds: float = 0.0,
    ):
        self.motion = motion
        self.flags = frozenset(flags)
        self.risk = risk
        self.norm_speed = norm_speed
        self.direction_deg = direction_deg
        self.dwell_seconds = dwell_seconds

    @property
    def is_suspicious(self) -> bool:
        """True when the risk score crosses the alert threshold."""
        return self.risk >= BEHAVIOR_ALERT_RISK

    @property
    def label(self) -> str:
        """Short HUD label, most significant behavior first."""
        if LOITERING in self.flags:
            return LOITERING
        if self.motion == RUNNING:
            return RUNNING
        if APPROACHING in self.flags:
            return APPROACHING
        if self.motion == WALKING:
            return WALKING
        return STATIONARY

    def describe(self) -> str:
        """Human-readable one-liner for logs, e.g. for a BEHAVIOR_ALERT."""
        return (
            f"{self.label} speed={self.norm_speed:.2f}/s "
            f"dwell={self.dwell_seconds:.0f}s risk={self.risk:.2f}"
        )


class BehaviorAnalyzer:
    """Analyze every live track each detection frame.

    Keeps no cross-frame state of its own — each ``FaceTrack`` already
    carries its own trajectory — so there is nothing to leak or reset
    when tracks expire.
    """

    def __init__(self, enabled: bool = BEHAVIOR_ENABLED):
        self.enabled = enabled

    def analyze(self, tracks: dict, night_mode: bool = False) -> dict:
        """Return ``{track_id: Behavior}`` for the given tracks.

        Returns an empty dict when disabled, so callers can treat
        "behavior off" and "no behavior" identically.
        """
        if not self.enabled:
            return {}
        return {
            tid: self.analyze_track(track, night_mode)
            for tid, track in tracks.items()
        }

    @staticmethod
    def analyze_track(track, night_mode: bool = False) -> Behavior:
        """Compute the Behavior for a single ``FaceTrack``."""
        motion = classify_motion(track.norm_speed)

        flags = set()
        if is_loitering(
            track.dwell_seconds, track.net_displacement, track.box_height
        ):
            flags.add(LOITERING)
        approach = classify_approach(track.height_growth)
        if approach:
            flags.add(approach)

        return Behavior(
            motion=motion,
            flags=flags,
            risk=risk_score(
                track.majority_name, motion, flags, night_mode
            ),
            norm_speed=track.norm_speed,
            direction_deg=track.direction_deg,
            dwell_seconds=track.dwell_seconds,
        )