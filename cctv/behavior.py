"""Human behavior analysis: turn tracked trajectories into behavior.

cctv/tracking.py answers "where is each face, and who is it?". This
module answers the next question: "what is the person *doing*?" — are
they standing still, walking, running, loitering, pacing back & forth,
lingering a long time, or walking toward / away from the camera? Each track gets a small, explainable behavior summary
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
    BEHAVIOR_RISK_PACING,
    BEHAVIOR_RISK_RETREAT,
    BEHAVIOR_RISK_WALK,
    BEHAVIOR_PACING_SWITCHES,
    BEHAVIOR_PACING_WINDOW,
    BEHAVIOR_CROUCH_DROP,
    BEHAVIOR_CROUCH_GROWTH,
    BEHAVIOR_DART_SPEED,
    BEHAVIOR_DART_DECEL,
    BEHAVIOR_IDLE_SECONDS,
    BEHAVIOR_PASSERBY_MAX_DWELL,
    BEHAVIOR_GROUP_RADIUS,
    BEHAVIOR_FOLLOW_MIN_SPEED,
    BEHAVIOR_RISK_CROUCH,
    BEHAVIOR_RISK_DART,
    BEHAVIOR_RISK_IDLE,
    BEHAVIOR_RISK_GROUP,
    BEHAVIOR_RISK_FOLLOWING,
    BEHAVIOR_RISK_PASSERBY_CALM,
)


# Motion states (exactly one per track)
STATIONARY = "STATIONARY"
WALKING = "WALKING"
RUNNING = "RUNNING"

# Behavior flags (zero or more per track, alongside the motion state)
LOITERING = "LOITERING"
PACING = "PACING"
APPROACHING = "APPROACHING"
RETREATING = "RETREATING"

# Labels worth waking the operator for. Kept here (not in main.py) so the
# HUD, the alert filter and the tests all agree on what "suspicious" means.
BEHAVIOR_SUSPICIOUS_LABELS = frozenset({LOITERING, PACING, RUNNING, APPROACHING})

# Dwell tiers — human-readable "how long has this person been around".
PACING = "PACING"
CROUCHING = "CROUCHING"        # face dropped + grew: bending / hiding
DARTING = "DARTING"            # sudden sprint burst (lunge toward/away)
IDLING = "IDLING"              # long dwell without loiter radius (watching)
GROUP = "GROUP"                # moving with at least one other tracked person
FOLLOWING = "FOLLOWING"        # trailing another person along a similar path
FAST_PASSERBY = "FAST_PASSERBY"  # quick lateral cross, benign by default

# Motion + posture state shown on the box (derived from box kinematics).
BENT_DOWN = "BENT_DOWN"  # posture state: crouched low vs standing normally

# Dominant lateral/axial travel direction (exactly one informative value;
# "" when the track has barely moved).
DIR_LEFT = "LEFT"
DIR_RIGHT = "RIGHT"
DIR_TOWARD = "TOWARD"
DIR_AWAY = "AWAY"

# Dwell tiers — how long the person has been continuously visible.
DWELL_NEW = "NEW"            # just appeared
DWELL_LINGERING = "LINGERING"  # staying a while
DWELL_LOITERING = "LOITERING"  # long enough to count as loitering-age


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


def is_crouching(
    height_growth: float,
    vertical_drop: float,
    box_height: float,
    accel: float = 0.0,
) -> bool:
    """True when the face dropped low while growing (bend-down / hide).

    A crouch reads as a downward centroid shift combined with a growing
    face box (getting closer/lower to the lens). A sharp deceleration at
    the bottom of the drop confirms the person settled low instead of
    just walking past the camera.
    """
    if box_height <= 0:
        return False
    drop_ratio = vertical_drop / box_height
    return (
        drop_ratio >= BEHAVIOR_CROUCH_DROP
        and height_growth >= BEHAVIOR_CROUCH_GROWTH
        and accel <= BEHAVIOR_DART_DECEL
    )


def is_darting(
    peak_speed: float, norm_speed: float, sustained_run: bool = False
) -> bool:
    """True on a sudden sprint burst — a lunge toward or away.

    Fires when the fastest recent sample crosses the dart threshold while
    the smoothed speed is still catching up. A fully sustained run is
    already covered by the RUNNING motion state, so pure sustained
    running is excluded here to keep the two signals distinct.
    """
    return (
        peak_speed >= BEHAVIOR_DART_SPEED
        and norm_speed >= BEHAVIOR_WALK_SPEED
        and not sustained_run
    )


def is_idling(
    dwell_seconds: float,
    norm_speed: float,
    loitering: bool = False,
) -> bool:
    """True when someone has lingered long without pacing or loitering.

    Idling = visible for at least ``BEHAVIOR_IDLE_SECONDS`` while barely
    moving. Loitering (tight radius) takes precedence, so idling covers
    the wider "standing around watching" case.
    """
    return (
        not loitering
        and dwell_seconds >= BEHAVIOR_IDLE_SECONDS
        and norm_speed < BEHAVIOR_WALK_SPEED
    )


def is_fast_passerby(
    norm_speed: float, dwell_seconds: float, net_displacement: float,
    box_height: float,
) -> bool:
    """True for a quick lateral cross — fast, brief, and gone.

    A passerby moves briskly across the frame with almost no dwell time.
    Flagged so the risk model can explicitly *calm down* instead of
    treating every fast mover as a threat.
    """
    if box_height <= 0:
        return False
    return (
        norm_speed >= BEHAVIOR_WALK_SPEED
        and dwell_seconds <= BEHAVIOR_PASSERBY_MAX_DWELL
        and net_displacement > BEHAVIOR_LOITER_RADIUS * box_height
    )


def proximity_flags(
    own_centroid: tuple,
    own_height: float,
    others: list,
    own_speed: float = 0.0,
) -> set:
    """Cross-track flags: GROUP / FOLLOWING relative to other tracks.

    ``others`` is ``[(centroid, height, speed), ...]`` for every *other*
    live track. Two people whose faces stay within ``BEHAVIOR_GROUP_RADIUS``
    face-heights of each other count as a GROUP; when the pair also moves
    together above ``BEHAVIOR_FOLLOW_MIN_SPEED`` the trailing one (slower
    or equal speed, slightly behind) is FOLLOWING.
    """
    flags: set = set()
    if not others or own_height <= 0:
        return flags
    for (ox, oy), o_height, o_speed in others:
        scale = max(1.0, (own_height + max(1.0, o_height)) / 2.0)
        dist = math.hypot(own_centroid[0] - ox, own_centroid[1] - oy)
        if dist <= BEHAVIOR_GROUP_RADIUS * scale:
            flags.add(GROUP)
            if (
                own_speed >= BEHAVIOR_FOLLOW_MIN_SPEED
                and o_speed >= BEHAVIOR_FOLLOW_MIN_SPEED
            ):
                flags.add(FOLLOWING)
    return flags


def is_pacing(switches: int) -> bool:
    """True when the track reversed lateral direction enough times.

    Repeated LEFT↔RIGHT turns inside one trajectory window is the classic
    pacing / casing pattern — high lateral energy with little net progress.
    """
    return switches >= BEHAVIOR_PACING_SWITCHES


def classify_direction(dx: float, dy: float, height_growth: float) -> str:
    """Dominant travel direction from net pixel deltas.

    Picks the stronger axis: lateral (|dx|) reports LEFT/RIGHT (screen
    coordinates: +x is right), axial (|dy| backed by face-size growth)
    reports TOWARD/AWAY. Returns "" when nothing moved meaningfully.
    """
    adx, ady = abs(dx), abs(dy)
    if adx < 4 and ady < 4 and abs(height_growth) < BEHAVIOR_APPROACH_GROWTH:
        return ""
    if adx >= ady:
        return DIR_RIGHT if dx > 0 else DIR_LEFT
    if height_growth >= BEHAVIOR_APPROACH_GROWTH:
        return DIR_TOWARD
    if height_growth <= -BEHAVIOR_APPROACH_GROWTH:
        return DIR_AWAY
    # Vertical drift without size change (e.g. crouching / camera tilt) —
    # report it axially by sign: up-frame motion reads as approaching.
    return DIR_AWAY if dy > 0 else DIR_TOWARD


def classify_dwell(dwell_seconds: float) -> str:
    """Map continuous visibility time to a dwell tier."""
    if dwell_seconds >= BEHAVIOR_LOITER_SECONDS:
        return DWELL_LOITERING
    if dwell_seconds >= BEHAVIOR_LOITER_SECONDS / 2.0:
        return DWELL_LINGERING
    return DWELL_NEW


def risk_score(
    name: str,
    motion: str,
    flags,
    night_mode: bool = False,
) -> float:
    """Combine identity, motion and behavior into a 0..1 suspicion score.

    Weights come from config so the model stays tunable without touching
    code: unknown baseline + loitering/pacing/running/approaching terms,
    small walking/retreating adjustments, night multiplier, clamped 0..1.
    Family members still score ~0 — the UNKNOWN baseline dominates.
    """
    score = 0.0
    if name == "UNKNOWN":
        score += BEHAVIOR_RISK_UNKNOWN
    if LOITERING in flags:
        score += BEHAVIOR_RISK_LOITER
    if PACING in flags:
        score += BEHAVIOR_RISK_PACING
    if motion == RUNNING:
        score += BEHAVIOR_RISK_RUN
    elif motion == WALKING:
        score += BEHAVIOR_RISK_WALK
    if APPROACHING in flags:
        score += BEHAVIOR_RISK_APPROACH
    if RETREATING in flags:
        score += BEHAVIOR_RISK_RETREAT
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
        direction: str = "",
        dwell_tier: str = DWELL_NEW,
        switches: int = 0,
    ):
        self.motion = motion
        self.flags = frozenset(flags)
        self.risk = risk
        self.norm_speed = norm_speed
        self.direction_deg = direction_deg
        self.dwell_seconds = dwell_seconds
        self.direction = direction
        self.dwell_tier = dwell_tier
        self.switches = switches

    @property
    def is_suspicious(self) -> bool:
        """True when the risk score crosses the alert threshold."""
        return self.risk >= BEHAVIOR_ALERT_RISK

    @property
    def label(self) -> str:
        """Short HUD label, most significant behavior first."""
        if LOITERING in self.flags:
            return LOITERING
        if PACING in self.flags:
            return PACING
        if self.motion == RUNNING:
            return RUNNING
        if APPROACHING in self.flags:
            return APPROACHING
        if RETREATING in self.flags:
            return RETREATING
        if self.motion == WALKING:
            return WALKING
        return STATIONARY

    def describe(self) -> str:
        """Human-readable one-liner for logs, e.g. for a BEHAVIOR_ALERT."""
        extra = ""
        if self.direction:
            extra += f" dir={self.direction}"
        if self.dwell_tier != DWELL_NEW:
            extra += f" dwell_tier={self.dwell_tier}"
        return (
            f"{self.label} speed={self.norm_speed:.2f}/s "
            f"dwell={self.dwell_seconds:.0f}s risk={self.risk:.2f}{extra}"
        )


def analyze(tracks: dict, night_mode: bool = False) -> dict:
    """Return ``{track_id: Behavior}`` for the given tracks.

    Pure function over the tracks' own trajectories — no cross-call
    state, so there is nothing to leak or reset when tracks expire.
    Returns an empty dict when ``BEHAVIOR_ENABLED`` is False, so callers
    can treat "behavior off" and "no behavior" identically.
    """
    if not BEHAVIOR_ENABLED:
        return {}
    return {
        tid: analyze_track(track, night_mode)
        for tid, track in tracks.items()
    }


def analyze_track(track, night_mode: bool = False) -> Behavior:
    """Compute the Behavior for a single ``FaceTrack``."""
    motion = classify_motion(track.norm_speed)

    flags = set()
    if is_loitering(
        track.dwell_seconds, track.net_displacement, track.box_height
    ):
        flags.add(LOITERING)
    if is_pacing(track.direction_switches):
        flags.add(PACING)
    approach = classify_approach(track.height_growth)
    if approach:
        flags.add(approach)

    dx, dy = track.net_delta
    direction = classify_direction(dx, dy, track.height_growth)

    return Behavior(
        motion=motion,
        flags=flags,
        risk=risk_score(
            track.majority_name, motion, flags, night_mode
        ),
        norm_speed=track.norm_speed,
        direction_deg=track.direction_deg,
        dwell_seconds=track.dwell_seconds,
        direction=direction,
        dwell_tier=classify_dwell(track.dwell_seconds),
        switches=track.direction_switches,
    )


# Behavior labels worth logging from the alarm pipeline (rate-limited).
# Plain STATIONARY / WALKING / RETREATING are normal movement — not events.
BEHAVIOR_SUSPICIOUS_LABELS = frozenset(
    {"LOITERING", "PACING", "RUNNING", "APPROACHING"}
)