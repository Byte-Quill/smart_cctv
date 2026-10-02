"""Temporal face tracking: smoothed boxes, majority-vote identity, and the
short trajectory that human-behavior analysis (cctv/behavior.py) reads."""

import math
import time

from collections import Counter, deque

from config import (
    ENSEMBLE_FRAMES,
    TRACKING_SKIP_FRAMES,
    TRACKING_SMOOTH_ALPHA,
    TRACKING_PATIENCE,
    IDENTITY_MIN_VOTES,
    BEHAVIOR_TRAJECTORY_LEN,
)


class FaceHistory:
    """Rolling classification history for one tracked face."""

    def __init__(self, window: int = ENSEMBLE_FRAMES):
        self.window = window
        self.classes: deque = deque(maxlen=window)
        self.confidences: deque = deque(maxlen=window)

    def add(self, name: str, confidence: float):
        self.classes.append(name)
        self.confidences.append(confidence)

    @property
    def majority_name(self) -> str:
        """Return the majority-vote name over the window.

        An identity needs at least IDENTITY_MIN_VOTES votes before it is
        trusted, so a single lucky frame cannot flash a wrong name or
        start an alarm.
        """
        if len(self.classes) < IDENTITY_MIN_VOTES:
            return "UNKNOWN"
        counts = Counter(self.classes)
        return counts.most_common(1)[0][0]

    @property
    def avg_confidence(self) -> float:
        """Mean confidence over the window."""
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)


class FaceTrack:
    """Track state for one face.

    Holds the smoothed box and identity history, plus a short trajectory
    of recent positions. The trajectory is what turns raw boxes into
    *behavior*: from it cctv/behavior.py derives speed, direction, dwell
    time and whether the person is loitering or walking toward the
    camera. Samples are ``(cx, cy, t, height)`` in detection-scale
    coordinates, where ``height`` is the face height (a cheap proxy for
    how close the person is).
    """

    def __init__(self, location, name: str, confidence: float, now=None):
        self.history = FaceHistory()
        self.history.add(name, confidence)
        # Smoothed location (detection-scale coords)
        self.smoothed = location
        self.patience = TRACKING_PATIENCE  # frames remaining before expiry
        self.last_seen = location
        self.first_seen = time.time() if now is None else now
        self.trajectory = deque(maxlen=BEHAVIOR_TRAJECTORY_LEN)
        self._append_sample(location, now)

    def _append_sample(self, location, now=None):
        top, right, bottom, left = location
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        height = max(1, bottom - top)
        self.trajectory.append(
            (cx, cy, time.time() if now is None else now, height)
        )

    def update(self, location, name: str, confidence: float, now=None):
        self.history.add(name, confidence)
        # EMA smoothing on each coordinate
        a = TRACKING_SMOOTH_ALPHA
        self.smoothed = tuple(
            int(a * loc_coord + (1 - a) * smooth_coord)
            for loc_coord, smooth_coord in zip(location, self.smoothed)
        )
        self.last_seen = location
        self.patience = TRACKING_PATIENCE  # reset patience
        self._append_sample(location, now)

    def decay_patience(self):
        self.patience -= 1

    @property
    def is_alive(self) -> bool:
        return self.patience > 0

    @property
    def majority_name(self) -> str:
        return self.history.majority_name

    @property
    def avg_confidence(self) -> float:
        return self.history.avg_confidence

    # ── Kinematics derived from the trajectory ──────────────────────────

    @property
    def centroid(self) -> tuple[int, int]:
        """Latest detection-scale centroid (cx, cy)."""
        return _centroid(self.last_seen)

    @property
    def box_height(self) -> int:
        """Latest face height in detection-scale pixels (>= 1)."""
        top, _right, bottom, _left = self.last_seen
        return max(1, bottom - top)

    @property
    def dwell_seconds(self) -> float:
        """How long this track has been observed, in seconds.

        Measured from the track's very first sighting (not just the
        bounded trajectory window), so a long-lived track is reported
        correctly even after older samples roll off.
        """
        if not self.trajectory:
            return 0.0
        return max(0.0, self.trajectory[-1][2] - self.first_seen)

    @property
    def speed(self) -> float:
        """Recent speed in detection-scale px per second."""
        if len(self.trajectory) < 2:
            return 0.0
        x1, y1, t1, _ = self.trajectory[-1]
        x0, y0, t0, _ = self.trajectory[-2]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return math.hypot(x1 - x0, y1 - y0) / dt

    @property
    def norm_speed(self) -> float:
        """Speed in face-heights per second (resolution/scale independent)."""
        return self.speed / self.box_height

    @property
    def direction_deg(self) -> float:
        """Heading of the last motion step, in degrees (0 = right, 90 = down)."""
        if len(self.trajectory) < 2:
            return 0.0
        x1, y1, _t1, _ = self.trajectory[-1]
        x0, y0, _t0, _ = self.trajectory[-2]
        return math.degrees(math.atan2(y1 - y0, x1 - x0))

    @property
    def net_displacement(self) -> float:
        """Straight-line distance between the oldest and newest sample."""
        if len(self.trajectory) < 2:
            return 0.0
        x0, y0, _t0, _ = self.trajectory[0]
        x1, y1, _t1, _ = self.trajectory[-1]
        return math.hypot(x1 - x0, y1 - y0)

    @property
    def path_length(self) -> float:
        """Total distance travelled across the trajectory window."""
        total = 0.0
        prev = None
        for x, y, _t, _h in self.trajectory:
            if prev is not None:
                total += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)
        return total

    @property
    def height_growth(self) -> float:
        """Fractional change in face height across the window.

        Positive = the face grew (moving toward the camera); negative =
        it shrank (moving away). 0.0 when there is too little history.
        """
        if len(self.trajectory) < 2:
            return 0.0
        first_h = self.trajectory[0][3]
        if first_h <= 0:
            return 0.0
        return (self.trajectory[-1][3] - first_h) / first_h


TrackDict = dict[int, FaceTrack]

# Max centroid distance (detection-scale px) for a detection to match a track.
MATCH_DISTANCE_PX = 60


def _centroid(location) -> tuple[int, int]:
    top, right, bottom, left = location
    return ((left + right) // 2, (top + bottom) // 2)


def _decay_survivors(
    tracks: TrackDict,
    matched: frozenset = frozenset(),
) -> TrackDict:
    """Decay patience on every unmatched track; return those still alive."""
    survivors: TrackDict = {}
    for tid, track in tracks.items():
        if tid in matched:
            continue
        track.decay_patience()
        if track.is_alive:
            survivors[tid] = track
    return survivors


def match_tracks(
    current_faces: list,
    prev_tracks: TrackDict,
    frame_counter: int,
    now: float | None = None,
) -> TrackDict:
    """
    Match current-frame face locations to existing tracks by centroid distance.
    - On detection frames: matches detections to tracks
    - On skip frames: decays patience, keeps tracks alive
    Returns updated {track_id: FaceTrack} dict.

    ``now`` is the wall-clock time (seconds) stamped onto each trajectory
    sample; behavior analysis uses it to derive speed and dwell time. When
    omitted it defaults to ``time.time()``.
    """
    # On skip frames, just decay all tracks and return
    if frame_counter % TRACKING_SKIP_FRAMES != 0:
        return _decay_survivors(prev_tracks)

    if now is None:
        now = time.time()

    new_tracks: TrackDict = {}
    matched = set()
    next_id = max(prev_tracks.keys(), default=-1) + 1

    for loc, name, conf in current_faces:
        cx, cy = _centroid(loc)
        candidates = (
            (math.hypot(cx - tcx, cy - tcy), tid)
            for tid, track in prev_tracks.items()
            if tid not in matched
            for tcx, tcy in (_centroid(track.last_seen),)
        )
        best_dist, best_id = min(
            candidates, default=(MATCH_DISTANCE_PX, -1)
        )
        if best_dist < MATCH_DISTANCE_PX:
            matched.add(best_id)
            track = prev_tracks[best_id]
            track.update(loc, name, conf, now=now)
            new_tracks[best_id] = track
        else:
            new_tracks[next_id] = FaceTrack(loc, name, conf, now=now)
            next_id += 1

    # Keep unmatched tracks alive (patience decay)
    new_tracks.update(_decay_survivors(prev_tracks, matched))

    return new_tracks
