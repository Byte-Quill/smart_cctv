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
    BEHAVIOR_PACING_SWITCHES,
    BEHAVIOR_PACING_WINDOW,
    BEHAVIOR_VEL_WINDOW,
)

# Max centroid distance (detection-scale px) for a detection to match a
# track. This is the *base* radius — _match_radius() enlarges it for
# big/near faces and fast movers so a sprint across a skip-frame gap keeps
# the same identity instead of spawning a fresh track.
MATCH_DISTANCE_PX = 60

# Minimum IoU (0-1) that rescues a match when the centroid drifted just past
# the radius (e.g. a jumpy HOG box). Pure geometry on boxes already in hand.
MATCH_IOU_RESCUE = 0.15

# EMA weight for the tracker's internal velocity estimate (0-1; higher =
# follows sudden sprints faster but is noisier). Kept separate from
# TRACKING_SMOOTH_ALPHA (box smoothing) — this one smooths *speed*.
_SPEED_SMOOTH_ALPHA = 0.5


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
        # Smoothed centroid speed (detection-scale px/s). Starts at 0.0;
        # seeded from the first real sample pair in _append_sample().
        self._smooth_speed: float = 0.0
        # Short rolling window of per-sample instantaneous speeds (px/s),
        # newest last. The EMA above is the primary velocity signal; this
        # window backs behavior features that need *variance* (still-vs-
        # fidget-vs-walk) instead of a single mean. Bounded to a handful
        # of samples so per-track memory stays flat.
        self._speed_window: deque = deque(maxlen=BEHAVIOR_VEL_WINDOW)
        self.trajectory = deque(maxlen=BEHAVIOR_TRAJECTORY_LEN)
        self._append_sample(location, now)
        # EMA of unit direction (dx, dy) components. Magnitude (0..1) is the
        # straightness confidence: ~1 = consistent heading, ~0 = jitter /
        # random walk. Starts at zero (no data yet).
        self._dir_x: float = 0.0
        self._dir_y: float = 0.0

    def _append_sample(self, location, now=None):
        top, right, bottom, left = location
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        height = max(1, bottom - top)
        self.trajectory.append(
            (cx, cy, time.time() if now is None else now, height)
        )
        # Keep the EMA velocity in sync with the newest pair of samples.
        # Costs one hypot() per update — negligible — and gives the matcher
        # a motion-aware gating radius on the very next detection frame.
        # The instantaneous speed also feeds a short rolling window (for
        # variance-based movement-quality features) and a unit-direction
        # EMA (for straightness confidence): both are O(1) bookkeeping.
        traj = self.trajectory
        if len(traj) >= 2:
            (px, py, pt, _), (cx2, cy2, ct, _) = traj[-2], traj[-1]
            dt = max(1e-3, ct - pt)
            dx, dy = cx2 - px, cy2 - py
            step = math.hypot(dx, dy)
            inst = step / dt
            if self._smooth_speed > 0.0:
                self._smooth_speed = (
                    _SPEED_SMOOTH_ALPHA * inst
                    + (1.0 - _SPEED_SMOOTH_ALPHA) * self._smooth_speed
                )
            else:
                self._smooth_speed = inst
            self._speed_window.append(inst)
            # Direction EMA over unit steps; sub-pixel jitter (< 1.0 px)
            # carries no heading information and would only dilute the
            # straightness confidence, so it is skipped.
            if step >= 1.0:
                ux, uy = dx / step, dy / step
                if self._dir_x == 0.0 and self._dir_y == 0.0:
                    self._dir_x, self._dir_y = ux, uy
                else:
                    self._dir_x = (
                        _SPEED_SMOOTH_ALPHA * ux
                        + (1.0 - _SPEED_SMOOTH_ALPHA) * self._dir_x
                    )
                    self._dir_y = (
                        _SPEED_SMOOTH_ALPHA * uy
                        + (1.0 - _SPEED_SMOOTH_ALPHA) * self._dir_y
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
    def net_delta(self) -> tuple[float, float]:
        """Net (dx, dy) in detection-scale pixels across the whole window."""
        if len(self.trajectory) < 2:
            return (0.0, 0.0)
        x1, y1, _t1, _ = self.trajectory[-1]
        x0, y0, _t0, _ = self.trajectory[0]
        return (x1 - x0, y1 - y0)

    @property
    def direction_switches(self) -> int:
        """Lateral direction reversals inside the recent window.

        Counts sign changes of per-step dx over the last
        ``BEHAVIOR_PACING_WINDOW`` samples (steps smaller than ~5% of a
        face-height are ignored as jitter). Several reversals with little
        net progress is the pacing / casing signature that
        ``behavior.is_pacing()`` reads.
        """
        traj = list(self.trajectory)[-BEHAVIOR_PACING_WINDOW:]
        if len(traj) < 3:
            return 0
        switches = 0
        last_sign = 0
        jitter = max(1.0, self.box_height * 0.05)
        for (x0, *_), (x1, *_) in zip(traj, traj[1:]):
            dx = x1 - x0
            if abs(dx) < jitter:
                continue
            sign = 1 if dx > 0 else -1
            if last_sign != 0 and sign != last_sign:
                switches += 1
            last_sign = sign
        return switches

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
# This is the *base* radius — _match_radius() enlarges it for big/near faces
# and fast movers so a sprint across a skip-frame gap keeps the same identity
# instead of spawning a fresh track.
MATCH_DISTANCE_PX = 60


def _centroid(location) -> tuple[int, int]:
    top, right, bottom, left = location
    return ((left + right) // 2, (top + bottom) // 2)


def _box_iou(box_a, box_b) -> float:
    """Intersection-over-union of two (top, right, bottom, left) boxes."""
    a_top, a_right, a_bottom, a_left = box_a
    b_top, b_right, b_bottom, b_left = box_b
    inter_left = max(a_left, b_left)
    inter_top = max(a_top, b_top)
    inter_right = min(a_right, b_right)
    inter_bottom = min(a_bottom, b_bottom)
    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    a_area = max(0, a_right - a_left) * max(0, a_bottom - a_top)
    b_area = max(0, b_right - b_left) * max(0, b_bottom - b_top)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def _match_radius(track) -> float:
    """Adaptive gating radius for one track: base + size + motion.

    - Big/near faces shift more pixels per step, so the radius grows with
      half the face height.
    - Fast movers cover ground during skip-frame gaps, so the radius grows
      with the EMA speed across one skip interval.
    Returns detection-scale pixels.
    """
    size_term = 0.5 * track.box_height
    motion_term = track._smooth_speed * (TRACKING_SKIP_FRAMES / 10.0)
    return MATCH_DISTANCE_PX + size_term + motion_term


def _box_iou(box_a, box_b) -> float:
    """Intersection-over-union of two (top, right, bottom, left) boxes."""
    a_top, a_right, a_bottom, a_left = box_a
    b_top, b_right, b_bottom, b_left = box_b
    inter_left = max(a_left, b_left)
    inter_top = max(a_top, b_top)
    inter_right = min(a_right, b_right)
    inter_bottom = min(a_bottom, b_bottom)
    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    a_area = max(0, a_right - a_left) * max(0, a_bottom - a_top)
    b_area = max(0, b_right - b_left) * max(0, b_bottom - b_top)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


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
        # Adaptive radius: each candidate track brings its own gate, sized
        # by its face scale + motion. A detection matches the nearest track
        # whose gate contains it; an IoU rescue catches jumpy boxes whose
        # centroid just overshoots. Falls back to a brand-new track.
        best_id, best_dist = -1, float("inf")
        best_iou, iou_id = 0.0, -1
        for tid, track in prev_tracks.items():
            if tid in matched:
                continue
            tcx, tcy = _centroid(track.last_seen)
            dist = math.hypot(cx - tcx, cy - tcy)
            if dist < _match_radius(track) and dist < best_dist:
                best_id, best_dist = tid, dist
            iou = _box_iou(loc, track.last_seen)
            if iou > best_iou:
                best_iou, iou_id = iou, tid
        if best_id >= 0:
            matched.add(best_id)
            track = prev_tracks[best_id]
            track.update(loc, name, conf, now=now)
            new_tracks[best_id] = track
        elif best_iou >= MATCH_IOU_RESCUE and iou_id not in matched:
            matched.add(iou_id)
            track = prev_tracks[iou_id]
            track.update(loc, name, conf, now=now)
            new_tracks[iou_id] = track
        else:
            new_tracks[next_id] = FaceTrack(loc, name, conf, now=now)
            next_id += 1

    # Keep unmatched tracks alive (patience decay)
    new_tracks.update(_decay_survivors(prev_tracks, matched))

    return new_tracks
