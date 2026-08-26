"""Temporal face tracking: smoothed boxes and majority-vote identity."""

from collections import Counter, deque

import numpy as np

from config import (
    ENSEMBLE_FRAMES,
    TRACKING_SKIP_FRAMES,
    TRACKING_SMOOTH_ALPHA,
    TRACKING_PATIENCE,
    IDENTITY_MIN_VOTES,
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
        return float(np.mean(self.confidences))


class FaceTrack:
    """Track state for one face: smoothed location, history, patience."""

    def __init__(self, location, name: str, confidence: float):
        self.history = FaceHistory()
        self.history.add(name, confidence)
        # Smoothed location (detection-scale coords)
        self.smoothed = location
        self.patience = TRACKING_PATIENCE  # frames remaining before expiry
        self.last_seen = location

    def update(self, location, name: str, confidence: float):
        self.history.add(name, confidence)
        # EMA smoothing on each coordinate
        a = TRACKING_SMOOTH_ALPHA
        self.smoothed = tuple(
            int(a * loc_coord + (1 - a) * smooth_coord)
            for loc_coord, smooth_coord in zip(location, self.smoothed)
        )
        self.last_seen = location
        self.patience = TRACKING_PATIENCE  # reset patience

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


TrackDict = dict[int, FaceTrack]


def _centroid(location) -> tuple[int, int]:
    top, right, bottom, left = location
    return ((left + right) // 2, (top + bottom) // 2)


def match_tracks(
    current_faces: list,
    prev_tracks: TrackDict,
    _frame_counter: int
) -> TrackDict:
    """
    Match current-frame face locations to existing tracks by centroid distance.
    - On detection frames: matches detections to tracks
    - On skip frames: decays patience, keeps tracks alive
    Returns updated {track_id: FaceTrack} dict.
    """
    new_tracks: TrackDict = {}

    # On skip frames, just decay all tracks and return
    if _frame_counter % TRACKING_SKIP_FRAMES != 0:
        for tid, track in prev_tracks.items():
            track.decay_patience()
            if track.is_alive:
                new_tracks[tid] = track
        return new_tracks

    matched = set()
    next_id = max(prev_tracks.keys(), default=-1) + 1

    for _loc, name, conf in current_faces:
        best_id = -1
        best_dist = 60  # centroid distance threshold (detection-scale pixels)
        cx, cy = _centroid(_loc)
        for tid, track in prev_tracks.items():
            if tid in matched:
                continue
            tcx, tcy = _centroid(track.last_seen)
            d = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_id = tid
        if best_id >= 0:
            matched.add(best_id)
            track = prev_tracks[best_id]
            track.update(_loc, name, conf)
            new_tracks[best_id] = track
        else:
            new_tracks[next_id] = FaceTrack(_loc, name, conf)
            next_id += 1

    # Keep unmatched tracks alive (patience decay)
    for tid, track in prev_tracks.items():
        if tid not in matched:
            track.decay_patience()
            if track.is_alive:
                new_tracks[tid] = track

    return new_tracks
