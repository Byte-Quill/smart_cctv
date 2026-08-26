"""On-screen HUD rendering helpers.

All image-drawing for the monitoring window lives here so ``main.py`` only
has to describe *what* to show, not *how* to draw it.
"""

import cv2

_YELLOW = (200, 150, 0)
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_AMBER = (0, 255, 255)
_WHITE = (255, 255, 255)


def draw_face_boxes(frame, faces) -> None:
    """Draw tracked faces directly on *frame*.

    ``faces`` is an iterable of ``((lx, ty, rx, by), label, color, conf)``
    where the box is in full-resolution pixels. A confidence bar is drawn
    under each box (green / amber / red).
    """
    for ((lx, ty, rx, by), label, color, conf) in faces:
        cv2.rectangle(frame, (lx, ty), (rx, by), color, 2)
        cv2.putText(frame, label, (lx, max(30, ty - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Confidence bar (green/amber/red)
        bar_len = int((rx - lx) * conf)
        bar_color = (
            _GREEN if conf > 0.6
            else _AMBER if conf > 0.3
            else _RED
        )
        cv2.rectangle(
            frame, (lx, by + 6), (lx + bar_len, by + 14),
            bar_color, -1
        )


def draw_countdown(frame, remaining: float, x: int = 20, y: int = 40) -> None:
    """Overlay the 'UNKNOWN - X.Xs' alarm countdown."""
    cv2.putText(
        frame,
        f"UNKNOWN - {remaining:.1f}s",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        _RED,
        3
    )


def draw_family_text(frame, names: list, x: int = 20, y: int = 80) -> None:
    """Draw the recognized family-member banner at the top left."""
    cv2.putText(
        frame,
        "Family: " + ", ".join(names),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        _GREEN,
        2
    )


def draw_mode(frame, night_mode: bool, x: int = 20) -> None:
    """Draw the 'DAY MODE' / 'NIGHT SECURITY' line at the bottom."""
    text = "NIGHT SECURITY MODE" if night_mode else "DAY MODE"
    color = _YELLOW if night_mode else _GREEN
    cv2.putText(
        frame,
        text,
        (x, frame.shape[0] - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


def draw_status(frame, alarm_active: bool, x: int = 20) -> None:
    """Draw the 'ALARM ACTIVE' / 'SYSTEM OK' line at the bottom."""
    text = "ALARM ACTIVE" if alarm_active else "SYSTEM OK"
    color = _RED if alarm_active else _GREEN
    cv2.putText(
        frame,
        text,
        (x, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


def draw_fps(frame, fps: float) -> None:
    """Draw the live FPS counter in the top-right corner."""
    text = f"{fps:.0f} FPS"
    color = _GREEN if fps >= 15 else _AMBER if fps >= 8 else _RED
    (w, _), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    cv2.putText(
        frame,
        text,
        (frame.shape[1] - w - 15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


def draw_unknown_alert(frame) -> None:
    """Draw a prominent red 'UNKNOWN PERSON DETECTED' banner, top center."""
    text = "UNKNOWN PERSON DETECTED"
    (w, h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    x = (frame.shape[1] - w) // 2
    y = 35

    # Dark translucent backdrop so the red text stays readable
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 12, y - h - 10),
        (x + w + 12, y + 10),
        (0, 0, 60),
        -1
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        _RED,
        2
    )