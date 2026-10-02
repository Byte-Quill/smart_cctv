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
        # Unknown faces carry an empty label — the red 'UNKNOWN PERSON
        # DETECTED' banner is their only on-screen warning.
        if label:
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


def draw_countdown(frame, remaining: float) -> None:
    """Overlay the siren countdown, centered below the unknown banner.

    The red 'UNKNOWN PERSON DETECTED' banner is the single unknown
    warning, so this only shows the time left before the siren fires.
    It is drawn on its own line (y = 85) so the two texts never overlap.
    """
    text = f"SIREN IN {remaining:.1f}s"
    (w, _), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3
    )
    x = (frame.shape[1] - w) // 2
    cv2.putText(
        frame,
        text,
        (x, 85),
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

    # Dark translucent backdrop so the red text stays readable. Blend only
    # the banner's rectangle (clipped to the frame): pixel-identical to
    # blending a full-frame copy, but far less work during an alert.
    # cv2.rectangle fills inclusively of the corner pixel, so the region is
    # one pixel larger than the (x - 12 .. x + w + 12) span.
    rx0 = max(0, x - 12)
    ry0 = max(0, y - h - 10)
    rx1 = min(frame.shape[1], x + w + 12 + 1)
    ry1 = min(frame.shape[0], y + 10 + 1)
    if rx1 > rx0 and ry1 > ry0:
        roi = frame[ry0:ry1, rx0:rx1]
        overlay = roi.copy()
        cv2.rectangle(
            overlay, (0, 0), (rx1 - rx0, ry1 - ry0), (0, 0, 60), -1
        )
        frame[ry0:ry1, rx0:rx1] = cv2.addWeighted(
            overlay, 0.65, roi, 0.35, 0.0
        )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        _RED,
        2
    )


def draw_add_button(frame, hover: bool = False):
    """Draw the clickable '+ ADD FAMILY' button, bottom right.

    Returns the button rectangle as ``(x0, y0, x1, y1)`` so the caller
    can hit-test mouse clicks against it. When *hover* is True the
    button brightens to give visual feedback.
    """
    text = "+ ADD FAMILY"
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    pad_x, pad_y = 14, 10
    bw, bh = tw + 2 * pad_x, th + 2 * pad_y
    x1 = frame.shape[1] - 15
    y1 = frame.shape[0] - 15
    x0, y0 = x1 - bw, y1 - bh

    fill = (0, 140, 0) if hover else (0, 100, 0)
    border = _GREEN if hover else (0, 180, 0)

    # Slightly translucent fill so the button reads as a UI element.
    # Only the button's rectangle is copied and blended: this is
    # pixel-identical to blending a full-frame copy (outside the rectangle
    # the blend is a no-op), but avoids copying/blending the whole frame
    # on every single frame. The blend result is contiguous and written
    # back with numpy, so OpenCV never gets an aliased/strided dst.
    bx0, by0 = max(0, x0), max(0, y0)
    bx1, by1 = min(frame.shape[1], x1 + 1), min(frame.shape[0], y1 + 1)
    if bx1 > bx0 and by1 > by0:
        roi = frame[by0:by1, bx0:bx1]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (bx1 - bx0, by1 - by0), fill, -1)
        frame[by0:by1, bx0:bx1] = cv2.addWeighted(
            overlay, 0.85, roi, 0.15, 0.0
        )
    cv2.rectangle(frame, (x0, y0), (x1, y1), border, 2)

    cv2.putText(
        frame,
        text,
        (x0 + pad_x, y1 - pad_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        _WHITE,
        2
    )

    return (x0, y0, x1, y1)