# Smart CCTV — Face Recognition Security System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/OpenCV-5.x-5C3EE8.svg?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Face_Recognition-1.3-00A4EF.svg" alt="face_recognition 1.3">
  <img src="https://img.shields.io/badge/Ultralytics-YOLOv8n-111F68.svg" alt="Ultralytics YOLOv8n">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
</p>

Real-time facial recognition surveillance: identify family members, detect intruders, and trigger alarms — using nothing more than a webcam.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Family Registration](#family-registration)
- [Running the System](#running-the-system)
- [Night Mode](#night-mode)
- [Animal Detection](#animal-detection)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Database Schema](#database-schema)
- [Keyboard Controls](#keyboard-controls)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

Smart CCTV turns a webcam into an intelligent security system. It:

- **Learns** family members' faces through a guided registration process
- **Monitors** the camera feed in real time
- **Recognizes** registered individuals and logs their presence
- **Detects** unknown people, saves snapshots, and sounds an audible siren
- **Suppresses false alarms** caused by animals using YOLOv8 object detection
- **Adapts** its behavior based on time of day, with faster alerts at night

All events are logged to a SQLite database (`logs/events.db`) and a text log file (`logs/security.log`) for review.

---

## Features

### Face Detection

| Feature | Description |
| --- | --- |
| HOG detection | Fast real-time face detection on scaled frames |
| CNN fallback | Deep-learning detection when HOG finds nothing |
| Min face size filter | Rejects tiny false-positive detections |
| Dual-resolution scanning | Detection at 50% scale with optional full-res CNN |

### Image Preprocessing

| Technique | Benefit |
| --- | --- |
| Auto gamma correction | Brightens dark frames, dims overexposed ones |
| CLAHE contrast enhancement | Improves face visibility in shadows and glare |
| Bilateral denoising | Reduces noise while preserving edges |

### Temporal Tracking

| Technique | Benefit |
| --- | --- |
| Majority-vote ensemble | Identity decided over a window of frames, not one |
| Centroid track matching | Stable identity across detection gaps |
| EMA box smoothing | Steady bounding boxes, less jitter |
| Frame-skip detection | Full detection every N frames; tracks persist between |

### Registration (`register.py`)

- **Auto-capture** — saves photos automatically once the subject holds still
- **Blur detection** — rejects blurry photos (Laplacian variance threshold)
- **Brightness check** — rejects too-dark or too-light captures
- **Duplicate rejection** — skips photos too similar to already-captured poses
- **Quality overlay** — green box for good frames, red for rejected, plus a progress bar

### Alarm Behavior

- Looping WAV siren when an unknown person lingers past the delay
- Automatic stop when the area is clear; manual silence with `S`
- Snapshots of unknown individuals saved at configurable intervals
- Animal-only scenes suppress the siren entirely

### Event Logging

- **SQLite database** (`logs/events.db`) — structured, queryable history
- **Text log** (`logs/security.log`) — human-readable audit trail
- Rate-limited family sighting entries to keep the database lean

#### Event Types

| Event | Description |
| --- | --- |
| `SIREN_ON` | Siren alarm activated |
| `SIREN_OFF` | Siren alarm deactivated |
| `UNKNOWN_CONFIRMED` | Unknown person detected and confirmed |
| `UNKNOWN_SNAPSHOT` | Snapshot saved of unknown person |
| `FAMILY_SIGHTING` | Recognized family member spotted |

---

## Requirements

| Dependency | Purpose |
| --- | --- |
| Python 3.8+ | Runtime |
| Webcam (USB or built-in) | Video input |
| `opencv-python` | Camera capture, image processing |
| `face_recognition` | Face detection and recognition |
| `pygame` | Siren audio playback |
| `numpy` | Numerical operations |
| `ultralytics` | YOLOv8 animal/human object detection |
| `setuptools<81` (Python 3.12+) | Provides `pkg_resources` for `face_recognition_models` |

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Register a family member
python register.py

# 4. Run the system
python main.py
```

> **Python 3.12+ note:** `face_recognition_models` depends on the deprecated
> `pkg_resources` module. Pinning `setuptools<81` restores it. A harmless
> deprecation warning may appear on import.

---

## Family Registration

Before running the main system, register the people it should recognize:

```bash
python register.py
```

1. **Enter name** — type the person's name when prompted
2. **Auto-capture** — look at the camera and hold still. The system captures **10 photos**, each required to be:
   - large enough (minimum face height)
   - sharp (passes the blur threshold)
   - well lit (brightness within range)
   - sufficiently different from previous captures
3. **Quality feedback** — green bounding box for good frames, red for rejected, with a capture progress bar
4. **Review results** — accepted and skipped counts are shown on screen

**Tips for best results**

- Use even lighting (natural daylight is ideal)
- Look directly at the camera
- Move your head slightly between captures (turn, tilt)
- Keep expressions neutral
- Register from multiple angles

Photos are stored in `family/<name>/` — one folder per person.

---

## Running the System

```bash
python main.py
```

Startup prints a summary:

```
SMART CCTV
--------------------------------
Family face samples: 20
Recognition tolerance: 0.45
Unknown delay: 10s
Allowed time: 06:00 - 22:00
--------------------------------
```

Per frame, the system:

1. **Enhances** the image (gamma, CLAHE, denoise)
2. **Detects** faces (HOG, with CNN fallback)
3. **Recognizes** faces against known encodings
4. **Tracks** identities with majority vote and smoothed boxes
5. **Logs** family sightings (rate-limited)
6. **Tracks** unknown individuals with a countdown timer
7. **Saves** snapshots of unknown people
8. **Triggers** the siren if an unknown person stays too long

On-screen display: green boxes for family, red boxes for unknown, status overlay.

---

## Night Mode

When the current time is outside the `ALLOWED_START_HOUR`–`ALLOWED_END_HOUR` window:

- The display switches to **NIGHT MODE** (amber text)
- The unknown alert delay drops from `UNKNOWN_DELAY_SECONDS` (10s) to `NIGHT_UNKNOWN_DELAY_SECONDS` (2s)

> **Note on the siren:** by design, the audible siren only sounds during
> **allowed hours**. Outside that window the system still confirms unknowns,
> saves snapshots, and shows the countdown, but it will **not** trigger the
> loud siren (to avoid disturbing neighbors overnight). If you want around-
> the-clock siren coverage, extend `ALLOWED_END_HOUR` to `24` (or set
> `ALLOWED_START_HOUR`/`ALLOWED_END_HOUR` to span the full day).

This provides rapid response during nighttime hours when intrusions are most likely.

---

## Animal Detection

A bundled YOLOv8n model classifies the scene while an unknown face lingers:

- **Animal-only scene** (and no human) — the siren is suppressed
- **Human confirmed** — the alarm delay shortens to `UNKNOWN_HUMAN_DELAY_SECONDS` (1s)
- **Neither** — the normal delay applies

To save CPU, YOLO runs only every `YOLO_SKIP_FRAMES` frames and only while an unknown face is present. If `ultralytics` or the weights file is unavailable, the system logs a warning and continues with face recognition alone.

---

## System Architecture

```
Camera (720p)
    |
    v
Image Enhancement          gamma -> CLAHE -> bilateral denoise
    |
    v
Face Detection             HOG @ 50% scale, CNN fallback full-res
    |
    v
Face Encoding              128-dimensional vector per face
    |
    v
Recognition                face_distance() vs known encodings
    |                      distance <= 0.45 -> FAMILY
    |                      distance >  0.45 -> UNKNOWN
    +------------------+------------------+
    |                                     |
    v                                     v
FAMILY                                UNKNOWN
Log sighting (rate-limited)           confirm over N frames
Draw green box                        countdown timer
                                      snapshots every 5s
                                      YOLO check: animal -> suppress
                                      siren after delay
```

All pipeline stages are separate modules in the `cctv/` package; `main.py` is a thin orchestrator that wires them together.

---

## Project Structure

```
.
├── config.py           # All configurable settings and thresholds
├── main.py             # Thin orchestrator: wires the cctv/ modules into the loop
├── register.py         # Family member registration tool
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT license
├── README.md           # This file
├── .gitignore          # Git ignore rules
├── yolov8n.pt          # Bundled YOLOv8n weights (animal/human detection)
│
├── cctv/               # Modular pipeline components
│   ├── __init__.py
│   ├── enhance.py      # Frame preprocessing (gamma, CLAHE, denoise)
│   ├── faces.py        # Face detection, encoding, recognition
│   ├── tracking.py     # Temporal tracking and majority-vote identity
│   ├── siren.py        # Siren audio control
│   ├── yolo.py         # YOLOv8 animal/human detection (alarm suppression)
│   ├── quality.py      # Registration quality checks (blur, brightness, dupes)
│   └── storage.py      # SQLite + file event logging
│
├── family/             # Registered family photos (one folder per person)
│   └── <name>/
│       └── face_01_20260824_120000.jpg
│
├── snapshots/          # Unknown-person snapshots
│   └── unknown_20260824_220000.jpg
│
├── logs/               # Security events
│   ├── security.log    # Human-readable log
│   └── events.db       # SQLite database
│
├── sounds/             # Audio files
│   └── siren.wav       # Siren alarm sound
│
└── .venv/              # Python virtual environment (not tracked)
```

---

## Configuration Reference

All tunable parameters live in `config.py`.

### Camera

| Setting | Default | Description |
| --- | --- | --- |
| `CAMERA_INDEX` | `0` | Camera device index (`0` = built-in/webcam) |

### Recognition

| Setting | Default | Description |
| --- | --- | --- |
| `FACE_TOLERANCE` | `0.45` | Max Euclidean distance for a match (lower = stricter) |
| `UNKNOWN_CONFIRMATIONS` | `5` | Consecutive frames to confirm an unknown person |

### Alarm Timing

| Setting | Default | Description |
| --- | --- | --- |
| `UNKNOWN_DELAY_SECONDS` | `10` | Seconds an unknown must linger before siren (daytime) |
| `NIGHT_UNKNOWN_DELAY_SECONDS` | `2` | Seconds before siren outside allowed hours |
| `UNKNOWN_HUMAN_DELAY_SECONDS` | `1` | Seconds before siren when YOLO confirms a human |

### Operating Hours

| Setting | Default | Description |
| --- | --- | --- |
| `ALLOWED_START_HOUR` | `6` | Hour monitoring begins (24h) |
| `ALLOWED_END_HOUR` | `22` | Hour monitoring ends (24h) |

### Snapshots and Logging

| Setting | Default | Description |
| --- | --- | --- |
| `SNAPSHOT_INTERVAL` | `5` | Seconds between unknown-person snapshots |
| `SIGHTING_LOG_INTERVAL` | `30` | Min seconds between FAMILY_SIGHTING logs per person |

### Detection

| Setting | Default | Description |
| --- | --- | --- |
| `DETECTION_SCALE` | `0.5` | Face detection resolution (0.5 = half-size) |
| `MIN_FACE_SIZE` | `40` | Minimum face height (px) at detection scale |
| `ENABLE_CNN_FALLBACK` | `True` | Use CNN when HOG finds nothing |
| `ANIMAL_DETECTION_ENABLED` | `True` | Enable YOLO animal/human classification |
| `YOLO_SKIP_FRAMES` | `3` | Run YOLO every N frames while an unknown lingers |

### Tracking

| Setting | Default | Description |
| --- | --- | --- |
| `ENSEMBLE_FRAMES` | `5` | Majority-vote window per tracked face |
| `TRACKING_SKIP_FRAMES` | `2` | Run full face detection every N frames |
| `TRACKING_SMOOTH_ALPHA` | `0.6` | EMA alpha for bounding box smoothing |
| `TRACKING_PATIENCE` | `5` | Frames to keep a track alive after disappearance |

### Registration Quality

| Setting | Default | Description |
| --- | --- | --- |
| `MIN_REG_FACE_SIZE` | `80` | Minimum face height (px, full-res) |
| `BLUR_THRESHOLD` | `80` | Laplacian variance floor (lower = blurrier) |
| `MIN_BRIGHTNESS` | `40` | Minimum mean pixel brightness (0–255) |
| `MAX_BRIGHTNESS` | `215` | Maximum mean pixel brightness (0–255) |
| `MIN_ENCODING_DISTANCE` | `0.25` | Min distance to reject duplicate poses |
| `AUTO_CAPTURE_STABLE_FRAMES` | `8` | Good frames needed before auto-capture |

### Folders and Audio

| Setting | Default | Description |
| --- | --- | --- |
| `FAMILY_DIR` | `"family"` | Registered family photos |
| `SNAPSHOT_DIR` | `"snapshots"` | Unknown-person snapshots |
| `LOG_DIR` | `"logs"` | Security logs and database |
| `SIREN_FILE` | `"sounds/siren.wav"` | Path to siren WAV file |

---

## Database Schema

**File:** `logs/events.db`

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    person TEXT,
    snapshot TEXT
);
```

Example query:

```bash
sqlite3 logs/events.db "SELECT * FROM events ORDER BY id DESC LIMIT 10;"
```

---

## Keyboard Controls

| Key | Function |
| --- | --- |
| `Q` | Quit the system |
| `S` | Stop / silence the siren |

---

## Troubleshooting

### "Import could not be resolved" (Pylance)

The venv interpreter is selected but Pylance has not updated yet. Select the venv manually:

1. `Ctrl+Shift+P` → `Python: Select Interpreter`
2. Choose `./.venv/bin/python`

### "Please install face_recognition_models"

```bash
pip install "setuptools<81"
```

This restores the `pkg_resources` module needed by the older `face_recognition_models` package.

### Siren not working

- Ensure a WAV file exists at the path in `SIREN_FILE` (default: `sounds/siren.wav`)
- The system runs without a siren — it simply skips audio playback

### Camera not opening

- Check `CAMERA_INDEX` in `config.py` — try `1` for external USB cameras
- Ensure no other application is using the camera

### Poor recognition

- Register more photos per person (aim for 10+ with varied angles)
- Improve lighting during registration and monitoring
- Lower `FACE_TOLERANCE` for stricter matching (try 0.4)
- Ensure photos contain exactly one face each
- CNN fallback can help with difficult angles

### YOLO warning at startup

If `ultralytics` is missing or `yolov8n.pt` cannot be loaded, animal suppression is disabled automatically and the system continues with face recognition only. Reinstall with `pip install -r requirements.txt` to restore it.

---

## License

This project is open source and available under the [MIT License](LICENSE).
