# 🚨 Smart CCTV — Face Recognition Security System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0-5C3EE8?logo=opencv&logoColor=white)
![Face Recognition](https://img.shields.io/badge/Face_Recognition-1.3-00A4EF)
![License](https://img.shields.io/badge/license-MIT-green)

**Real-time facial recognition surveillance — identify family, detect intruders, trigger alarms.**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Requirements](#-requirements)
- [Quick Start](#-quick-start)
- [Step-by-Step Setup](#-step-by-step-setup)
- [Family Registration](#-family-registration)
- [Running the System](#-running-the-system)
- [Night Mode](#-night-mode)
- [System Architecture](#-system-architecture)
- [Configuration Reference](#-configuration-reference)
- [Database Schema](#-database-schema)
- [Keyboard Controls](#-keyboard-controls)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

Smart CCTV transforms your webcam into an intelligent security system. It:

- **Learns** family members' faces through a guided registration process
- **Monitors** the camera feed in real-time
- **Recognizes** registered individuals and logs their presence
- **Detects** unknown people, saves snapshots, and sounds an audible siren alarm
- **Adapts** its behavior based on time of day — faster alerts at night

All events are logged to both a SQLite database (`logs/events.db`) and a text log file (`logs/security.log`) for review.

---

## ✨ Features

### 🔍 Intelligent Face Detection

| Feature                      | Description                                                                                                                |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **HOG detection**            | Fast real-time face detection on scaled frames                                                                             |
| **CNN fallback**             | Automatically switches to deep-learning CNN detection when HOG finds nothing (uses bundled `mmod_human_face_detector.dat`) |
| **Min face size filter**     | Rejects tiny false-positive detections                                                                                     |
| **Dual-resolution scanning** | Detection runs at 50% scale with optional full-res CNN                                                                     |

### 🧠 Smart Preprocessing

| Technique                      | Benefit                                      |
| ------------------------------ | -------------------------------------------- |
| **CLAHE contrast enhancement** | Improves face visibility in shadows & glare  |
| **Auto gamma correction**      | Brightens dark frames, dims overexposed ones |
| **Bilateral denoising**        | Reduces noise while preserving edges         |

### 📸 Enhanced Registration (`register.py`)

- **Auto-capture** — automatically saves photos once the subject holds still
- **Blur detection** — rejects blurry photos (Laplacian variance threshold)
- **Brightness check** — rejects too-dark or too-light captures
- **Duplicate rejection** — skips photos too similar to already-captured ones
- **Quality overlay** — green box = good, red = rejected, plus quality score
- **Progress bar** — visual indicator of capture readiness

### 🌙 Night Mode

Outside configured hours (default 10 PM – 6 AM):

- Siren triggers in **just 2 seconds** instead of the normal 10
- On-screen status changes to **🌙 NIGHT MODE** with amber coloring
- Fully configurable via `NIGHT_UNKNOWN_DELAY_SECONDS`

### 🔊 Siren Alarm

- Plays a looping WAV siren when an unknown person lingers too long
- Automatically stops when the area is clear
- Press `S` to silence at any time

### 📊 Event Logging

- **SQLite database** (`logs/events.db`) — structured queryable history
- **Text log** (`logs/security.log`) — human-readable audit trail
- **Snapshot capture** — saves photos of unknown individuals at configurable intervals

### Event Types Recorded

| Event               | Description                           |
| ------------------- | ------------------------------------- |
| `SIREN_ON`          | Siren alarm activated                 |
| `SIREN_OFF`         | Siren alarm deactivated               |
| `UNKNOWN_CONFIRMED` | Unknown person detected and confirmed |
| `UNKNOWN_SNAPSHOT`  | Snapshot saved of unknown person      |
| `FAMILY_SIGHTING`   | Recognized family member spotted      |

---

## 📦 Requirements

| Dependency         | Minimum         | Purpose                                                |
| ------------------ | --------------- | ------------------------------------------------------ |
| Python             | 3.8+            | Runtime                                                |
| Webcam             | USB or built-in | Video input                                            |
| `opencv-python`    | —               | Camera capture, image processing                       |
| `face_recognition` | 1.3             | Face detection & recognition                           |
| `pygame`           | 2.x             | Audio playback for siren                               |
| `numpy`            | —               | Numerical operations                                   |
| `setuptools<81`    | (Python 3.12+)  | Provides `pkg_resources` for `face_recognition_models` |

---

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install opencv-python face_recognition pygame numpy "setuptools<81"

# 3. Register a family member
python register.py

# 4. Run the system
python main.py
```

---

## 🛠 Step-by-Step Setup

### 1. Create & activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install packages

```bash
pip install opencv-python face_recognition pygame numpy "setuptools<81"
```

> **Python 3.12+ note:** `face_recognition_models` depends on the deprecated `pkg_resources` module, removed from recent `setuptools`. Pinning `setuptools<81` restores it. A harmless deprecation warning appears on import.

### 3. Configure settings (optional)

Edit `config.py` to adjust:

- **Camera index** (`CAMERA_INDEX`) — change from `0` if using an external camera
- **Face tolerance** (`FACE_TOLERANCE`) — lower = stricter matching
- **Operating hours** — `ALLOWED_START_HOUR` / `ALLOWED_END_HOUR`
- **Night mode delay** — `NIGHT_UNKNOWN_DELAY_SECONDS`

See full [Configuration Reference](#-configuration-reference).

---

## 👨‍👩‍👧‍👦 Family Registration

Before running the main system, register the people your system should recognize:

```bash
python register.py
```

### What happens

1. **Enter name** — type the person's name when prompted
2. **Auto-capture** — look at the camera and hold still. The system automatically captures **10 photos** when:
   - A face is detected
   - The image isn't blurry
   - The lighting is adequate
   - The pose is sufficiently different from previous captures
3. **Quality feedback** — green bounding box = good quality; a progress bar fills as stable frames accumulate
4. **Review results** — accepted/skipped counts shown on-screen

### Tips for best results

- Use good, even lighting (natural daylight is ideal)
- Look directly at the camera
- Move your head slightly between captures (turn left/right, tilt up/down)
- Avoid extreme expressions (stick to neutral/slight smile)
- Register from multiple angles if possible

Photos are stored in `family/<name>/` — one folder per person.

---

## 🎬 Running the System

```bash
python main.py
```

### What happens

```
SMART CCTV
--------------------------------
Family face samples: 20
Recognition tolerance: 0.45
Unknown delay: 10s
Allowed time: 06:00 - 22:00
--------------------------------
```

1. **Startup** — loads all registered faces from `family/`, initializes the camera, and prints a summary
2. **Monitoring** — continuously processes frames:
   - **Enhances** the image (CLAHE + gamma + denoise)
   - **Detects** faces (HOG, with CNN fallback)
   - **Recognizes** faces by comparing against known encodings
   - **Logs** family sightings to the database
   - **Tracks** unknown individuals with countdown timer
   - **Saves** snapshots of unknown people
   - **Triggers** siren if an unknown person stays too long
3. **On-screen display** — bounding boxes (green = family, red = unknown), status overlay

---

## 🌙 Night Mode

When the current time is **outside** the `ALLOWED_START_HOUR`–`ALLOWED_END_HOUR` window:

- The display switches to **🌙 NIGHT MODE** (amber text)
- The unknown alert delay drops from `UNKNOWN_DELAY_SECONDS` (10s) to `NIGHT_UNKNOWN_DELAY_SECONDS` (2s)
- This provides **rapid response** during nighttime hours when intrusions are most likely

Configure both delays in `config.py`.

---

## 🏗 System Architecture

```
                    ┌─────────────────────────────────────┐
                    │           Camera (720p)              │
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────┐
                    │      Image Enhancement               │
                    │  ┌─────┐  ┌─────┐  ┌───────────┐   │
                    │  │Gamma│→ │CLAHE│→ │Denoising  │   │
                    │  └─────┘  └─────┘  └───────────┘   │
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────┐
                    │      Face Detection                  │
                    │  ┌────────┐    ┌─────────────┐      │
                    │  │HOG 50% │───→│CNN fallback  │      │
                    │  │ scale  │    │(full-res)    │      │
                    │  └────────┘    └─────────────┘      │
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────┐
                    │      Face Encoding                   │
                    │  128-dimensional vector per face     │
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────┐
                    │      Recognition                     │
                    │  face_distance() vs known encodings  │
                    │  ↓ euclidean distance ↓              │
                    │  ┌> FAMILY  if distance ≤ 0.45      │
                    │  └> UNKNOWN if distance >  0.45      │
                    └──────┬─────────────────┬────────────┘
                           │                 │
                    ┌──────▼──────┐   ┌──────▼──────────┐
                    │  FAMILY     │   │  UNKNOWN         │
                    │  Log sight- │   │  ┌→ countdown    │
                    │  ing to DB  │   │  ├→ snapshot     │
                    │  Draw green │   │  ├→ siren alarm  │
                    │  bounding   │   │  └→ draw red box │
                    │  box        │   │                  │
                    └─────────────┘   └─────────────────┘
```

---

## ⚙️ Configuration Reference

All tunable parameters in `config.py`:

### Camera & Video

| Setting        | Default | Description                                 |
| -------------- | ------- | ------------------------------------------- |
| `CAMERA_INDEX` | `0`     | Camera device index (`0` = built-in/webcam) |

### Recognition

| Setting                  | Default | Description                                           |
| ------------------------ | ------- | ----------------------------------------------------- |
| `FACE_TOLERANCE`         | `0.45`  | Max Euclidean distance for a match (lower = stricter) |
| `REQUIRED_CONFIRMATIONS` | `5`     | Consecutive frames to accept a family member          |
| `UNKNOWN_CONFIRMATIONS`  | `5`     | Consecutive frames to confirm an unknown person       |

### Alarm Timing

| Setting                       | Default | Description                                             |
| ----------------------------- | ------- | ------------------------------------------------------- |
| `UNKNOWN_DELAY_SECONDS`       | `10`    | Seconds an unknown must linger before siren (daytime)   |
| `NIGHT_UNKNOWN_DELAY_SECONDS` | `2`     | Seconds an unknown must linger before siren (nighttime) |

### Operating Hours

| Setting              | Default | Description                  |
| -------------------- | ------- | ---------------------------- |
| `ALLOWED_START_HOUR` | `6`     | Hour monitoring begins (24h) |
| `ALLOWED_END_HOUR`   | `22`    | Hour monitoring ends (24h)   |

### Snapshots

| Setting             | Default | Description                              |
| ------------------- | ------- | ---------------------------------------- |
| `SNAPSHOT_INTERVAL` | `5`     | Seconds between unknown-person snapshots |

### Detection

| Setting               | Default | Description                                 |
| --------------------- | ------- | ------------------------------------------- |
| `FRAME_SCALE`         | `0.25`  | Scale factor for pre-processing preview     |
| `DETECTION_SCALE`     | `0.5`   | Face detection resolution (0.5 = half-size) |
| `MIN_FACE_SIZE`       | `40`    | Minimum face height (px) at detection scale |
| `ENABLE_CNN_FALLBACK` | `True`  | Use CNN when HOG finds nothing              |

### Registration Quality (register.py)

| Setting                      | Default | Description                                 |
| ---------------------------- | ------- | ------------------------------------------- |
| `MIN_REG_FACE_SIZE`          | `80`    | Minimum face height (px, full-res)          |
| `BLUR_THRESHOLD`             | `80`    | Laplacian variance floor (lower = blurrier) |
| `MIN_BRIGHTNESS`             | `40`    | Minimum mean pixel brightness (0–255)       |
| `MAX_BRIGHTNESS`             | `215`   | Maximum mean pixel brightness (0–255)       |
| `MIN_ENCODING_DISTANCE`      | `0.25`  | Min distance to reject duplicate poses      |
| `AUTO_CAPTURE_STABLE_FRAMES` | `8`     | Good frames needed before auto-capture      |

### Folders

| Setting        | Default       | Description                |
| -------------- | ------------- | -------------------------- |
| `FAMILY_DIR`   | `"family"`    | Registered family photos   |
| `SNAPSHOT_DIR` | `"snapshots"` | Unknown-person snapshots   |
| `LOG_DIR`      | `"logs"`      | Security logs and database |

### Audio

| Setting      | Default              | Description            |
| ------------ | -------------------- | ---------------------- |
| `SIREN_FILE` | `"sounds/siren.wav"` | Path to siren WAV file |

---

## 🗄 Database Schema

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

## ⌨️ Keyboard Controls

| Key | Function                |
| --- | ----------------------- |
| `Q` | Quit the system         |
| `S` | Stop/ silence the siren |

---

## 📁 Project Structure

```
.
├── config.py           # All configurable settings & thresholds
├── main.py             # CCTV monitoring / recognition loop
├── register.py         # Family member registration tool
├── README.md           # This file
├── .gitignore          # Git ignore rules
│
├── family/             # Registered family photos (one folder per person)
│   └── <name>/
│       ├── face_01_20260824_120000.jpg
│       └── ...
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

## 🔧 Troubleshooting

### "Import could not be resolved" (Pylance)

The venv interpreter is selected but Pylance hasn't updated yet. Select the venv manually:

1. `Ctrl+Shift+P` → `Python: Select Interpreter`
2. Choose `./.venv/bin/python`

### "Please install `face_recognition_models`"

```bash
pip install "setuptools<81"
```

This restores the `pkg_resources` module needed by the older `face_recognition_models` package.

### Siren not working

- Ensure a WAV file exists at the path specified by `SIREN_FILE` (default: `sounds/siren.wav`)
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

---

## 📄 License

This project is open source and available under the MIT License.
