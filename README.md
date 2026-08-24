# Smart CCTV

A face-recognition based smart CCTV system built with Python.
It watches a camera feed, recognizes registered family members, logs
security events to a SQLite database, saves snapshots of unknown people,
and can trigger an audible siren when an unregistered person is detected.

## Features

- Real-time face detection and recognition using `face_recognition` + OpenCV
- Family member registration via a webcam photo capture tool
- SQLite event logging (`logs/events.db`) and file logging (`logs/security.log`)
- Automatic snapshots of unknown people (`snapshots/`)
- Configurable siren alarm triggered on unrecognized persons
- All tuning values live in one place: `config.py`

## Requirements

- Python 3.8+
- A working webcam

Python packages (install into a virtual environment, see below):

- `opencv-python`
- `face_recognition`
- `pygame`
- `numpy`
- `setuptools<81` (needed by `face_recognition_models` on Python 3.12+)

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the dependencies
pip install opencv-python face_recognition pygame numpy "setuptools<81"
```

> Note: On Python 3.12+, install `setuptools<81` so that the
> `pkg_resources` module (required by `face_recognition_models`) is available.

## Registering family members

Before running the main system, register the people that should be recognized:

```bash
python register.py
```

Follow the on-screen instructions:

1. Enter the family member's name.
2. Look directly at the camera.
3. Press **SPACE** to capture (10 photos per person are recommended).
4. Press **Q** to quit.

Captured photos are stored under `family/<name>/`.

## Running the system

```bash
python main.py
```

The system:

- Loads all registered faces from `family/`
- Detects faces in the camera feed
- Recognizes family members and logs their presence
- Flags unknown people, saves snapshots to `snapshots/`, and triggers the siren
- Logs events to the console, `logs/security.log`, and `logs/events.db`

## Configuration

All settings live in `config.py`:

| Setting                                   | Purpose                                     |
| ----------------------------------------- | ------------------------------------------- |
| `CAMERA_INDEX`                            | Which camera to use (`0` = built-in/webcam) |
| `FACE_TOLERANCE`                          | Max distance for a face match               |
| `REQUIRED_CONFIRMATIONS`                  | Frames before accepting a family member     |
| `UNKNOWN_CONFIRMATIONS`                   | Frames before treating someone as unknown   |
| `UNKNOWN_DELAY_SECONDS`                   | How long an unknown must stay before alarm  |
| `ALLOWED_START_HOUR` / `ALLOWED_END_HOUR` | Allowed operating hours                     |
| `SNAPSHOT_INTERVAL`                       | Seconds between unknown snapshots           |
| `FRAME_SCALE`                             | Scale factor to speed up recognition        |
| `FAMILY_DIR` / `SNAPSHOT_DIR` / `LOG_DIR` | Folder locations                            |
| `SIREN_FILE`                              | Alarm sound file path                       |

## Project structure

```
config.py   - all configurable settings
main.py     - the CCTV monitoring / recognition loop
register.py - family member registration tool
```
