# Smart CCTV — Architecture Guide

A narrative walkthrough of the system: what it does, how the code is
organized, how a frame travels through the pipeline, and where to look
when something goes wrong.

---

## 1. What the system does

Smart CCTV watches a camera feed and answers one question: **"Is the
person in front of the camera family, or a stranger?"**

- **Family** → their name appears in green, and the sighting is logged.
- **Stranger** → a red `UNKNOWN PERSON DETECTED` banner appears, a
  countdown starts, snapshots are saved, and — if the delay expires
  during allowed hours — a loud siren sounds.

Everything else in the codebase exists to make that one decision fast,
accurate, and reliable on modest hardware.

---

## 2. The two entry points

| Script        | Purpose                                                                         |
| ------------- | ------------------------------------------------------------------------------- |
| `register.py` | Teach the system a face. Captures quality-checked photos into `family/<Name>/`. |
| `main.py`     | Run the surveillance loop. Loads the family database and monitors the camera.   |

Both are thin orchestrators. All real work lives in the `cctv/` package.

---

## 3. The `cctv/` package — one module, one job

```
cctv/
├── enhance.py    → fix brightness/contrast/denoise before detection
├── faces.py      → load family DB, detect faces (HOG/CNN), recognize
├── tracking.py   → smooth boxes over time, majority-vote identity
├── motion.py     → cheap motion gate that skips heavy work when idle
├── yolo.py       → optional animal/human scene analysis (false alarms)
├── siren.py      → looping alarm sound, thread-safe on/off
├── storage.py    → SQLite events DB + text audit log + retention
├── quality.py    → blur/brightness/size/duplicate checks (registration)
└── hud.py        → every on-screen overlay the main loop draws
```

Read them in that order and you have read the whole system.

---

## 4. The life of a frame

`main.py` processes each camera frame through a pipeline ordered
**cheapest first**, so a quiet scene costs almost nothing:

```
camera.read()
   │
   ├─ FPS measurement (smoothed, shown top-right)
   │
   ├─ MOTION GATE (motion.py)
   │     Did anything move vs. the background model?
   │     No  → skip the heavy steps, but STILL display the frame
   │           (the video never freezes when idle).
   │     Yes → continue.
   │
   ├─ ENHANCE (enhance.py)
   │     Auto-gamma + CLAHE contrast (+ optional denoise).
   │
   ├─ DETECT (faces.py)
   │     HOG face detection on a downscaled frame.
   │     Only runs every TRACKING_SKIP_FRAMES frames; tracks bridge gaps.
   │
   ├─ RECOGNIZE (faces.py)
   │     Encode each face, compare to family encodings.
   │     distance ≤ FACE_TOLERANCE → name, else UNKNOWN.
   │
   ├─ TRACK (tracking.py)
   │     Match detections to tracks by centroid, smooth boxes (EMA),
   │     decide identity by majority vote over a window of frames.
   │
   ├─ DECIDE (main.py)
   │     Family  → green box + name + confidence, log sighting.
   │     Unknown → red box + banner + countdown + snapshot + siren.
   │     (yolo.py may suppress/shorten the alarm: animal vs human.)
   │
   └─ DISPLAY (hud.py)
         Boxes, banner, countdown, family text, mode, status, FPS.
```

---

## 5. How recognition actually works

1. `register.py` saves photos; each photo is turned into a **128-d
   encoding** (a numeric "face signature") by dlib.
2. At startup, `main.py` loads every family photo and keeps their
   encodings + names in memory (`load_family_database`).
3. For each live face, the system computes its encoding and measures the
   **distance** to every family encoding.
4. Smallest distance ≤ `FACE_TOLERANCE` (0.45) → that person is
   recognized; confidence is derived from how small the distance is.
5. A single frame can lie, so `tracking.py` keeps a **majority vote**
   over the last `ENSEMBLE_FRAMES` frames before trusting an identity.

---

## 6. Configuration — one file, three tiers

Every tunable lives in `config.py`. The single most important switch is
`PERFORMANCE_MODE`, which picks a hardware tier:

| Mode       | Target                    | Resolution | Detection       | Denoise | CNN | YOLO |
| ---------- | ------------------------- | ---------- | --------------- | ------- | --- | ---- |
| `low`      | ~4 GB RAM, 2-core CPU, Pi | 640×480    | every 4th frame | off     | off | off  |
| `balanced` | 4–8 core laptop/desktop   | 1280×720   | every 2nd frame | on      | off | on   |
| `high`     | strong multi-core / GPU   | 1920×1080  | every frame     | on      | on  | on   |

The profiles only change values of existing knobs — no feature is ever
removed. See the README "Performance Profiles" section for details.

---

## 7. Data on disk

```
family/<Name>/*.jpg     → reference photos per person (the face database)
snapshots/*.jpg         → saved images of unknown people
logs/events.db          → SQLite table of every event
logs/security.log       → human-readable audit log
sounds/siren.wav        → alarm sound
```

`storage.py` enforces `RETENTION_DAYS`: snapshots, DB rows, and log lines
older than the cutoff are deleted at startup.

---

## 8. Robustness & security posture

The system is written to **degrade, not crash**:

- **No audio device** → siren stays silent but state/logging still work
  (`siren.py` wraps `pygame.mixer.init()`).
- **No YOLO weights / ultralytics** → animal suppression is skipped with
  a warning; face recognition continues (`yolo.py`).
- **No CNN model** → HOG-only detection (`faces.py`).
- **Camera drops** → exponential-backoff reconnect (`main.py`).
- **Ctrl+C / error** → `try/finally` releases camera, stops siren,
  closes windows (`main.py`).
- **SQLite thread-safety** → single connection guarded by a lock with
  `check_same_thread=False` (`storage.py`).
- **SQL injection** → all queries use parameterized `?` placeholders.
- **Path safety** → registration sanitizes the name before creating the
  folder (`register.py`).

---

## 9. Where to look when…

| Symptom                 | Start here                                     |
| ----------------------- | ---------------------------------------------- |
| Video freezes / lags    | `main.py` motion gate + `CAP_PROP_BUFFERSIZE`  |
| Everyone shows UNKNOWN  | `family/<Name>/` empty → run `register.py`     |
| Wrong person recognized | `FACE_TOLERANCE` in `config.py`                |
| Too many false alarms   | `MOTION_THRESHOLD`, `MOTION_MIN_AREA`, YOLO    |
| No siren sound          | audio device + `sounds/siren.wav` (`siren.py`) |
| Slow on a small device  | `PERFORMANCE_MODE = "low"` in `config.py`      |
| Events not saved        | `logs/` permissions (`storage.py`)             |

---

## 10. Testing

Unit tests live in `tests/` and cover the pure-logic modules (faces,
motion, tracking). Run them with:

```bash
.venv/bin/python -m unittest discover -s tests
```

They need no camera and no audio, so they run anywhere.
