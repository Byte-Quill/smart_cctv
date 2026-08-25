"""YOLOv8 object detection used to suppress false alarms caused by animals."""

from collections import deque

from cctv.storage import logger


# COCO class ids
HUMAN_CLASS_ID = 0                    # person
ANIMAL_CLASS_IDS = frozenset(         # bird, cat, dog, horse, sheep,
    range(14, 24)                     # cow, elephant, bear, zebra, giraffe
)


class ObjectDetector:
    """YOLOv8 animal/human classification with a majority-vote window.

    A single frame can misclassify (a dog walking past, a TV face).
    Decisions are smoothed over the last ``window`` detections so the
    siren is only suppressed when animals consistently explain the scene.
    """

    def __init__(
        self,
        enabled: bool = True,
        weights: str = "yolov8n.pt",
        window: int = 5,
    ):
        self.enabled = enabled
        self.model = None
        self.window = window
        self._votes: deque = deque(maxlen=window)

        if not enabled:
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(weights)
        except Exception as error:
            logger.warning(
                "YOLO unavailable, animal suppression disabled: %s",
                error
            )
            self.enabled = False

    def detect(self, frame) -> tuple[bool, bool]:
        """Return (animal_seen, human_seen) for one frame."""

        if not self.enabled or self.model is None:
            return False, False

        animal_seen = False
        human_seen = False

        for result in self.model(frame, verbose=False):
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id == HUMAN_CLASS_ID:
                    human_seen = True
                elif class_id in ANIMAL_CLASS_IDS:
                    animal_seen = True

        self._votes.append((animal_seen, human_seen))
        return self.majority()

    def majority(self) -> tuple[bool, bool]:
        """Majority vote over the recent detection window."""
        if not self._votes:
            return False, False
        animals = sum(1 for a, _ in self._votes if a)
        humans = sum(1 for _, h in self._votes if h)
        half = len(self._votes) / 2
        return animals > half, humans > half

    def reset(self):
        self._votes.clear()
