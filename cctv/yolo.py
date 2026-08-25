"""YOLOv8 object detection used to suppress false alarms caused by animals."""

from cctv.storage import logger


# COCO class ids
HUMAN_CLASS_ID = 0                    # person
ANIMAL_CLASS_IDS = frozenset(         # bird, cat, dog, horse, sheep,
    range(14, 24)                     # cow, elephant, bear, zebra, giraffe
)


class ObjectDetector:
    """Thin wrapper around YOLOv8 for animal/human classification."""

    def __init__(self, enabled: bool = True, weights: str = "yolov8n.pt"):
        self.enabled = enabled
        self.model = None

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

        return animal_seen, human_seen
