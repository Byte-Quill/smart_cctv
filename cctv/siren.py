"""Siren audio control."""

import pygame

from config import SIREN_FILE

from cctv.storage import log_event, logger


class Siren:
    """Looping alarm sound with on/off state and event logging."""

    def __init__(self, sound_file: str = SIREN_FILE):
        pygame.mixer.init()
        self.sound = None
        self.active = False

        try:
            self.sound = pygame.mixer.Sound(sound_file)
        except Exception as error:
            print("\nWARNING: Could not load siren:")
            print(error)
            logger.warning(
                "Siren could not be loaded: %s", error
            )

    # Turn the siren on (loops forever until stopped)
    def start(self):

        if self.active:
            return

        print("\n🚨 SIREN ACTIVATED 🚨")

        logger.warning("SIREN ACTIVATED")

        if self.sound is not None:
            self.sound.play(-1)

        self.active = True

        log_event("SIREN_ON")

    # Turn the siren off
    def stop(self):

        if not self.active:
            return

        print("\nSiren stopped.")

        if self.sound is not None:
            self.sound.stop()

        self.active = False

        logger.info("SIREN STOPPED")

        log_event("SIREN_OFF")
