"""Siren audio control."""

import threading

import pygame

from config import SIREN_FILE

from cctv.storage import log_event, logger


class Siren:
    """Looping alarm sound with on/off state and event logging.

    All public methods are guarded by a lock so the alarm can safely be
    triggered or silenced from any thread (e.g. a future worker thread
    without stalling the main camera loop).
    """

    def __init__(self, sound_file: str = SIREN_FILE):
        # A headless box or a machine with no sound card makes
        # pygame.mixer.init() raise. The alarm must never take the whole
        # camera loop down with it, so audio is best-effort: if it cannot
        # be initialised the siren still tracks state and logs events, it
        # just stays silent.
        self.mixer_ok = False
        try:
            pygame.mixer.init()
            self.mixer_ok = True
        except Exception as error:
            print("\nWARNING: No audio device — siren will be silent:")
            print(error)
            logger.warning(
                "Audio unavailable, siren silent: %s", error
            )

        self.sound = None
        self.active = False
        self._lock = threading.Lock()

        if self.mixer_ok:
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
        with self._lock:
            self._start_locked()

    def _start_locked(self):
        if self.active:
            return

        print("\nSIREN ACTIVATED")

        logger.warning("SIREN ACTIVATED")

        if self.sound is not None:
            self.sound.play(-1)

        self.active = True

        log_event("SIREN_ON")

    @property
    def is_active(self) -> bool:
        """Thread-safe read of the alarm state."""
        with self._lock:
            return self.active

    # Turn the siren off
    def stop(self):
        with self._lock:
            self._stop_locked()

    def _stop_locked(self):
        if not self.active:
            return

        print("\nSiren stopped.")

        if self.sound is not None:
            self.sound.stop()

        self.active = False

        logger.info("SIREN STOPPED")

        log_event("SIREN_OFF")
