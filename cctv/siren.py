"""Siren audio control with automatic shutdown.

The siren never runs forever: every activation carries a duration, and a
background timer stops the alarm automatically when it expires. A family
member can always silence it early (press "s" in the window, or call
``Siren.stop()``).

Durations come from the caller (main.py asks cctv/timeutil for the right
value: 2 minutes in daytime, 5 minutes in night security mode).
"""

import threading
import time

import pygame

from config import SIREN_FILE

from cctv.storage import log_event, logger


class Siren:
    """Looping alarm sound with on/off state, auto-stop, and event logging.

    All public methods are guarded by a lock so the alarm can safely be
    triggered or silenced from any thread (e.g. a future worker thread
    without stalling the main camera loop).

    Auto-shutdown
    -------------
    ``start(duration=N)`` arms a ``threading.Timer`` that calls ``stop()``
    after N seconds. Starting again while active refreshes the timer, and
    any manual stop cancels it, so there is exactly one timer at a time.
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
        self._auto_stop_timer: threading.Timer | None = None
        # Timestamp of the most recent stop (auto or manual). main.py uses
        # this to anchor the re-trigger cooldown so the alarm cannot restart
        # the instant it finishes. 0.0 = never stopped.
        self._last_stop = 0.0

        if self.mixer_ok:
            try:
                self.sound = pygame.mixer.Sound(sound_file)
            except Exception as error:
                print("\nWARNING: Could not load siren:")
                print(error)
                logger.warning(
                    "Siren could not be loaded: %s", error
                )

    def self_test(self, duration: float = 3.0):
        """Play a short siren burst to verify audio is audible.

        Used at startup when presenting the system. Deliberately NOT a
        real activation: it does not set ``active``, does not touch
        ``_last_stop`` (so the re-trigger cooldown is unaffected), and is
        logged as ``SIREN_TEST``. A daemon timer guarantees the sound
        stops after *duration* seconds no matter what.
        """
        with self._lock:
            if self.sound is None:
                print("\nSiren self-test skipped: no audio device.")
                logger.info("Siren self-test skipped (no audio)")
                return

            print(f"\nSIREN SELF-TEST ({duration:.0f}s)...")
            logger.info("SIREN SELF-TEST (%.0fs)", duration)

            self.sound.play(-1)
            log_event("SIREN_TEST")

            def _end_test():
                with self._lock:
                    self._stop_sound_locked()
                logger.info("SIREN SELF-TEST COMPLETE")
                print("Siren self-test complete.")

            timer = threading.Timer(duration, _end_test)
            timer.daemon = True  # never block process exit
            timer.start()

    # Turn the siren on; it auto-stops after *duration* seconds.
    def start(self, duration: float | None = None):
        with self._lock:
            self._start_locked(duration)

    def _start_locked(self, duration: float | None):
        if self.active:
            # Already ringing — refresh the auto-stop timer only.
            self._arm_timer_locked(duration)
            return

        print("\nSIREN ACTIVATED")

        logger.warning("SIREN ACTIVATED")

        if self.sound is not None:
            self.sound.play(-1)

        self.active = True

        log_event("SIREN_ON")

        self._arm_timer_locked(duration)

    def _arm_timer_locked(self, duration: float | None):
        """(Re)arm the auto-stop timer. Caller must hold ``_lock``."""
        # Cancel any previous timer so only one is ever pending.
        if self._auto_stop_timer is not None:
            self._auto_stop_timer.cancel()
            self._auto_stop_timer = None

        if duration is None:
            return

        timer = threading.Timer(duration, self._auto_stop)
        timer.daemon = True  # never block process exit
        timer.start()
        self._auto_stop_timer = timer

        logger.info("Siren will auto-stop in %.0fs", duration)

    def _auto_stop(self):
        """Timer callback: stop the siren and record why."""
        with self._lock:
            if not self.active:
                return
            print("\nSiren auto-stopped (duration expired).")
            logger.info("SIREN AUTO-STOPPED (duration expired)")
            self._stop_sound_locked()
            self.active = False
            self._auto_stop_timer = None
            self._last_stop = time.time()
            log_event("SIREN_AUTO_OFF")

    @property
    def last_stop(self) -> float:
        """Timestamp of the most recent stop; 0.0 if never stopped."""
        with self._lock:
            return self._last_stop

    @property
    def is_active(self) -> bool:
        """Thread-safe read of the alarm state."""
        with self._lock:
            return self.active

    # Turn the siren off (family member silencing it early)
    def stop(self):
        with self._lock:
            self._stop_locked()

    def _stop_locked(self):
        if not self.active:
            return

        print("\nSiren stopped.")

        self._stop_sound_locked()

        self.active = False

        # Disarm the pending auto-stop timer, if any.
        if self._auto_stop_timer is not None:
            self._auto_stop_timer.cancel()
            self._auto_stop_timer = None

        self._last_stop = time.time()

        logger.info("SIREN STOPPED")

        log_event("SIREN_OFF")

    def _stop_sound_locked(self):
        """Silence the audio. Caller must hold ``_lock``."""
        if self.sound is not None:
            self.sound.stop()
