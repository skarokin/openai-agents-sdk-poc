"""Hard-deadline enforcement via shared cancel_signal."""

import logging
import threading
import time


logger = logging.getLogger(__name__)


def start_hard_deadline_watchdog(cancel_signal: threading.Event, hard_epoch: float) -> threading.Timer:
    """daemon timer that sets cancel_signal when the hard deadline hits"""

    delay = max(0.0, hard_epoch - time.time())

    def _fire() -> None:
        logger.warning("hard deadline exceeded; setting cancel_signal")
        cancel_signal.set()

    timer = threading.Timer(delay, _fire)
    timer.daemon = True
    timer.start()
    return timer
