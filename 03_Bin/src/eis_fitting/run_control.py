import multiprocessing as mp
import time


class RunControl:
    def __init__(self):
        ctx = mp.get_context('spawn')
        self._cancel = ctx.Event()
        self._running = ctx.Event()
        self._running.set()
        self._lock = ctx.RLock()
        self._paused_at = ctx.Value('d', 0., lock=False)
        self._paused_total = ctx.Value('d', 0., lock=False)

    def is_set(self):
        return self._cancel.is_set()

    def set(self):
        self._cancel.set()
        self.resume()

    def clear(self):

        with self._lock:
            self._cancel.clear()
            self._paused_at.value = 0.
            self._paused_total.value = 0.
            self._running.set()

    def pause(self):
        with self._lock:
            if not self._cancel.is_set() and not self._paused_at.value:
                self._paused_at.value = time.monotonic()
                self._running.clear()

    def resume(self):
        with self._lock:
            if self._paused_at.value:
                self._paused_total.value += time.monotonic() - self._paused_at.value
                self._paused_at.value = 0.
            self._running.set()

    def is_paused(self):
        return not self._running.is_set()

    def checkpoint(self):
        while not self._cancel.is_set() and not self._running.wait(.1):
            pass

    def active_time(self):
        with self._lock:
            now = time.monotonic()
            stopped = now - self._paused_at.value if self._paused_at.value else 0.
            return now - self._paused_total.value - stopped


def pause_checkpoint(control):
    if isinstance(control, RunControl):
        control.checkpoint()


def active_time(control):
    return control.active_time() if isinstance(control, RunControl) else time.monotonic()
