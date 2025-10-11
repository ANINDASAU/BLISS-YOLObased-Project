import threading
import time
import platform

try:
    import winsound
except Exception:
    winsound = None


class Alarm:
    def __init__(self, freq: int = 1000, beep_ms: int = 400, pause_s: float = 0.25):
        self.freq = freq
        self.beep_ms = beep_ms
        self.pause_s = pause_s
        self._running = False
        self._thread = None

    def _beep_loop(self):
        while self._running:
            try:
                if winsound is not None and platform.system() == "Windows":
                    winsound.Beep(self.freq, self.beep_ms)
                else:
                    print("ALARM")
                    time.sleep(self.beep_ms / 1000.0)
            except Exception:
                time.sleep(self.pause_s)

            time.sleep(self.pause_s)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._beep_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None


_default_alarm = Alarm()

def start_alarm():
    _default_alarm.start()


def stop_alarm():
    _default_alarm.stop()
