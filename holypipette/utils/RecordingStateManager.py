import threading

class RecordingStateManager:
    def __init__(self):
        self._recording_enabled = False
        self._lock = threading.Lock()
    
    def toggle_recording(self):
        with self._lock:
            self._recording_enabled = not self._recording_enabled
            print("Recording state toggled to:", self._recording_enabled)
            return self._recording_enabled

    def set_recording(self, state: bool) -> None:
        with self._lock:
            self._recording_enabled = state
            print("Recording state set to:", self._recording_enabled)

    def is_recording_enabled(self) -> bool:
        with self._lock:
            return self._recording_enabled
