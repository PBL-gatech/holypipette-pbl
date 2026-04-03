import threading


class RecordingStateManager:
    """Thread-safe manager for controlling recording state and tracking sample numbers."""
    def __init__(self):
        """Initialize the RecordingStateManager."""
        self._recording_enabled = False
        self._lock = threading.Lock()
        self.sample_number = 0
        self.testMode = False
    
    def increment_sample_number(self):
        """Increment the sample number in a thread-safe manner."""
        with self._lock:
            self.sample_number += 1
            print("Sample number incremented to:", self.sample_number)

    def toggle_recording(self):
        """
        Toggle the recording state between enabled and disabled.

        Returns:
            bool: The new recording state.
        """
        with self._lock:
            self._recording_enabled = not self._recording_enabled
            # print("Recording state toggled to:", self._recording_enabled)
            return self._recording_enabled

    def set_recording(self, state: bool) -> None:
        """
        Set the recording state explicitly.

        Args:
            state (bool): True to enable recording, False to disable.
        """
        with self._lock:
            self._recording_enabled = state
            # print("Recording state set to:", self._recording_enabled)

    def is_recording_enabled(self) -> bool:
        """
        Check if recording is currently enabled.

        Returns:
            bool: True if recording is enabled, False otherwise.
        """
        with self._lock:
            return self._recording_enabled
