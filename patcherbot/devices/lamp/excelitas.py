from __future__ import annotations

import time

import nidaqmx

from .lamp import Lamp

__all__ = ["ExcelitasLamp"]


class ExcelitasLamp(Lamp):
    """
    Excelitas lamp filter-wheel driver via NI-DAQ digital lines.
    """
    def __init__(
        self,
        stepcmdChannel: str = "Dev1/port0/line0",
        dircmdChannel: str = "Dev1/port0/line1",
        *,
        steps_per_position: int = 800,
        time_per_step_ms: float = 1.0,
        cube_slots: int | None = 6,
        settle_ms: int = 100,
        initial_slot: int | None = 1,
        dir_high_for_increase: bool = True,
    ):
        """
        Initializes the lamp controller and configuration parameters.

        Args:
            stepcmdChannel (str): NI-DAQ channel for step control.
            dircmdChannel (str): NI-DAQ channel for direction control.
            steps_per_position (int): Number of steps between filter positions.
            time_per_step_ms (float): Delay per step in milliseconds.
            cube_slots (int | None): Number of filter slots (wraps if set).
            settle_ms (int): Delay after initializing tasks (ms).
            initial_slot (int | None): Initial filter slot.
            dir_high_for_increase (bool): Direction signal for increasing slot index.
        """
        self.stepcmdChannel = stepcmdChannel
        self.dircmdChannel = dircmdChannel
        self.step_task = None
        self.dir_task = None
        self._steps_per_position = int(steps_per_position)
        self._time_per_step_ms = float(time_per_step_ms)
        self._cube_slots = int(cube_slots) if cube_slots is not None else None
        self._settle_ms = int(settle_ms)
        self._dir_high_for_increase = bool(dir_high_for_increase)
        self._current_filter = None
        if initial_slot is not None:
            self._current_filter = self._normalize_slot(initial_slot)
        self.shutter_state = "closed"
        self.current_light = None
        self.current_excitation_filter = None

        super().__init__()

    def _initialize(self):
        """Initializes hardware tasks for the lamp controller."""
        self._setup_tasks()

    def _setup_tasks(self) -> None:
        """
        Creates and starts NI-DAQ tasks for step and direction control.

        Raises:
            Exception: If task setup fails.
        """
        import nidaqmx.constants as c

        self._close_tasks()
        self.step_task = nidaqmx.Task()
        self.dir_task = nidaqmx.Task()

        try:
            self.step_task.do_channels.add_do_chan(
                self.stepcmdChannel,
                line_grouping=c.LineGrouping.CHAN_PER_LINE,
            )
            self.step_task.start()
            time.sleep(self._settle_ms / 1000.0)

            self.dir_task.do_channels.add_do_chan(
                self.dircmdChannel,
                line_grouping=c.LineGrouping.CHAN_PER_LINE,
            )
            self.dir_task.start()
            time.sleep(self._settle_ms / 1000.0)
        except Exception:
            self._close_tasks()
            raise

    def _close_tasks(self) -> None:
        """Stops and closes all active NI-DAQ tasks."""
        for task in (self.step_task, self.dir_task):
            if task is not None:
                try:
                    task.stop()
                except Exception:
                    pass
                try:
                    task.close()
                except Exception:
                    pass
        self.step_task = None
        self.dir_task = None

    def _set_direction(self, direction: int | bool) -> None:
        """
        Sets the rotation direction of the filter wheel.

        Args:
            direction (int | bool): Direction signal value.

        Raises:
            RuntimeError: If the direction task is not initialized.
        """
        if self.dir_task is None:
            raise RuntimeError("DIR task is not initialized. Call initialize() first.")

        self.dir_task.write(bool(direction), auto_start=False)

    def _take_n_steps(self, n_steps: int, time_per_step_ms: float) -> None:
        """
        Mirrors TakeNSteps.vi:

        For i in range(n_steps):
          Frame 0: DAQmx Write TRUE  (Digital Bool 1Line 1Point), Wait(time_per_step_ms)
          Frame 1: DAQmx Write FALSE (Digital Bool 1Line 1Point), Wait(time_per_step_ms)

        Note: This matches the VI wiring (time_per_step goes to BOTH waits).
        
        Args:
            n_steps (int): Number of steps to execute.
            time_per_step_ms (float): Delay between step transitions (ms).

        Raises:
            RuntimeError: If the step task is not initialized.
        """
        if self.step_task is None:
            raise RuntimeError("STEP task is not initialized. Call initialize() first.")

        if n_steps <= 0:
            return

        t = max(0.0, time_per_step_ms) / 1000.0

        for _ in range(n_steps):
            # Frame 0
            self.step_task.write(True, auto_start=False)
            time.sleep(t)

            # Frame 1
            self.step_task.write(False, auto_start=False)
            time.sleep(t)

    def _normalize_slot(self, slot: int) -> int:
        """
        Normalizes a slot index within valid range.

        Args:
            slot (int): Raw slot value.

        Returns:
            int: Normalized slot index.
        """
        slot = int(slot)
        if self._cube_slots:
            return ((slot - 1) % self._cube_slots) + 1
        return slot

    def _coerce_slot(self, value):
        """
        Converts input into a valid slot index if possible.

        Args:
            value: Input slot value.

        Returns:
            int | None: Valid slot index or None if invalid.
        """
        if value is None:
            return None
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if isinstance(value, int):
            return self._normalize_slot(value)
        self.warning(f"ExcelitasLamp: Unsupported filter value {value!r}")
        return None

    def _slot_delta(self, current: int, target: int) -> int:
        """
        Computes the shortest step difference between two slots.

        Args:
            current (int): Current slot.
            target (int): Target slot.

        Returns:
            int: Step difference (positive or negative).
        """
        if not self._cube_slots:
            return target - current
        forward = (target - current) % self._cube_slots
        backward = forward - self._cube_slots
        if abs(backward) < abs(forward):
            return backward
        return forward

    def set_filter(self, filter: int | None = None):
        """
        Move the wheel to the requested numeric slot.

        Args:
            filter (int | None): Target filter slot.
        """
        slot = self._coerce_slot(filter)
        if slot is None:
            self.info("ExcelitasLamp: No filter specified, skipping set_filter.")
            return

        if self._current_filter is None:
            self._current_filter = slot
            return

        delta = self._slot_delta(self._current_filter, slot)
        if delta == 0:
            self._current_filter = slot
            return

        direction_high = self._dir_high_for_increase if delta > 0 else not self._dir_high_for_increase
        self._set_direction(direction_high)
        steps = abs(delta) * self._steps_per_position
        self._take_n_steps(steps, self._time_per_step_ms)
        self._current_filter = slot

    def get_filter(self):
        """
        Retrieves the current filter slot.

        Returns:
            int | None: Current filter slot.
        """
        return self._current_filter

    def open_shutter(self):
        """Opens the shutter."""
        self.shutter_state = "open"

    def close_shutter(self):
        """Closes the shutter."""
        self.shutter_state = "closed"

    def get_shutter_state(self):
        """
        Retrieves the shutter state.

        Returns:
            str: "open" or "closed".
        """
        return self.shutter_state

    def enable(self, light=None, excitation_filter=None):
        """
        Enables the lamp and optionally sets the excitation filter.

        Args:
            light: Light source object.
            excitation_filter: Desired filter slot.
        """
        self.current_light = light
        if excitation_filter is not None:
            self.current_excitation_filter = excitation_filter
            slot = self._coerce_slot(excitation_filter)
            if slot is not None:
                self.set_filter(slot)
        if light is None or getattr(light, "name", None) == "OFF":
            self.shutter_state = "closed"
        else:
            self.shutter_state = "open"


