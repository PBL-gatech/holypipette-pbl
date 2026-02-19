from __future__ import absolute_import

import numpy as np
from patcherbot.controller import TaskController


class PlateScanner(TaskController):
    """Collect two plate corner positions from image-space right clicks."""

    CORNER_LABELS = ("Top Left", "Bottom Right")

    def __init__(self, patch_interface):
        super().__init__()
        self.patch_interface = patch_interface
        self.reset()

    def reset(self):
        self._is_collecting = False
        self._corner_positions = []

    @property
    def is_collecting(self):
        return self._is_collecting

    @property
    def corner_positions(self):
        return list(self._corner_positions)

    @property
    def corner_count(self):
        return len(self._corner_positions)

    def _iter_speed_controllers(self, stage):
        """Yield candidate objects that may expose max-speed getters/setters."""
        seen = set()
        for obj in (stage, getattr(stage, "unit", None), getattr(stage, "dev", None), getattr(getattr(stage, "unit", None), "dev", None)):
            if obj is None:
                continue
            ident = id(obj)
            if ident in seen:
                continue
            seen.add(ident)
            if hasattr(obj, "set_max_speed") or hasattr(obj, "get_max_speed"):
                yield obj

    def _get_max_speed(self, stage):
        """Return first reachable max speed from available speed controllers."""
        for ctrl in self._iter_speed_controllers(stage):
            getter = getattr(ctrl, "get_max_speed", None)
            if not callable(getter):
                continue
            try:
                speed = getter()
            except Exception:
                continue
            if speed is not None:
                return speed, ctrl
        return None, None

    def _set_max_speed(self, stage, speed):
        """Try to apply max speed on the first controller that supports it."""
        for ctrl in self._iter_speed_controllers(stage):
            setter = getattr(ctrl, "set_max_speed", None)
            if not callable(setter):
                continue
            try:
                setter(speed)
                return True
            except Exception:
                continue
        return False

    def _store_position(self, click_position):
        click_position = np.array(click_position, dtype=float)

        camera = self.patch_interface.pipette_controller.calibrated_unit.camera
        click_position[0] += camera.width / 2.0
        click_position[1] += camera.height / 2.0

        stage_pos_pixels = np.array(self.patch_interface.pipette_controller.calibrated_stage.reference_position())
        stage_pos_pixels[0:2] -= click_position

        current_stage_um = self.patch_interface.pipette_controller.calibrated_stage.position()
        z_um = float(self.patch_interface.pipette_controller.calibrated_unit.microscope.position() / 5.0)
        stage_pos_um = np.array([current_stage_um[0], current_stage_um[1], z_um])

        description = self.CORNER_LABELS[self.corner_count]
        self._corner_positions.append(
            {
                "description": description,
                "stage_pixels": stage_pos_pixels,
                "stage_um": stage_pos_um,
            }
        )
        print(
            f"{description}: stage pixels {stage_pos_pixels}, "
            f"stage um {stage_pos_um}"
        )
        self.patch_interface.info(
            f"{description} selected: stage pixels={stage_pos_pixels}, stage um={stage_pos_um}"
        )

    def SelectCorners(self, click_position=None):
        """Collect the Top Left and Bottom Right clicks from the GUI."""
        if click_position is None:
            self.reset()
            self._is_collecting = True
            print("Store Corners armed. Shift-right click Top Left, then Bottom Right.")
            return

        if not self._is_collecting:
            print("Store corners is not active. Click 'Store corners' before Shift-right-clicking points.")
            return

        self._store_position(click_position)
        if self.corner_count == len(self.CORNER_LABELS):
            self._is_collecting = False
            top_left, bottom_right = self._corner_positions
            print("Corner capture complete.")
            print(f"Top Left stage pixels: {top_left['stage_pixels']}")
            print(f"Top Left stage um: {top_left['stage_um']}")
            print(f"Bottom Right stage pixels: {bottom_right['stage_pixels']}")
            print(f"Bottom Right stage um: {bottom_right['stage_um']}")

    def ScanArea(self):
        """Scan the selected rectangular area from top-left to bottom-right at constant velocity."""
        self.abort_if_requested()
        if self.corner_count < len(self.CORNER_LABELS):
            self.patch_interface.warning(
                "ScanArea requires both Top Left and Bottom Right corners from Select Corners."
            )
            return

        top_left = np.array(self._corner_positions[0]["stage_um"], dtype=float)
        bottom_right = np.array(self._corner_positions[1]["stage_um"], dtype=float)

        stage = self.patch_interface.pipette_controller.calibrated_stage
        camera = self.patch_interface.pipette_controller.calibrated_unit.camera
        row_step_px = float(camera.height) - 200.0
        if row_step_px <= 0:
            self.patch_interface.warning("Camera height is too small for scan step calculation.")
            return

        top_left_xy = np.array([float(top_left[0]), float(top_left[1])], dtype=float)
        bottom_right_xy = np.array([float(bottom_right[0]), float(bottom_right[1])], dtype=float)

        scan_speed_um_per_sec = 70
        row_step_um = stage.pixels_to_um_relative(np.array([0.0, row_step_px, 0.0]))
        row_step = np.array([float(row_step_um[0]), float(row_step_um[1])], dtype=float)
        if np.isclose(row_step[1], 0.0):
            self.patch_interface.warning("Unable to determine a valid Y step for the stage.")
            return

        target_delta_y = bottom_right_xy[1] - top_left_xy[1]
        if np.isclose(target_delta_y, 0.0):
            self.patch_interface.warning("Top Left and Bottom Right have the same Y coordinate; no vertical scan range.")
            return

        target_y = bottom_right_xy[1]
        target_direction = 1.0 if target_delta_y > 0 else -1.0
        if row_step[1] * target_direction < 0:
            row_step = -row_step

        if row_step[1] * target_direction <= 0:
            self.patch_interface.warning("Could not compute a valid scan step that moves toward Bottom Right.")
            return

        reached_target_y = lambda y: (y - target_y) * target_direction >= 0
        left_x = top_left_xy[0]
        right_x = bottom_right_xy[0]

        prior_speed = None
        prior_speed_controller = None
        try:
            prior_speed, prior_speed_controller = self._get_max_speed(stage)
            if prior_speed is None:
                # Last-resort fallback: use the configured default max speed on the
                # underlying controller, if available.
                for ctrl in self._iter_speed_controllers(stage):
                    default_speed = getattr(ctrl, "DEFAULT_MAX_SPEED", None)
                    if default_speed is not None:
                        prior_speed = default_speed
                        prior_speed_controller = ctrl
                        break

            if prior_speed is not None and not self._set_max_speed(stage, scan_speed_um_per_sec):
                self.patch_interface.warning("Could not set stage max speed for scan; speed override may not apply.")

            stage.absolute_move(top_left_xy, speed=scan_speed_um_per_sec)
            self.abort_if_requested()
            stage.wait_until_still()

            move_toward_right = True
            while True:
                self.abort_if_requested()
                target_x = right_x if move_toward_right else left_x
                stage.absolute_move([target_x, stage.position()[1]], speed=scan_speed_um_per_sec)
                stage.wait_until_still()

                self.abort_if_requested()
                current_y = stage.position()[1]
                if reached_target_y(current_y):
                    break

                next_y = current_y + row_step[1]
                if reached_target_y(next_y):
                    stage.absolute_move([stage.position()[0], target_y], speed=scan_speed_um_per_sec)
                    stage.wait_until_still()
                    move_toward_right = not move_toward_right
                    final_target_x = right_x if move_toward_right else left_x
                    stage.absolute_move([final_target_x, target_y], speed=scan_speed_um_per_sec)
                    stage.wait_until_still()
                    break

                self.abort_if_requested()
                stage.absolute_move([stage.position()[0], next_y], speed=scan_speed_um_per_sec)
                stage.wait_until_still()

                move_toward_right = not move_toward_right
        finally:
            if prior_speed is not None:
                try:
                    if prior_speed_controller is not None:
                        prior_speed_set = getattr(prior_speed_controller, "set_max_speed", None)
                        if callable(prior_speed_set):
                            prior_speed_set(prior_speed)
                        else:
                            raise RuntimeError("No speed setter available on saved speed controller.")
                    elif not self._set_max_speed(stage, prior_speed):
                        raise RuntimeError("No speed setter available for this stage.")
                except Exception:
                    self.patch_interface.warning(
                        "Failed to restore stage max speed after scan."
                    )
