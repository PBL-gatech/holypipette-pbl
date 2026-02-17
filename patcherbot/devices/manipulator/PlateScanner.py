from __future__ import absolute_import

import numpy as np


class PlateScanner:
    """Collect two plate corner positions from image-space right clicks."""

    CORNER_LABELS = ("Top Left", "Bottom Right")

    def __init__(self, patch_interface):
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
            print("Store Corners armed. Right-click Top Left, then Bottom Right.")
            return

        if not self._is_collecting:
            print("Store corners is not active. Click 'Store corners' before right-clicking points.")
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
