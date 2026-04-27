"""PyQt5 GUI wrapper for experiments.SimpleDatasetBuilder."""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Iterable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.SimpleDatasetBuilder import (  # noqa: E402
    ActionSelector,
    AxisToggle,
    FilterSettings,
    ObservationSelector,
    SimpleDatasetBuilder,
)


class DatasetBuilderGUI(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DatasetBuilderGUI")
        self.resize(1200, 900)
        self._build_ui()
        self._connect()
        self._update_dataset_name_preview()
        self._sync_cv_generation_controls()
        self._sync_gigaseal_cutoff_controls()
        self._sync_selector_constraints()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        meta_group = QGroupBox("Metadata")
        meta_row = QHBoxLayout(meta_group)
        self.metadata_path = QLineEdit()
        self.metadata_path.setReadOnly(True)
        self.load_metadata_btn = QPushButton("Load Metadata...")
        meta_row.addWidget(QLabel("File:"))
        meta_row.addWidget(self.metadata_path, 1)
        meta_row.addWidget(self.load_metadata_btn)
        root.addWidget(meta_group)

        folders_group = QGroupBox("Rig Recorder Folders")
        folders_layout = QVBoxLayout(folders_group)
        root_row = QHBoxLayout()
        self.data_root = QLineEdit(str(REPO_ROOT / "experiments" / "Data" / "rig_recorder_data"))
        self.browse_root_btn = QPushButton("Browse Root...")
        root_row.addWidget(QLabel("Data Root:"))
        root_row.addWidget(self.data_root, 1)
        root_row.addWidget(self.browse_root_btn)
        folders_layout.addLayout(root_row)

        btn_row = QHBoxLayout()
        self.import_folder_btn = QPushButton("Import Folder...")
        self.import_parent_btn = QPushButton("Import Parent...")
        self.add_existing_dirs_btn = QPushButton("Add Date Folders...")
        self.remove_folder_btn = QPushButton("Remove Selected")
        self.clear_folders_btn = QPushButton("Clear")
        btn_row.addWidget(self.import_folder_btn)
        btn_row.addWidget(self.import_parent_btn)
        btn_row.addWidget(self.add_existing_dirs_btn)
        btn_row.addWidget(self.remove_folder_btn)
        btn_row.addWidget(self.clear_folders_btn)
        btn_row.addStretch(1)
        folders_layout.addLayout(btn_row)

        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QListWidget.ExtendedSelection)
        folders_layout.addWidget(self.folder_list)
        root.addWidget(folders_group)

        settings_row = QHBoxLayout()
        settings_row.addWidget(self._build_settings_group(), 1)
        settings_row.addWidget(self._build_selector_group(), 1)
        root.addLayout(settings_row)

        self.build_btn = QPushButton("Build Dataset")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        root.addWidget(self.build_btn)
        root.addWidget(self.log, 1)

    def _build_settings_group(self) -> QGroupBox:
        g = QGroupBox("Builder Settings")
        form = QFormLayout(g)

        self.dataset_name = QLineEdit("PatcherBot_dataset_gui.hdf5")
        self.append_test_name = QCheckBox("Use test dataset naming")
        self.append_test_name.setToolTip(
            "If checked, names like 'PatcherBot_dataset_v0_510' become 'PatcherBot_test_dataset_v0_510'."
        )
        self.dataset_name_preview = QLabel("")
        self.dataset_name_preview.setStyleSheet("color: #666;")
        self.val_ratio = QDoubleSpinBox()
        self.val_ratio.setRange(0.0, 1.0)
        self.val_ratio.setDecimals(3)
        self.val_ratio.setSingleStep(0.01)
        self.val_ratio.setValue(0.1)

        self.random_seed = QSpinBox()
        self.random_seed.setRange(0, 999999999)
        self.freq_mask = QSpinBox()
        self.freq_mask.setRange(1, 1000)
        self.freq_mask.setValue(1)

        self.image_resize = QSpinBox()
        self.image_resize.setRange(1, 4096)
        self.image_resize.setValue(85)

        self.inaction = QSpinBox()
        self.inaction.setRange(0, 100000)
        self.inaction_tolerance = QDoubleSpinBox()
        self.inaction_tolerance.setRange(0, 1000000)
        self.inaction_tolerance.setDecimals(4)
        self.inaction_tolerance.setValue(0.0)
        self.skip_invalid_observations = QCheckBox("Skip timesteps with NaN/Inf in selected data")
        self.skip_invalid_observations.setChecked(True)
        self.skip_invalid_observations.setToolTip(
            "Drop any timestep whose selected observation or action payload contains NaN or Inf values."
        )
        self.gigaseal_cutoff_enabled = QCheckBox("Gigaseal: stop at resistance cutoff")
        self.gigaseal_cutoff_enabled.setToolTip(
            "When enabled, gigaseal trajectories stop once resistance reaches the configured cutoff."
        )
        self.gigaseal_cutoff_value = QDoubleSpinBox()
        self.gigaseal_cutoff_value.setRange(0.0, 1000000.0)
        self.gigaseal_cutoff_value.setDecimals(3)
        self.gigaseal_cutoff_value.setSingleStep(10.0)
        self.gigaseal_cutoff_value.setValue(1200.0)

        self.obs_resistance_slope = QCheckBox("Resistance Slope")
        self.obs_resistance_slope.setChecked(False)
        self.obs_resistance_slope.setToolTip("Compute windowed numerical average slope of resistance per sample.")
        self.slope_window_spin = QSpinBox()
        self.slope_window_spin.setRange(5, 50)
        self.slope_window_spin.setValue(20)
        self.slope_window_spin.setEnabled(False)
        self.slope_window_spin.setToolTip("Window size for resistance slope computation (5–50).")

        self.load_next_obs = QCheckBox("Load next observations")
        self.use_velocities = QCheckBox("Use velocities")
        self.use_velocities.setToolTip(
            "Convert deltas to per-observation velocities using forward/backward Euler."
        )
        self.omit_stage_movement = QCheckBox("Omit stage movement attempts")
        self.center_crop = QCheckBox("Center crop camera")
        self.pipette_dot = QCheckBox("Add final pipette dot")
        self.use_cv_defined_coords = QCheckBox(
            "Use CV-defined coordinates (generate cv_movement_recording.csv)"
        )
        self.use_cv_defined_coords.setToolTip(
            "When enabled, the GUI runs ImageDatasetPreparer before building demos."
        )
        self.cv_filter_images = QCheckBox(
            "CV generation: filter frames to successful-attempt windows"
        )
        self.cv_filter_images.setChecked(True)
        self.cv_filter_images.setToolTip(
            "If unchecked, CV generation uses all frames in each camera_frames folder."
        )
        self.cv_focus_with_detector_crop = QCheckBox(
            "CV generation: crop focus input around detected pipette"
        )
        self.cv_focus_with_detector_crop.setChecked(True)
        self.cv_focus_with_detector_crop.setToolTip(
            "If checked, pipette focus inference runs on a detector-centered crop instead of the full frame."
        )
        self.cv_use_kalman_focus_fusion = QCheckBox(
            "CV generation: Kalman fuse encoder Z + focuser Z"
        )
        self.cv_use_kalman_focus_fusion.setChecked(False)
        self.cv_use_kalman_focus_fusion.setToolTip(
            "Encoder Z dominates motion while focuser Z gradually realigns focal-plane reference."
        )

        self.enable_filter = QCheckBox("Enable random image filter")
        self.filter_prob = QDoubleSpinBox()
        self.filter_prob.setRange(0.0, 1.0)
        self.filter_prob.setDecimals(3)
        self.filter_prob.setSingleStep(0.05)
        self.filter_prob.setValue(0.65)
        self.filter_train_only = QCheckBox("Filter train split only")
        self.filter_same_demo = QCheckBox("Same filter per demo")

        form.addRow("Dataset Name:", self.dataset_name)
        form.addRow(self.append_test_name)
        form.addRow("Effective Name:", self.dataset_name_preview)
        form.addRow("Validation Ratio:", self.val_ratio)
        form.addRow("Random Seed:", self.random_seed)
        form.addRow("Frequency Mask:", self.freq_mask)
        form.addRow("Image Resize:", self.image_resize)
        form.addRow("Inaction Steps:", self.inaction)
        form.addRow("Inaction Tolerance:", self.inaction_tolerance)
        form.addRow(self.skip_invalid_observations)
        form.addRow(self.gigaseal_cutoff_enabled)
        form.addRow("Gigaseal Resistance Cutoff:", self.gigaseal_cutoff_value)
        form.addRow(self.load_next_obs)
        form.addRow(self.use_velocities)
        form.addRow(self.omit_stage_movement)
        form.addRow(self.center_crop)
        form.addRow(self.pipette_dot)
        form.addRow(self.use_cv_defined_coords)
        form.addRow(self.cv_filter_images)
        form.addRow(self.cv_focus_with_detector_crop)
        form.addRow(self.cv_use_kalman_focus_fusion)
        form.addRow(self.enable_filter)
        form.addRow("Filter Probability:", self.filter_prob)
        form.addRow(self.filter_train_only)
        form.addRow(self.filter_same_demo)
        return g

    def _build_selector_group(self) -> QGroupBox:
        g = QGroupBox("Selectors")
        root = QVBoxLayout(g)

        obs_box = QGroupBox("Observations")
        obs_grid = QGridLayout(obs_box)

        self.obs_pressure = QCheckBox("Pressure")
        self.obs_pressure.setChecked(True)
        self.obs_resistance = QCheckBox("Resistance")
        self.obs_resistance.setChecked(True)
        self.obs_current = QCheckBox("Current")
        self.obs_voltage = QCheckBox("Voltage")
        self.obs_stage = QCheckBox("Stage Position")
        self.obs_stage.setChecked(True)
        self.obs_pipette = QCheckBox("Pipette Position")
        self.obs_pipette.setChecked(True)
        self.obs_camera = QCheckBox("Camera Image")
        self.obs_camera.setChecked(True)

        self.obs_stage_x = QCheckBox("Stage X")
        self.obs_stage_x.setChecked(True)
        self.obs_stage_y = QCheckBox("Stage Y")
        self.obs_stage_y.setChecked(True)
        self.obs_stage_z = QCheckBox("Stage Z")
        self.obs_stage_z.setChecked(True)
        self.obs_pip_x = QCheckBox("Pipette X")
        self.obs_pip_x.setChecked(True)
        self.obs_pip_y = QCheckBox("Pipette Y")
        self.obs_pip_y.setChecked(True)
        self.obs_pip_z = QCheckBox("Pipette Z")
        self.obs_pip_z.setChecked(True)

        obs_grid.addWidget(self.obs_pressure, 0, 0)
        obs_grid.addWidget(self.obs_resistance, 0, 1)
        obs_grid.addWidget(self.obs_resistance_slope, 0, 2)
        obs_grid.addWidget(self.obs_current, 0, 3)
        obs_grid.addWidget(self.obs_voltage, 1, 0)
        obs_grid.addWidget(self.obs_stage, 1, 1)
        obs_grid.addWidget(self.obs_pipette, 1, 2)
        obs_grid.addWidget(self.obs_camera, 1, 3)
        obs_grid.addWidget(QLabel("Slope Window:"), 2, 1)
        obs_grid.addWidget(self.slope_window_spin, 2, 2)

        obs_grid.addWidget(QLabel("Stage Axes:"), 3, 0)
        obs_grid.addWidget(self.obs_stage_x, 3, 1)
        obs_grid.addWidget(self.obs_stage_y, 3, 2)
        obs_grid.addWidget(self.obs_stage_z, 3, 3)

        obs_grid.addWidget(QLabel("Pipette Axes:"), 4, 0)
        obs_grid.addWidget(self.obs_pip_x, 4, 1)
        obs_grid.addWidget(self.obs_pip_y, 4, 2)
        obs_grid.addWidget(self.obs_pip_z, 4, 3)

        act_box = QGroupBox("Actions")
        act_grid = QGridLayout(act_box)

        self.act_stage = QCheckBox("Stage Action")
        self.act_pipette = QCheckBox("Pipette Action")
        self.act_pipette.setChecked(True)
        self.act_pressure = QCheckBox("Pressure Action")
        self.act_pressure_raw = QCheckBox("Use Raw Pressure Action Values")
        self.act_pressure_raw.setChecked(True)
        self.act_pressure_raw.setToolTip(
            "If checked, store commanded pressure as carried raw mbar setpoints. "
            "If unchecked, store only per-command pressure deltas from the previous setpoint. "
            "ATM state is always one-hot."
        )

        self.act_stage_x = QCheckBox("Stage X")
        self.act_stage_x.setChecked(True)
        self.act_stage_y = QCheckBox("Stage Y")
        self.act_stage_y.setChecked(True)
        self.act_stage_z = QCheckBox("Stage Z")
        self.act_stage_z.setChecked(True)
        self.act_pip_x = QCheckBox("Pipette X")
        self.act_pip_x.setChecked(True)
        self.act_pip_y = QCheckBox("Pipette Y")
        self.act_pip_y.setChecked(True)
        self.act_pip_z = QCheckBox("Pipette Z")
        self.act_pip_z.setChecked(True)

        act_grid.addWidget(self.act_stage, 0, 0)
        act_grid.addWidget(self.act_pipette, 0, 1)
        act_grid.addWidget(self.act_pressure, 0, 2)
        act_grid.addWidget(self.act_pressure_raw, 0, 3)

        act_grid.addWidget(QLabel("Stage Axes:"), 1, 0)
        act_grid.addWidget(self.act_stage_x, 1, 1)
        act_grid.addWidget(self.act_stage_y, 1, 2)
        act_grid.addWidget(self.act_stage_z, 1, 3)

        act_grid.addWidget(QLabel("Pipette Axes:"), 2, 0)
        act_grid.addWidget(self.act_pip_x, 2, 1)
        act_grid.addWidget(self.act_pip_y, 2, 2)
        act_grid.addWidget(self.act_pip_z, 2, 3)

        root.addWidget(obs_box)
        root.addWidget(act_box)
        return g

    def _connect(self) -> None:
        self.load_metadata_btn.clicked.connect(self.load_metadata)
        self.browse_root_btn.clicked.connect(self.browse_root)
        self.import_folder_btn.clicked.connect(self.import_folder)
        self.import_parent_btn.clicked.connect(self.import_parent)
        self.add_existing_dirs_btn.clicked.connect(self.add_existing_date_folders)
        self.remove_folder_btn.clicked.connect(self.remove_selected)
        self.clear_folders_btn.clicked.connect(self.folder_list.clear)
        self.build_btn.clicked.connect(self.build_dataset)

        self.dataset_name.textChanged.connect(self._update_dataset_name_preview)
        self.append_test_name.toggled.connect(self._update_dataset_name_preview)
        self.use_cv_defined_coords.toggled.connect(self._sync_cv_generation_controls)
        self.gigaseal_cutoff_enabled.toggled.connect(self._sync_gigaseal_cutoff_controls)

        self.obs_resistance.toggled.connect(self._sync_selector_constraints)
        self.obs_resistance_slope.toggled.connect(self._sync_selector_constraints)
        self.act_stage.toggled.connect(self._sync_selector_constraints)
        self.act_pipette.toggled.connect(self._sync_selector_constraints)
        self.act_pressure.toggled.connect(self._sync_selector_constraints)
        self.obs_stage.toggled.connect(self._sync_selector_constraints)
        self.obs_pipette.toggled.connect(self._sync_selector_constraints)
        self.obs_stage_x.toggled.connect(self._sync_selector_constraints)
        self.obs_stage_y.toggled.connect(self._sync_selector_constraints)
        self.obs_stage_z.toggled.connect(self._sync_selector_constraints)
        self.obs_pip_x.toggled.connect(self._sync_selector_constraints)
        self.obs_pip_y.toggled.connect(self._sync_selector_constraints)
        self.obs_pip_z.toggled.connect(self._sync_selector_constraints)
        self.act_stage_x.toggled.connect(self._sync_selector_constraints)
        self.act_stage_y.toggled.connect(self._sync_selector_constraints)
        self.act_stage_z.toggled.connect(self._sync_selector_constraints)
        self.act_pip_x.toggled.connect(self._sync_selector_constraints)
        self.act_pip_y.toggled.connect(self._sync_selector_constraints)
        self.act_pip_z.toggled.connect(self._sync_selector_constraints)

    def _sync_cv_generation_controls(self) -> None:
        enabled = self.use_cv_defined_coords.isChecked()
        self.cv_filter_images.setEnabled(enabled)
        self.cv_focus_with_detector_crop.setEnabled(enabled)
        self.cv_use_kalman_focus_fusion.setEnabled(enabled)

    def _sync_gigaseal_cutoff_controls(self) -> None:
        self.gigaseal_cutoff_value.setEnabled(self.gigaseal_cutoff_enabled.isChecked())

    def _with_blocked_signals(self, *widgets):
        class _Blocker:
            def __init__(self, controls):
                self.controls = controls
                self.states = []

            def __enter__(self):
                for control in self.controls:
                    self.states.append(control.blockSignals(True))
                return self

            def __exit__(self, exc_type, exc, tb):
                for control, state in zip(self.controls, self.states):
                    control.blockSignals(state)
                return False

        return _Blocker(widgets)

    def _link_include(self, obs_cb: QCheckBox, act_cb: QCheckBox) -> None:
        if obs_cb.isChecked() and not act_cb.isChecked():
            with self._with_blocked_signals(act_cb):
                act_cb.setChecked(True)

    def _link_axis(self, obs_cb: QCheckBox, act_cb: QCheckBox) -> None:
        if obs_cb.isChecked() and not act_cb.isChecked():
            with self._with_blocked_signals(act_cb):
                act_cb.setChecked(True)

    def _sync_selector_constraints(self) -> None:
        self._link_include(self.obs_stage, self.act_stage)
        self._link_include(self.obs_pipette, self.act_pipette)

        self._link_axis(self.obs_stage_x, self.act_stage_x)
        self._link_axis(self.obs_stage_y, self.act_stage_y)
        self._link_axis(self.obs_stage_z, self.act_stage_z)
        self._link_axis(self.obs_pip_x, self.act_pip_x)
        self._link_axis(self.obs_pip_y, self.act_pip_y)
        self._link_axis(self.obs_pip_z, self.act_pip_z)

        for control in (self.obs_stage_x, self.obs_stage_y, self.obs_stage_z):
            control.setEnabled(self.obs_stage.isChecked())
        for control in (self.obs_pip_x, self.obs_pip_y, self.obs_pip_z):
            control.setEnabled(self.obs_pipette.isChecked())
        self.obs_resistance_slope.setEnabled(self.obs_resistance.isChecked())
        self.slope_window_spin.setEnabled(
            self.obs_resistance.isChecked() and self.obs_resistance_slope.isChecked()
        )
        for control in (self.act_stage_x, self.act_stage_y, self.act_stage_z):
            control.setEnabled(self.act_stage.isChecked())
        for control in (self.act_pip_x, self.act_pip_y, self.act_pip_z):
            control.setEnabled(self.act_pipette.isChecked())
        self.act_pressure_raw.setEnabled(self.act_pressure.isChecked())

    def _append(self, text: str) -> None:
        self.log.append(text)

    def _effective_dataset_name(self, raw_name: str) -> str:
        name = raw_name.strip()
        if not name:
            return ""
        if not name.lower().endswith((".h5", ".hdf5")):
            name = f"{name}.hdf5"

        if self.append_test_name.isChecked():
            stem = Path(name).stem
            suffix = Path(name).suffix
            if "_test_dataset_" in stem:
                return f"{stem}{suffix}"
            if "_dataset_" in stem:
                stem = stem.replace("_dataset_", "_test_dataset_", 1)
            elif not stem.endswith("_test"):
                stem = f"{stem}_test"
            name = f"{stem}{suffix}"
        return name

    def _update_dataset_name_preview(self) -> None:
        effective = self._effective_dataset_name(self.dataset_name.text())
        self.dataset_name_preview.setText(effective or "(empty)")

    def browse_root(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select rig_recorder_data root", self.data_root.text())
        if d:
            self.data_root.setText(d)

    def import_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select rig-recorder folder", self.data_root.text())
        if d:
            self._import_one(Path(d), ask_on_conflict=True)

    def import_parent(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select parent folder", self.data_root.text())
        if not d:
            return
        parent = Path(d)
        candidates = [p for p in sorted(parent.iterdir()) if p.is_dir() and self._looks_like_demo(p)]
        if not candidates:
            QMessageBox.information(self, "No folders", "No child folders with movement CSV files found.")
            return
        count = 0
        for p in candidates:
            if self._import_one(p, ask_on_conflict=False):
                count += 1
        self._append(f"Imported {count} folder(s) from {parent}")

    def add_existing_date_folders(self) -> None:
        root = Path(self.data_root.text().strip())
        if not root.is_dir():
            QMessageBox.warning(self, "Data root", f"Not a directory:\n{root}")
            return

        candidates = [p for p in sorted(root.iterdir()) if p.is_dir() and self._looks_like_demo(p)]
        if not candidates:
            QMessageBox.information(
                self,
                "No date folders",
                "No date folders containing movement CSV files were found under the current data root.",
            )
            return

        selected = self._pick_multiple_directories(candidates)
        if not selected:
            return

        count = 0
        for directory in selected:
            if self._add_folder(directory.name, directory):
                count += 1
        self._append(f"Added {count} existing folder(s) from data root.")

    def _pick_multiple_directories(self, directories: Iterable[Path]) -> list[Path]:
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Date Folders")
        dialog.resize(640, 520)
        layout = QVBoxLayout(dialog)

        info = QLabel(
            "Select one or more date directories to add to the build list. "
            "You can click multiple rows directly."
        )
        layout.addWidget(info)

        list_widget = QListWidget()
        list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        for directory in directories:
            item = QListWidgetItem(directory.name)
            item.setData(Qt.UserRole, str(directory))
            item.setToolTip(str(directory))
            list_widget.addItem(item)
        layout.addWidget(list_widget, 1)

        utility_row = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        clear_btn = QPushButton("Clear")
        utility_row.addWidget(select_all_btn)
        utility_row.addWidget(clear_btn)
        utility_row.addStretch(1)
        layout.addLayout(utility_row)

        select_all_btn.clicked.connect(list_widget.selectAll)
        clear_btn.clicked.connect(list_widget.clearSelection)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec_() != QDialog.Accepted:
            return []

        selected_paths: list[Path] = []
        for item in list_widget.selectedItems():
            raw = item.data(Qt.UserRole)
            if isinstance(raw, str):
                selected_paths.append(Path(raw))
        return selected_paths

    def _import_one(self, src: Path, ask_on_conflict: bool) -> bool:
        if not self._looks_like_demo(src):
            QMessageBox.warning(self, "Invalid folder", f"Missing movement CSV in:\n{src}")
            return False

        root = Path(self.data_root.text().strip())
        root.mkdir(parents=True, exist_ok=True)
        dst = root / src.name

        if src.resolve() != dst.resolve():
            if dst.exists():
                if ask_on_conflict:
                    ans = QMessageBox.question(
                        self,
                        "Folder exists",
                        f"{dst} exists. Reuse existing folder?",
                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                        QMessageBox.Yes,
                    )
                    if ans == QMessageBox.Cancel:
                        return False
                    if ans == QMessageBox.No:
                        shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                else:
                    pass
            else:
                shutil.copytree(src, dst)

        return self._add_folder(dst.name, dst)

    def _looks_like_demo(self, p: Path) -> bool:
        return (p / "movement_recording.csv").exists() or (p / "cv_movement_recording.csv").exists()

    def _add_folder(self, name: str, path: Path) -> bool:
        existing = {self.folder_list.item(i).data(Qt.UserRole) for i in range(self.folder_list.count())}
        if name in existing:
            return False
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, name)
        item.setToolTip(str(path))
        self.folder_list.addItem(item)
        self._append(f"Added folder: {name}")
        return True

    def remove_selected(self) -> None:
        for item in self.folder_list.selectedItems():
            self.folder_list.takeItem(self.folder_list.row(item))

    def load_metadata(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Select metadata.json", str(REPO_ROOT / "experiments" / "Datasets"), "JSON (*.json)"
        )
        if not p:
            return
        try:
            payload = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "Metadata error", str(exc))
            return

        self.metadata_path.setText(p)
        settings = payload.get("settings", {}) if isinstance(payload, dict) else {}
        selectors = payload.get("selectors", {}) if isinstance(payload, dict) else {}
        loaded_name = settings.get("dataset_name") or payload.get("dataset_name")
        self._set_if(self.dataset_name, loaded_name)
        if isinstance(loaded_name, str) and loaded_name.strip():
            stem = Path(loaded_name).stem.lower()
            is_test_name = "_test_dataset_" in stem or stem.endswith("_test")
            self.append_test_name.setChecked(is_test_name)
        self._set_if(self.val_ratio, settings.get("val_ratio"))
        self._set_if(self.random_seed, settings.get("random_seed"))
        self._set_if(self.freq_mask, settings.get("freq_mask", settings.get("frequency_mod")))
        self._set_if(self.image_resize, settings.get("image_resize"))
        self._set_if(self.inaction, settings.get("inaction"))
        self._set_if(self.inaction_tolerance, settings.get("inaction_tolerance"))
        self._set_if(self.skip_invalid_observations, settings.get("skip_invalid_observations"))
        self._set_if(self.gigaseal_cutoff_enabled, settings.get("gigaseal_resistance_cutoff_enabled"))
        self._set_if(self.gigaseal_cutoff_value, settings.get("gigaseal_resistance_cutoff"))
        self._set_if(self.slope_window_spin, settings.get("resistance_slope_window"))
        self._set_if(self.load_next_obs, settings.get("load_next_obs"))
        self._set_if(self.use_velocities, settings.get("use_velocities"))
        self._set_if(self.use_cv_defined_coords, settings.get("prefer_cv_movement"))
        self._set_if(self.cv_filter_images, settings.get("cv_filter_images"))
        self._set_if(self.cv_focus_with_detector_crop, settings.get("cv_focus_with_detector_crop"))
        self._set_if(
            self.cv_use_kalman_focus_fusion,
            settings.get("cv_use_kalman_focus_fusion"),
        )
        self._set_if(self.omit_stage_movement, settings.get("omit_stage_movement"))
        self._set_if(self.center_crop, settings.get("center_crop"))
        self._set_if(self.pipette_dot, settings.get("pipette_final_pos_color_dot"))

        fcfg = settings.get("filter", {})
        if isinstance(fcfg, dict):
            self._set_if(self.enable_filter, fcfg.get("enable_random_filter"))
            self._set_if(self.filter_prob, fcfg.get("image_filter_prob"))
            self._set_if(self.filter_train_only, fcfg.get("filter_train_only"))
            self._set_if(self.filter_same_demo, fcfg.get("filter_same_per_demo"))

        ocfg = settings.get("observation_selector", selectors.get("observations", {}))
        acfg = settings.get("action_selector", selectors.get("actions", {}))
        if isinstance(ocfg, dict):
            self._set_if(self.obs_pressure, ocfg.get("include_pressure"))
            self._set_if(self.obs_resistance, ocfg.get("include_resistance"))
            self._set_if(self.obs_resistance_slope, ocfg.get("include_resistance_slope"))
            self._set_if(self.obs_current, ocfg.get("include_current"))
            self._set_if(self.obs_voltage, ocfg.get("include_voltage"))
            self._set_if(self.obs_stage, ocfg.get("include_stage"))
            self._set_if(self.obs_pipette, ocfg.get("include_pipette"))
            self._set_if(self.obs_camera, ocfg.get("include_camera"))
            self._set_axis(self.obs_stage_x, self.obs_stage_y, self.obs_stage_z, ocfg.get("stage_axes"), "stage")
            self._set_axis(self.obs_pip_x, self.obs_pip_y, self.obs_pip_z, ocfg.get("pipette_axes"), "pipette")
        if isinstance(acfg, dict):
            self._set_if(self.act_stage, acfg.get("include_stage"))
            self._set_if(self.act_pipette, acfg.get("include_pipette"))
            self._set_if(self.act_pressure, acfg.get("include_pressure"))
            self._set_if(self.act_pressure_raw, acfg.get("pressure_use_raw_values"))
            self._set_axis(self.act_stage_x, self.act_stage_y, self.act_stage_z, acfg.get("stage_axes"), "stage")
            self._set_axis(self.act_pip_x, self.act_pip_y, self.act_pip_z, acfg.get("pipette_axes"), "pipette")

        folders = payload.get("processed_folders", [])
        if isinstance(folders, list) and folders:
            if QMessageBox.question(
                self, "Load folders", "Load processed_folders into the list?", QMessageBox.Yes | QMessageBox.No
            ) == QMessageBox.Yes:
                root = Path(self.data_root.text().strip())
                for f in folders:
                    if isinstance(f, str) and (root / f).is_dir():
                        self._add_folder(f, root / f)
        self._update_dataset_name_preview()
        self._sync_cv_generation_controls()
        self._sync_gigaseal_cutoff_controls()
        self._sync_selector_constraints()
        self._append(f"Loaded metadata: {p}")

    def _set_if(self, widget, value) -> None:
        if isinstance(widget, QLineEdit):
            if isinstance(value, str) and value.strip():
                widget.setText(value.strip())
            return
        if isinstance(widget, QCheckBox):
            if isinstance(value, bool):
                widget.setChecked(value)
            return
        if value is None:
            return
        try:
            widget.setValue(value)
        except Exception:
            pass

    def _set_axis(self, xcb, ycb, zcb, axis_value, prefix: str) -> None:
        if isinstance(axis_value, dict):
            xcb.setChecked(bool(axis_value.get("x", False)))
            ycb.setChecked(bool(axis_value.get("y", False)))
            zcb.setChecked(bool(axis_value.get("z", False)))
            return
        if isinstance(axis_value, list):
            vals = {str(v).lower() for v in axis_value}
            xcb.setChecked("x" in vals or f"{prefix}_x" in vals)
            ycb.setChecked("y" in vals or f"{prefix}_y" in vals)
            zcb.setChecked("z" in vals or f"{prefix}_z" in vals)

    def _selected_folders(self):
        return [self.folder_list.item(i).data(Qt.UserRole) for i in range(self.folder_list.count())]

    def _collect_kwargs(self):
        name = self._effective_dataset_name(self.dataset_name.text())
        return dict(
            dataset_name=name,
            val_ratio=float(self.val_ratio.value()),
            omit_stage_movement=self.omit_stage_movement.isChecked(),
            random_seed=int(self.random_seed.value()),
            freq_mask=int(self.freq_mask.value()),
            load_next_obs=self.load_next_obs.isChecked(),
            use_velocities=self.use_velocities.isChecked(),
            prefer_cv_movement=self.use_cv_defined_coords.isChecked(),
            filter=FilterSettings(
                enable_random_filter=self.enable_filter.isChecked(),
                image_filter_prob=float(self.filter_prob.value()),
                filter_train_only=self.filter_train_only.isChecked(),
                filter_same_per_demo=self.filter_same_demo.isChecked(),
            ),
            image_resize=int(self.image_resize.value()),
            pipette_final_pos_color_dot=self.pipette_dot.isChecked(),
            center_crop=self.center_crop.isChecked(),
            inaction=int(self.inaction.value()),
            inaction_tolerance=float(self.inaction_tolerance.value()),
            skip_invalid_observations=self.skip_invalid_observations.isChecked(),
            gigaseal_resistance_cutoff_enabled=self.gigaseal_cutoff_enabled.isChecked(),
            gigaseal_resistance_cutoff=float(self.gigaseal_cutoff_value.value()),
            resistance_slope_window=int(self.slope_window_spin.value()),
            observation_selector=ObservationSelector(
                include_pressure=self.obs_pressure.isChecked(),
                include_resistance=self.obs_resistance.isChecked(),
                include_resistance_slope=self.obs_resistance_slope.isChecked(),
                include_current=self.obs_current.isChecked(),
                include_voltage=self.obs_voltage.isChecked(),
                include_stage=self.obs_stage.isChecked(),
                include_pipette=self.obs_pipette.isChecked(),
                include_camera=self.obs_camera.isChecked(),
                stage_axes=AxisToggle(self.obs_stage_x.isChecked(), self.obs_stage_y.isChecked(), self.obs_stage_z.isChecked()),
                pipette_axes=AxisToggle(self.obs_pip_x.isChecked(), self.obs_pip_y.isChecked(), self.obs_pip_z.isChecked()),
            ),
            action_selector=ActionSelector(
                include_stage=self.act_stage.isChecked(),
                include_pipette=self.act_pipette.isChecked(),
                include_pressure=self.act_pressure.isChecked(),
                pressure_use_raw_values=self.act_pressure_raw.isChecked(),
                include_high_level=False,
                stage_axes=AxisToggle(self.act_stage_x.isChecked(), self.act_stage_y.isChecked(), self.act_stage_z.isChecked()),
                pipette_axes=AxisToggle(self.act_pip_x.isChecked(), self.act_pip_y.isChecked(), self.act_pip_z.isChecked()),
            ),
        )

    def _generate_cv_movement_files(self, folders: list[str]) -> None:
        from experiments.ImageDatasetPreparer import ImageDatasetPreparer

        rig_root = Path(self.data_root.text().strip())
        if not rig_root.is_dir():
            raise FileNotFoundError(f"Rig data root not found: {rig_root}")
        filter_images = self.cv_filter_images.isChecked()
        focus_with_detector_crop = self.cv_focus_with_detector_crop.isChecked()
        use_kalman_focus_fusion = self.cv_use_kalman_focus_fusion.isChecked()
        preparer = ImageDatasetPreparer(
            rig_root,
            use_detector1=False,
            filter_images=filter_images,
            focus_with_detector_crop=focus_with_detector_crop,
            use_kalman_focus_fusion=use_kalman_focus_fusion,
        )

        self._append(
            "Generating cv_movement_recording.csv files "
            f"(filter_images={'on' if filter_images else 'off'}, "
            f"focus_crop={'on' if focus_with_detector_crop else 'off'}, "
            f"kalman={'on' if use_kalman_focus_fusion else 'off'})..."
        )
        for i, folder in enumerate(folders, 1):
            self._append(f"[cv {i}/{len(folders)}] {folder}")
            QApplication.processEvents()
            preparer.build_csv(folder, output_name="cv_movement_recording.csv")
        self._append("CV coordinate generation complete.")

    def build_dataset(self) -> None:
        folders = [f for f in self._selected_folders() if isinstance(f, str)]
        if not folders:
            QMessageBox.warning(self, "No folders", "Import at least one folder.")
            return
        if not self.dataset_name.text().strip():
            QMessageBox.warning(self, "Dataset name", "Dataset name is required.")
            return

        kwargs = self._collect_kwargs()
        self._append(f"Building dataset: {kwargs['dataset_name']}")
        self._append(f"Folders: {', '.join(folders)}")

        self.setEnabled(False)
        try:
            if self.use_cv_defined_coords.isChecked():
                self._generate_cv_movement_files(folders)

            builder = SimpleDatasetBuilder(**kwargs)
            for i, folder in enumerate(folders, 1):
                self._append(f"[{i}/{len(folders)}] {folder}")
                QApplication.processEvents()
                builder.add_demo(folder, record_to_file=True)
            builder.write_split_masks()
            self._append(f"Done. Dataset path: {builder.dataset_path}")
            QMessageBox.information(self, "Completed", f"Dataset built:\n{builder.dataset_path}")
        except Exception:
            err = traceback.format_exc()
            self._append(err)
            QMessageBox.critical(self, "Build failed", err)
        finally:
            self.setEnabled(True)


def main() -> int:
    app = QApplication(sys.argv)
    win = DatasetBuilderGUI()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
