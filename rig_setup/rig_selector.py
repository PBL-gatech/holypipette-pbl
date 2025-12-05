from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QFormLayout,
)

from .rig_config import DEVICE_SLOTS, RigConfigError, RigConfigManager


def _get_nested(params: Dict[str, Any], dotted: str, default=None):
    cur = params
    parts = dotted.split(".")
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _set_nested(params: Dict[str, Any], dotted: str, value: Any) -> None:
    cur = params
    parts = dotted.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _coerce(text: str, typ: str) -> Any:
    if typ == "int":
        return int(text)
    if typ == "float":
        return float(text)
    if typ == "bool":
        return text.strip().lower() in ("1", "true", "yes", "on")
    return text


class SettingsDialog(QDialog):
    def __init__(self, slot: str, fields: List[Dict[str, Any]], params: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{slot} settings")
        self.fields = fields
        self.params = params.copy()
        self.inputs: Dict[str, QLineEdit] = {}
        layout = QVBoxLayout()
        form = QFormLayout()
        layout.addLayout(form)

        if not fields:
            label = QLabel("No configurable settings for this device.")
            label.setStyleSheet("color: gray;")
            layout.addWidget(label)
        else:
            for field in fields:
                key = field["key"]
                label_text = field.get("label", key)
                typ = field.get("type", "str")
                readonly = field.get("readonly", False)
                val = _get_nested(self.params, key, "")
                line = QLineEdit(str(val) if val not in (None, "") else "")
                line.setReadOnly(readonly)
                line.setPlaceholderText(typ)
                form.addRow(QLabel(label_text), line)
                self.inputs[key] = line

        btn_row = QHBoxLayout()
        save = QPushButton("OK")
        cancel = QPushButton("Cancel")
        save.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(save)
        btn_row.addWidget(cancel)
        layout.addLayout(btn_row)
        self.setLayout(layout)

    def get_params(self) -> Dict[str, Any]:
        params = self.params.copy()
        for key, widget in self.inputs.items():
            text = widget.text().strip()
            field = next((f for f in self.fields if f["key"] == key), {})
            typ = field.get("type", "str")
            if text == "" and field.get("optional"):
                continue
            try:
                val = _coerce(text, typ)
            except Exception:
                val = text
            _set_nested(params, key, val)
        return params


class RigBuilderDialog(QDialog):
    """Dialog for composing a new rig configuration."""

    def __init__(self, manager: RigConfigManager, parent=None, initial_config: Dict[str, Any] | None = None, save_path: Path | None = None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Create Rig Configuration")
        self.resize(900, 600)
        self.saved_path: Path | None = None
        self.edit_path = save_path
        self.options = manager.get_device_options()
        self.slot_rows: Dict[str, Dict[str, Any]] = {}
        self.initial_config = initial_config
        self._build_ui()
        if self.initial_config:
            self._load_config(self.initial_config)

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        self.setLayout(layout)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Rig name:"))
        self.name_edit = QLineEdit(self.initial_config.get("name", "Custom Rig") if self.initial_config else "Custom Rig")
        name_row.addWidget(self.name_edit)
        layout.addLayout(name_row)

        for slot in DEVICE_SLOTS:
            row = QHBoxLayout()
            row.addWidget(QLabel(slot))
            combo = QComboBox()
            for opt in self.options.get(slot, []):
                combo.addItem(opt.get("label", opt.get("class", "unknown")), opt)
            combo.currentIndexChanged.connect(lambda _, s=slot, c=combo: self._apply_option_defaults(s, c))
            settings_btn = QPushButton("Settings")
            settings_btn.clicked.connect(lambda _, s=slot: self._open_settings(s))
            row.addWidget(combo, stretch=3)
            row.addWidget(settings_btn, stretch=1)
            layout.addLayout(row)
            self.slot_rows[slot] = {"combo": combo, "params": {}}
            self._apply_option_defaults(slot, combo)

        btns = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        save_btn.clicked.connect(self._on_save)
        cancel_btn.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(save_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

    def _apply_option_defaults(self, slot: str, combo: QComboBox) -> None:
        opt = combo.currentData()
        if not isinstance(opt, dict):
            return
        params = opt.get("params", {})
        self.slot_rows[slot]["params"] = params.copy()

    def _open_settings(self, slot: str) -> None:
        widgets = self.slot_rows[slot]
        opt = widgets["combo"].currentData()
        if not isinstance(opt, dict):
            QMessageBox.warning(self, "No device", "Select a device first.")
            return
        fields = opt.get("fields", [])
        dlg = SettingsDialog(slot, fields, widgets["params"], self)
        if dlg.exec_() == QDialog.Accepted:
            new_params = dlg.get_params()
            widgets["params"] = new_params

    def _on_save(self) -> None:
        name = self.name_edit.text().strip() or "Custom Rig"
        devices: Dict[str, Any] = {}

        try:
            for slot, widgets in self.slot_rows.items():
                opt = widgets["combo"].currentData()
                if not opt or "class" not in opt:
                    raise RigConfigError(f"No class selected for slot '{slot}'.")
                devices[slot] = {"class": opt["class"], "params": widgets["params"]}
        except (RigConfigError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Invalid configuration", str(exc))
            return

        config = {"name": name, "schema_version": 2, "devices": devices}
        path = self.edit_path or self.manager.make_config_path(name)
        try:
            self.manager.save_config(path, config)
        except Exception as exc:  # pragma: no cover
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        self.saved_path = path
        self.accept()

    def _load_config(self, config: Dict[str, Any]) -> None:
        devices = config.get("devices", {})
        for slot, spec in devices.items():
            if slot not in self.slot_rows:
                continue
            target_class = spec.get("class")
            combo: QComboBox = self.slot_rows[slot]["combo"]
            # find matching index
            match_idx = -1
            for i in range(combo.count()):
                data = combo.itemData(i)
                if isinstance(data, dict) and data.get("class") == target_class:
                    match_idx = i
                    break
            if match_idx >= 0:
                combo.setCurrentIndex(match_idx)
            params = spec.get("params", {})
            self.slot_rows[slot]["params"] = params


class RigSelectorDialog(QDialog):
    """Dialog to select or create a rig configuration before launching the GUI."""

    def __init__(self, manager: RigConfigManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Select Rig Configuration")
        self.resize(500, 200)
        self.selected_path: Path | None = None
        self.manager.ensure_default_config()
        self._build_ui()
        self._reload_configs()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel("Choose a rig configuration JSON to load:"))
        self.config_combo = QComboBox()
        self.config_combo.currentIndexChanged.connect(self._update_path_display)
        layout.addWidget(self.config_combo)

        btn_row = QHBoxLayout()
        browse_btn = QPushButton("Browse...")
        new_btn = QPushButton("New config")
        edit_btn = QPushButton("Edit selected")
        refresh_btn = QPushButton("Refresh list")
        browse_btn.clicked.connect(self._browse_for_config)
        new_btn.clicked.connect(self._create_config)
        edit_btn.clicked.connect(self._edit_config)
        refresh_btn.clicked.connect(self._reload_configs)
        btn_row.addWidget(browse_btn)
        btn_row.addWidget(new_btn)
        btn_row.addWidget(edit_btn)
        btn_row.addWidget(refresh_btn)
        layout.addLayout(btn_row)

        action_row = QHBoxLayout()
        load_btn = QPushButton("Load")
        cancel_btn = QPushButton("Cancel")
        load_btn.clicked.connect(self._on_accept)
        cancel_btn.clicked.connect(self.reject)
        action_row.addStretch()
        action_row.addWidget(load_btn)
        action_row.addWidget(cancel_btn)
        layout.addLayout(action_row)

    def _reload_configs(self) -> None:
        current = self.config_combo.currentData()
        self.config_combo.blockSignals(True)
        self.config_combo.clear()
        for path in self.manager.list_configs():
            self.config_combo.addItem(path.name, path)
        self.config_combo.blockSignals(False)

        if current:
            idx = self.config_combo.findData(current)
            if idx >= 0:
                self.config_combo.setCurrentIndex(idx)
        if self.config_combo.count() and self.config_combo.currentIndex() < 0:
            self.config_combo.setCurrentIndex(0)
        self._update_path_display()

    def _update_path_display(self) -> None:
        pass

    def _browse_for_config(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(self, "Select rig config", str(self.manager.config_dir), "JSON (*.json)")
        if path_str:
            self.selected_path = Path(path_str)

    def _create_config(self) -> None:
        builder = RigBuilderDialog(self.manager, self)
        if builder.exec_() == QDialog.Accepted and builder.saved_path:
            self.selected_path = builder.saved_path
            self._reload_configs()
            idx = self.config_combo.findData(builder.saved_path)
            if idx >= 0:
                self.config_combo.setCurrentIndex(idx)

    def _edit_config(self) -> None:
        path = self.selected_path or self.config_combo.currentData()
        if not isinstance(path, Path):
            QMessageBox.warning(self, "No configuration", "Select a config to edit.")
            return
        try:
            config = self.manager.load_config(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        builder = RigBuilderDialog(self.manager, self, initial_config=config, save_path=path)
        if builder.exec_() == QDialog.Accepted and builder.saved_path:
            self.selected_path = builder.saved_path
            self._reload_configs()
            idx = self.config_combo.findData(builder.saved_path)
            if idx >= 0:
                self.config_combo.setCurrentIndex(idx)

    def _on_accept(self) -> None:
        if self.selected_path is None:
            data = self.config_combo.currentData()
            if isinstance(data, Path):
                self.selected_path = data
        if not self.selected_path:
            QMessageBox.warning(self, "No configuration", "Please select or create a rig configuration.")
            return
        self.accept()
