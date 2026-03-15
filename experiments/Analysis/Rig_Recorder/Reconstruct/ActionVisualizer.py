import os
import sys

import h5py
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QPainter, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QComboBox,
)


class ActionVisualizer(QWidget):
    """Visualize aggregated action deltas across an entire HDF5 dataset."""

    def __init__(self, hdf5_path):
        """
        Visualize aggregated action deltas stored in an HDF5 dataset.
        
        Args:
            hdf5_path (str or Path): Path to the HDF5 file containing the dataset
                with action values to visualize.
        """
        super().__init__()
        self.hdf5_path = hdf5_path
        self.actions = None
        self.action_dim = 0
        self.demo_count = 0

        self._init_ui()
        self._load_dataset()
        self.update_histogram()

    # ------------------------------------------------------------------
    def _init_ui(self):
        """
        Initialize and configure the graphical user interface for the action histogram viewer.
        """
        self.setWindowTitle("Action Histogram Viewer")
        self.resize(900, 600)

        root_layout = QVBoxLayout(self)

        self.dataset_label = QLabel(self)
        root_layout.addWidget(self.dataset_label)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Action component:", self))

        self.action_selector = QComboBox(self)
        controls.addWidget(self.action_selector)

        controls.addWidget(QLabel("Bins:", self))
        self.bin_selector = QSpinBox(self)
        self.bin_selector.setRange(5, 200)
        self.bin_selector.setValue(40)
        controls.addWidget(self.bin_selector)

        self.refresh_button = QPushButton("Update", self)
        controls.addWidget(self.refresh_button)
        controls.addStretch(1)

        root_layout.addLayout(controls)

        self.stats_label = QLabel(self)
        root_layout.addWidget(self.stats_label)

        self.hist_scene = QGraphicsScene(self)
        self.hist_view = QGraphicsView(self.hist_scene, self)
        self.hist_view.setRenderHint(QPainter.Antialiasing)
        self.hist_view.setMinimumSize(800, 400)
        root_layout.addWidget(self.hist_view)

        self.refresh_button.clicked.connect(self.update_histogram)
        self.action_selector.currentIndexChanged.connect(self.update_histogram)
        self.bin_selector.valueChanged.connect(self.update_histogram)

    # ------------------------------------------------------------------
    def _load_dataset(self):
        """
        Load and aggregate action data from an HDF5 dataset.
        """
        try:
            h5_file = h5py.File(self.hdf5_path, "r")
        except OSError as exc:
            QMessageBox.critical(self, "File Error", f"Could not open file:\n{exc}")
            sys.exit(1)

        with h5_file as f:
            if "data" not in f:
                QMessageBox.critical(
                    self,
                    "File Error",
                    "The selected file does not contain a 'data' group.",
                )
                sys.exit(1)

            all_actions = []
            expected_dim = None

            for demo_key in f["data"].keys():
                demo_group = f["data"][demo_key]
                if "actions" not in demo_group:
                    continue

                actions_ds = demo_group["actions"][:]
                if actions_ds.size == 0:
                    continue

                arr = np.asarray(actions_ds)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                elif arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)

                if expected_dim is None:
                    expected_dim = arr.shape[1]
                elif arr.shape[1] != expected_dim:
                    QMessageBox.warning(
                        self,
                        "Dimension Mismatch",
                        (
                            f"Skipping demo '{demo_key}' because its action dimension "
                            f"({arr.shape[1]}) does not match prior demos ({expected_dim})."
                        ),
                    )
                    continue

                all_actions.append(arr)

            if not all_actions:
                QMessageBox.critical(
                    self,
                    "Data Error",
                    "No usable 'actions' datasets were found in the file.",
                )
                sys.exit(1)

            self.actions = np.vstack(all_actions)
            self.action_dim = expected_dim or self.actions.shape[1]
            self.demo_count = len(all_actions)

        dataset_name = os.path.basename(self.hdf5_path) or self.hdf5_path
        self.dataset_label.setText(
            f"Dataset: {dataset_name} — {self.demo_count} demos, {self.actions.shape[0]:,} samples"
        )

        self.action_selector.clear()
        for idx in range(self.action_dim):
            self.action_selector.addItem(f"Action {idx}")

    # ------------------------------------------------------------------
    def update_histogram(self):
        """
        Compute and render a histogram of the selected action component.
        """
        if self.actions is None or self.actions.size == 0:
            self.hist_scene.clear()
            self.hist_scene.addText("No action data to display")
            return

        action_idx = self.action_selector.currentIndex()
        if action_idx < 0 or action_idx >= self.action_dim:
            return

        data = self.actions[:, action_idx]
        bins = self.bin_selector.value()

        counts, edges = np.histogram(data, bins=bins)
        max_count = counts.max() if counts.size else 1
        max_count = max(max_count, 1)

        plot_w = 700
        plot_h = 360
        bar_space = plot_w / bins if bins else plot_w

        self.hist_scene.clear()

        axis_pen = QPen(Qt.black)
        axis_pen.setWidth(2)
        self.hist_scene.addLine(0, plot_h, plot_w, plot_h, axis_pen)
        self.hist_scene.addLine(0, 0, 0, plot_h, axis_pen)

        bar_pen = QPen(Qt.black)
        bar_brush = QBrush(Qt.darkGray)

        for idx, count in enumerate(counts):
            if count == 0:
                continue
            height = (count / max_count) * plot_h
            x_pos = idx * bar_space
            bar_width = bar_space * 0.85
            rect = self.hist_scene.addRect(
                x_pos,
                plot_h - height,
                bar_width,
                height,
                bar_pen,
                bar_brush,
            )
            rect.setToolTip(f"Count: {count}\nRange: [{edges[idx]:.4f}, {edges[idx + 1]:.4f})")

        tick_pen = QPen(Qt.gray)
        tick_pen.setStyle(Qt.DashLine)
        tick_count = 5
        value_range = edges[-1] - edges[0]
        for tick in range(tick_count + 1):
            x = (tick / tick_count) * plot_w
            value = edges[0] + (tick / tick_count) * value_range
            self.hist_scene.addLine(x, plot_h, x, plot_h + 5, axis_pen)
            label = self.hist_scene.addText(f"{value:.3f}")
            label.setPos(x - label.boundingRect().width() / 2, plot_h + 8)

        for tick in range(1, tick_count + 1):
            y = plot_h - (tick / tick_count) * plot_h
            self.hist_scene.addLine(0, y, -5, y, axis_pen)
            label = self.hist_scene.addText(f"{(tick / tick_count) * max_count:.0f}")
            label.setPos(-label.boundingRect().width() - 10, y - label.boundingRect().height() / 2)
            self.hist_scene.addLine(0, y, plot_w, y, tick_pen)

        self.hist_scene.setSceneRect(-100, -20, plot_w + 150, plot_h + 120)
        self.hist_view.fitInView(self.hist_scene.sceneRect(), Qt.KeepAspectRatio)

        stats = (
            f"Action {action_idx}: samples={data.size:,}  "
            f"mean={np.mean(data):.4f}  std={np.std(data):.4f}  "
            f"min={np.min(data):.4f}  max={np.max(data):.4f}"
        )
        self.stats_label.setText(stats)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    if len(sys.argv) > 1:
        hdf5_path = sys.argv[1]
    else:
        hdf5_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select HDF5 dataset",
            "",
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if not hdf5_path:
            sys.exit(0)

    viewer = ActionVisualizer(hdf5_path)
    viewer.show()
    sys.exit(app.exec_())
