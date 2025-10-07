import sys
import h5py
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QGraphicsScene, QGraphicsView, QPushButton, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QPen, QPainter, QBrush
from PyQt5.QtCore import Qt, QTimer, QPointF


class DemoPlayer(QWidget):
    def __init__(self, hdf5_path):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.playing = True  # video plays by default
        self.pipette_positions = np.empty((0, 2), dtype=np.float32)
        self._pipette_frame_count = 0
        self.pipette_tail_length = 25

        # Try opening the HDF5 file.
        try:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        except OSError as e:
            QMessageBox.critical(self, "File Error", f"Could not open file {hdf5_path}:\n{e}")
            sys.exit(1)

        if 'data' not in self.hdf5_file:
            QMessageBox.critical(self, "File Error", f"The file {hdf5_path} does not contain a 'data' group.")
            sys.exit(1)

        self.demo_keys = sorted(self.hdf5_file['data'].keys())
        if not self.demo_keys:
            QMessageBox.critical(self, "Data Error", "No demos found in the HDF5 file!")
            sys.exit(1)

        self.current_demo_idx = 0
        self.current_frame = 0

        self.init_ui()
        self.load_demo(self.current_demo_idx)

    # ------------------------------------------------------------------
    # 1. init_ui() — removed hard-coded 500 × 500, added centring
    # ------------------------------------------------------------------
    def init_ui(self):
        self.setWindowTitle("Demo Viewer")
        self.resize(800, 500)

        main_layout = QHBoxLayout(self)

        # Left side: video display and navigation buttons.
        left_layout = QVBoxLayout()

        self.video_label = QLabel(self)
        # self.video_label.setFixedSize(500, 500)   # ← removed per request
        self.video_label.setAlignment(Qt.AlignCenter)     # keep pixmap centred
        left_layout.addWidget(self.video_label)

        # Button layout for navigation.
        button_layout = QHBoxLayout()

        self.prev_button = QPushButton("Previous Demo", self)
        self.prev_button.clicked.connect(self.prev_demo)
        button_layout.addWidget(self.prev_button)

        self.play_pause_button = QPushButton("Pause", self)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        button_layout.addWidget(self.play_pause_button)

        self.next_button = QPushButton("Next Demo", self)
        self.next_button.clicked.connect(self.next_demo)
        button_layout.addWidget(self.next_button)

        left_layout.addLayout(button_layout)
        main_layout.addLayout(left_layout)

        # Right side: resistance plot area (fixed to 400x400).
        self.scene = QGraphicsScene(self)
        self.plot_view = QGraphicsView(self.scene, self)
        self.plot_view.setFixedSize(500, 500)
        main_layout.addWidget(self.plot_view)


        self.action_scene = QGraphicsScene(self)
        self.action_view  = QGraphicsView(self.action_scene, self)
        self.action_view.setFixedSize(500, 400)      # tweak height as you like
        main_layout.addWidget(self.action_view)



        self.setLayout(main_layout)

        # Timer for video frame updates (~33 fps).
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ------------------------------------------------------------------
    # 2. load_demo() — resize label to native image size (once)
    # ------------------------------------------------------------------
    def load_demo(self, idx):
        """Load images (video) and resistance data for the given demo."""
        self.current_frame = 0
        demo_key = self.demo_keys[idx]
        demo_path = f'data/{demo_key}/obs'
        print(f"Loading demo: {demo_key}")  # Debug message

        self.actions = np.zeros((0, 6))
        self.pipette_positions = np.empty((0, 2), dtype=np.float32)
        self._pipette_frame_count = 0

        for name, attr in (('camera_image', 'images'), ('resistance', 'resistance')):
            try:
                setattr(self, attr, self.hdf5_file[f'{demo_path}/{name}'][:])
            except KeyError as e:
                print(f"Warning: missing '{name}' in {demo_path}: {e}")
                setattr(self, attr, np.array([]))

        try:
            data = self.hdf5_file[f'data/{demo_key}/actions'][:]
            if data.ndim == 2:
                cols = min(data.shape[1], 6)
                self.actions = np.zeros((data.shape[0], 6))
                self.actions[:, :cols] = data[:, :cols]
            else:
                print(f"Warning: unexpected 'actions' shape {data.shape} in data/{demo_key}")
        except KeyError:
            print(f"Warning: missing 'actions' dataset in data/{demo_key}")



        # Resize the label *once* to the first frame’s native size
        if self.images.size:
            h_img, w_img = self.images[0].shape[:2]
            self.video_label.setFixedSize(w_img, h_img)

        # Validate the resistance data.
        # if self.resistance.ndim != 1:
        #     QMessageBox.critical(
        #         self,
        #         "Data Error",
        #         f"Expected 'resistance' to be 1D, got shape {self.resistance.shape}"
        #     )
        #     self.resistance = np.array([])
        if self.resistance.ndim != 1:
            # reshape if possible
            if self.resistance.ndim == 2 and 1 in self.resistance.shape:
                self.resistance = self.resistance.flatten()
            else:
                print(f"Warning: expected 1D resistance, got {self.resistance.shape}")
                self.resistance = np.array([])

        self._load_pipette_positions(demo_path)
        self.plot_resistance()
        self.plot_actions()

    def _load_pipette_positions(self, demo_path):
        """Load pipette positions for the current demo if available."""
        self.pipette_positions = np.empty((0, 2), dtype=np.float32)
        self._pipette_frame_count = 0

        try:
            dataset = self.hdf5_file[f'{demo_path}/pipette_positions']
        except KeyError:
            return

        data = np.asarray(dataset[:], dtype=np.float32)

        if data.ndim == 1:
            if data.size % 3 == 0:
                cols = 3
            elif data.size % 2 == 0:
                cols = 2
            else:
                cols = 1
            data = data.reshape(-1, cols)

        if data.size == 0:
            return

        if data.ndim != 2 or data.shape[1] < 2:
            print(f"Warning: '{demo_path}/pipette_positions' has shape {data.shape}, expected >= (N, 2)")
            return

        positions = np.asarray(data[:, :2], dtype=np.float32)

        frame_dims = None
        if self.images.size:
            frame_count = min(len(self.images), positions.shape[0])
            if frame_count != positions.shape[0]:
                print(f"Warning: trimming pipette positions from {positions.shape[0]} to {frame_count} to match frames")
            positions = positions[:frame_count]
            h_img, w_img = self.images[0].shape[:2]
            frame_dims = np.array([w_img, h_img], dtype=np.float32)
        else:
            frame_count = positions.shape[0]

        if frame_count == 0:
            return

        def _in_frame(arr: np.ndarray, dims: np.ndarray, margin: float = 0.5) -> bool:
            if arr.size == 0 or dims is None:
                return True
            x_ok = np.logical_and(arr[:, 0] >= -margin, arr[:, 0] <= dims[0] - 1.0 + margin)
            y_ok = np.logical_and(arr[:, 1] >= -margin, arr[:, 1] <= dims[1] - 1.0 + margin)
            return bool(np.all(x_ok) and np.all(y_ok))

        if frame_dims is not None and not _in_frame(positions, frame_dims):
            base_scale = np.where(frame_dims > 0.0, frame_dims / 1280.0, 0.0)
            scaled = positions * base_scale
            if _in_frame(scaled, frame_dims):
                positions = scaled
                print(f"Info: scaled pipette positions to match {int(frame_dims[0])}x{int(frame_dims[1])} frames")
            else:
                upper = np.maximum(frame_dims - 1.0, 0.0)
                outside_mask = np.logical_or(np.any(scaled < 0.0, axis=1), np.any(scaled > upper, axis=1))
                if np.any(outside_mask):
                    count = int(outside_mask.sum())
                    print(f"Warning: clipped {count} pipette position rows to image bounds ({int(frame_dims[0])}x{int(frame_dims[1])})")
                positions = np.clip(scaled, [0.0, 0.0], upper)

        finite_mask = np.all(np.isfinite(positions), axis=1)
        if not np.all(finite_mask):
            dropped = int(finite_mask.size - finite_mask.sum())
            print(f"Warning: dropping {dropped} pipette position rows with non-finite values")
            positions = positions[finite_mask]

        self.pipette_positions = positions
        self._pipette_frame_count = self.pipette_positions.shape[0]

        if self._pipette_frame_count and frame_dims is not None and not _in_frame(self.pipette_positions, frame_dims):
            print(f"Warning: pipette positions fall outside image bounds ({int(frame_dims[0])}x{int(frame_dims[1])}); overlay may not be visible")

    # ------------------------------------------------------------------
    # Resistance-plotting code (unchanged)
    # ------------------------------------------------------------------
    def plot_resistance(self):
        self.scene.clear()
        if self.resistance.size == 0:
            self.scene.addText("No resistance data available")
            return

        plot_w, plot_h = 400, 350
        max_res = float(np.max(self.resistance))
        min_res = float(np.min(self.resistance))
        points = len(self.resistance)

        if points < 2:
            self.scene.addText("Not enough data points for plot")
            return

        pen = QPen(Qt.blue)
        step = plot_w / (points - 1)
        diff = max_res - min_res if max_res != min_res else 1e-6

        for i in range(points - 1):
            x1 = i * step
            y1 = plot_h - ((self.resistance[i] - min_res) / diff) * plot_h
            x2 = (i + 1) * step
            y2 = plot_h - ((self.resistance[i + 1] - min_res) / diff) * plot_h
            self.scene.addLine(x1, y1, x2, y2, pen)

        # y-axis labels
        for i in range(6):
            y = plot_h - (i / 5) * plot_h
            value = min_res + (i / 5) * diff
            t = self.scene.addText(f"{value:.2f}")
            t.setPos(-50, y - t.boundingRect().height() / 2)

        # x-axis labels: first and last
        t0 = self.scene.addText("0")
        t0.setPos(0, plot_h + 5)
        tN = self.scene.addText(f"{points - 1}")
        tN.setPos(plot_w - tN.boundingRect().width(), plot_h + 5)

    def plot_actions(self):
        """Draw a 6-column bar chart of non-zero action counts."""
        self.action_scene.clear()

        counts = np.zeros(6, dtype=int)
        if self.actions.size:
            raw = np.count_nonzero(self.actions, axis=0)
            counts[:min(raw.size, 6)] = raw[:6]

        plot_w, plot_h = 400, 200
        bar_w   = plot_w / 6
        max_ct  = counts.max() or 1

        pen   = QPen(Qt.black)
        brush = Qt.gray

        for i, c in enumerate(counts):
            x = i * bar_w
            h = (c / max_ct) * plot_h
            # bar
            self.action_scene.addRect(x, plot_h - h, bar_w * 0.8, h, pen, brush)
            # value label
            t = self.action_scene.addText(str(int(c)))
            t.setPos(x + bar_w*0.4 - t.boundingRect().width()/2,
                     plot_h - h - t.boundingRect().height() - 2)

        # x-axis labels
        for i in range(6):
            lbl = self.action_scene.addText(str(i))
            lbl.setPos(i*bar_w + bar_w*0.4 - lbl.boundingRect().width()/2,
                       plot_h + 5)


    def _draw_pipette_overlay(self, pixmap, frame_idx):
        if self._pipette_frame_count == 0:
            return

        if frame_idx >= self._pipette_frame_count:
            frame_idx = self._pipette_frame_count - 1
        if frame_idx < 0:
            return

        point = self.pipette_positions[frame_idx]
        if not np.all(np.isfinite(point)):
            return

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(Qt.red)
        pen_width = max(2, max(pixmap.width(), pixmap.height()) // 180)
        pen.setWidth(int(pen_width))
        painter.setPen(pen)
        painter.setBrush(QBrush(Qt.red))

        radius = max(3, min(pixmap.width(), pixmap.height()) // 50)
        painter.drawEllipse(QPointF(float(point[0]), float(point[1])), radius, radius)

        if frame_idx > 0 and self.pipette_tail_length > 0:
            start_idx = max(0, frame_idx - self.pipette_tail_length)
            tail = self.pipette_positions[start_idx:frame_idx + 1]
            finite_tail = tail[np.all(np.isfinite(tail), axis=1)]
            if finite_tail.shape[0] > 1:
                q_points = [QPointF(float(p[0]), float(p[1])) for p in finite_tail]
                for p0, p1 in zip(q_points[:-1], q_points[1:]):
                    painter.drawLine(p0, p1)

        painter.end()
    # ------------------------------------------------------------------
    # 3. update_frame() — shrink oversized frames, never upscale
    # ------------------------------------------------------------------
    def update_frame(self):
        if self.images.size == 0:
            return

        if self.current_frame >= len(self.images):
            self.current_frame = 0

        frame_idx = self.current_frame

        try:
            img_array = self.images[frame_idx]
        except IndexError:
            self.current_frame = 0
            return

        # Handle grayscale vs. colour frames.
        if img_array.ndim == 2:
            h_img, w_img = img_array.shape
            bytes_per_line = w_img
            q_img = QImage(img_array.data, w_img, h_img, bytes_per_line, QImage.Format_Grayscale8)
        elif img_array.ndim == 3 and img_array.shape[-1] in (3, 4):
            h_img, w_img, channels = img_array.shape
            bytes_per_line = w_img * channels
            q_format = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
            q_img = QImage(img_array.data, w_img, h_img, bytes_per_line, q_format)
        else:
            return  # unexpected format

        pixmap = QPixmap.fromImage(q_img)
        self._draw_pipette_overlay(pixmap, frame_idx)

        # Only shrink if the frame exceeds the label dimensions
        if (pixmap.width() > self.video_label.width() or
                pixmap.height() > self.video_label.height()):
            pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

        self.video_label.setPixmap(pixmap)
        self.current_frame = frame_idx + 1

    # ------------------------------------------------------------------
    # Control / navigation handlers (unchanged)
    # ------------------------------------------------------------------
    def toggle_play_pause(self):
        if self.playing:
            self.timer.stop()
            self.play_pause_button.setText("Play")
        else:
            self.timer.start(30)
            self.play_pause_button.setText("Pause")
        self.playing = not self.playing

    def next_demo(self):
        self.current_demo_idx = (self.current_demo_idx + 1) % len(self.demo_keys)
        self.load_demo(self.current_demo_idx)

    def prev_demo(self):
        self.current_demo_idx = (self.current_demo_idx - 1) % len(self.demo_keys)
        self.load_demo(self.current_demo_idx)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.next_demo()
        elif event.key() == Qt.Key_Left:
            self.prev_demo()
        elif event.key() == Qt.Key_Space:
            self.toggle_play_pause()

    def closeEvent(self, event):
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()
        event.accept()


# ----------------------------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    data_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\PatcherBot_test_dataset_v0_180\PatcherBot_test_dataset_v0_180_find_pipette.hdf5"

    viewer = DemoPlayer(data_path)
    viewer.show()
    sys.exit(app.exec_())
