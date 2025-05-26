import sys
import h5py
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QGraphicsScene, QGraphicsView, QPushButton, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QPen
from PyQt5.QtCore import Qt, QTimer


class DemoPlayer(QWidget):
    def __init__(self, hdf5_path):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.playing = True  # video plays by default

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

        try:
            self.images = self.hdf5_file[f'{demo_path}/camera_image'][:]
            self.resistance = self.hdf5_file[f'{demo_path}/resistance'][:]
        except KeyError as e:
            QMessageBox.critical(self, "Data Error", f"Missing dataset in {demo_path}:\n{e}")
            self.images = np.array([])
            self.resistance = np.array([])
            return
        
                # --- NEW: pull the actions matrix (shape: [N, 6]) -----------
        try:
            self.actions = self.hdf5_file[f'data/{demo_key}/actions'][:]
        except KeyError:
            QMessageBox.critical(self, "Data Error",
                                 f"Missing 'actions' dataset in {demo_path}")
            self.actions = np.empty((0, 6))


        # Resize the label *once* to the first frame’s native size
        if self.images.size:
            h_img, w_img = self.images[0].shape[:2]
            self.video_label.setFixedSize(w_img, h_img)

        # Validate the resistance data.
        if self.resistance.ndim != 1:
            QMessageBox.critical(
                self,
                "Data Error",
                f"Expected 'resistance' to be 1D, got shape {self.resistance.shape}"
            )
            self.resistance = np.array([])

        self.plot_resistance()
        self.plot_actions()

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
        if self.actions.size == 0:
            self.action_scene.addText("No actions data available")
            return

        counts  = np.count_nonzero(self.actions, axis=0)       # [6]
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


    # ------------------------------------------------------------------
    # 3. update_frame() — shrink oversized frames, never upscale
    # ------------------------------------------------------------------
    def update_frame(self):
        if self.images.size == 0:
            return

        if self.current_frame >= len(self.images):
            self.current_frame = 0

        try:
            img_array = self.images[self.current_frame]
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

        # Only shrink if the frame exceeds the label dimensions
        if (pixmap.width() > self.video_label.width() or
                pixmap.height() > self.video_label.height()):
            pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

        self.video_label.setPixmap(pixmap)
        self.current_frame += 1

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
    # data_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\HEK_dataset_v0_017.hdf5"
    data_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\experiments\Datasets\HEK_dataset_rotation_test.hdf5"
    viewer = DemoPlayer(data_path)
    viewer.show()
    sys.exit(app.exec_())
