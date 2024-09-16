from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QHBoxLayout, QFrame, QMessageBox, QShortcut
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPixmap
import os
import json

class ImageLabeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Labeler")
        self.setGeometry(50, 50, 1200, 1200)  # Assuming square images, resized the window accordingly
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout(self.central_widget)

        # Horizontal layout for image and controls
        self.top_layout = QHBoxLayout()

        # Set up the image label and frame (for square images)
        self.image_label = QLabel("No Image loaded yet!")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 600)  # Adjusted to 600x600 for square images
        self.image_label.setMaximumSize(600, 600)

        self.image_frame = QFrame()
        image_frame_layout = QVBoxLayout(self.image_frame)
        image_frame_layout.addWidget(self.image_label)
        self.image_frame.setFrameShape(QFrame.StyledPanel)
        self.image_frame.setFrameShadow(QFrame.Sunken)

        # Graphics scene and view for bounding boxes
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setMinimumSize(600, 600)

        # Add the image frame to the top layout
        self.top_layout.addWidget(self.view)
        self.main_layout.addLayout(self.top_layout)

        # Info pane layout for displaying messages to the user
        self.info_label = QLabel("No info yet!")
        self.info_label.setAlignment(Qt.AlignLeft)
        self.info_label.setFixedHeight(20)
        self.info_frame = QFrame()
        info_frame_layout = QHBoxLayout(self.info_frame)
        info_frame_layout.addWidget(self.info_label)
        self.info_frame.setFrameShape(QFrame.StyledPanel)
        self.info_frame.setFrameShadow(QFrame.Sunken)

        self.main_layout.addWidget(self.info_frame)

        # Add buttons for navigation and directory selection at the bottom
        self.buttons_layout = QHBoxLayout()

        self.load_button = QPushButton("Select Data Directory")
        self.load_button.clicked.connect(self.open_directory)
        
        self.prev_button = QPushButton("Previous Image")
        self.prev_button.clicked.connect(self.show_previous_image)
        
        self.next_button = QPushButton("Next Image")
        self.next_button.clicked.connect(self.show_next_image)

        self.buttons_layout.addWidget(self.prev_button)
        self.buttons_layout.addWidget(self.load_button)
        self.buttons_layout.addWidget(self.next_button)

        self.main_layout.addLayout(self.buttons_layout)

        # Set initial state
        self.image_paths = []
        self.labels_path = ''
        self.current_index = 0

                # Keyboard shortcuts
        self.shortcut_left = QShortcut(Qt.Key_Left, self)
        self.shortcut_left.activated.connect(self.show_previous_image)
        self.shortcut_right = QShortcut(Qt.Key_Right, self)
        self.shortcut_right.activated.connect(self.show_next_image)

    def open_directory(self):
        """Open a directory with 'images' and 'labels' folders."""
        directory = QFileDialog.getExistingDirectory(self, "Open Directory", "")
        if directory:
            images_dir = os.path.join(directory, 'P_DET_IMAGES')
            labels_dir = os.path.join(directory, 'P_DET_LABELS')
            
            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                self.load_images_from_directory(images_dir)
                self.labels_path = labels_dir
                self.info_label.setText(f"Loaded images and labels from {directory}.")
            else:
                QMessageBox.warning(self, "Directory Error", "Selected folder must contain 'images' and 'labels' subfolders.")

    def load_images_from_directory(self, directory):
        """Load images from the 'images' folder."""
        if not os.path.exists(directory):
            self.info_label.setText(f"Directory {directory} does not exist")
            return
        
        self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg','webp'))]
        if not self.image_paths:
            self.info_label.setText(f"No images found in {directory}")
            return
        
        self.current_index = 0
        self.display_image(self.image_paths[self.current_index])
        self.info_label.setText(f"Loaded {len(self.image_paths)} images.")

    def display_image(self, image_path):
        """Display the image on the QGraphicsView using QGraphicsScene."""
        if not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", f"Image file {image_path} does not exist.")
            return  # Exit if image path is invalid

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", "Unable to load image.")
        else:
            # Clear the scene to remove any previous content
            self.scene.clear()

            # Add the image to the QGraphicsScene
            pixmap_item = self.scene.addPixmap(pixmap)

            # Fit the image to the view
            self.view.fitInView(pixmap_item, Qt.KeepAspectRatio)
            self.info_label.setText(f"Displaying {os.path.basename(image_path)}")


    def load_bounding_boxes(self, image_path):
        """Load bounding boxes from the corresponding label file."""
        self.scene.clear()  # Clear previous bounding boxes
        label_file = os.path.join(self.labels_path, os.path.basename(image_path).replace('.jpg', '.json').replace('.png', '.json').replace('.jpeg', '.json').replace('.webp', '.json'))
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                bounding_boxes = json.load(f).get('bounding_boxes', [])
                for box in bounding_boxes:
                    rect_item = QGraphicsRectItem(QRectF(box['x'], box['y'], box['width'], box['height']))
                    rect_item.setFlag(QGraphicsRectItem.ItemIsMovable)
                    rect_item.setFlag(QGraphicsRectItem.ItemIsSelectable)
                    rect_item.setFlag(QGraphicsRectItem.ItemIsFocusable)
                    self.scene.addItem(rect_item)
            self.info_label.setText(f"Loaded labels for {os.path.basename(image_path)}")
        else:
            self.info_label.setText(f"No labels found for {os.path.basename(image_path)}")

    def save_bounding_boxes(self, image_path):
        """Save bounding boxes to the corresponding label file."""
        label_file = os.path.join(self.labels_path, os.path.basename(image_path).replace('.jpg', '.json').replace('.png', '.json').replace('.jpeg', '.json').replace('.webp', '.json'))
        bounding_boxes = []
        
        for item in self.scene.items():
            if isinstance(item, QGraphicsRectItem):
                bounding_boxes.append({
                    'x': item.rect().x(),
                    'y': item.rect().y(),
                    'width': item.rect().width(),
                    'height': item.rect().height()
                })
        
        with open(label_file, 'w') as f:
            json.dump({'bounding_boxes': bounding_boxes}, f)
        self.info_label.setText(f"Saved labels for {os.path.basename(image_path)}")

    def show_previous_image(self):
        """Show the previous image and load its bounding boxes."""
        if self.current_index > 0:
            self.save_bounding_boxes(self.image_paths[self.current_index])  # Save before switching
            self.current_index -= 1
            self.display_image(self.image_paths[self.current_index])

    def show_next_image(self):
        """Show the next image and load its bounding boxes."""
        if self.current_index < len(self.image_paths) - 1:
            self.save_bounding_boxes(self.image_paths[self.current_index])  # Save before switching
            self.current_index += 1
            self.display_image(self.image_paths[self.current_index])

if __name__ == "__main__":
    app = QApplication([])
    labeler = ImageLabeler()
    labeler.show()
    app.exec_()
