from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
    QFileDialog, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QHBoxLayout,
    QFrame, QMessageBox, QShortcut
)
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPixmap, QPen, QColor, QKeySequence
import os
import json

class ResizableRectItem(QGraphicsRectItem):
    """Custom QGraphicsRectItem that can be resized."""
    def __init__(self, rect, parent=None):
        super().__init__(rect, parent)
        self.setFlags(
            QGraphicsRectItem.ItemIsSelectable |
            QGraphicsRectItem.ItemIsMovable |
            QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.handle_size = 8.0
        self.handles = {}
        self._current_handle = None
        self._resizing = False
        self.update_handles()

    def update_handles(self):
        """Update the positions of the resize handles."""
        s = self.handle_size
        rect = self.rect()
        self.handles = {
            'top_left': QRectF(rect.topLeft().x() - s/2, rect.topLeft().y() - s/2, s, s),
            'top_right': QRectF(rect.topRight().x() - s/2, rect.topRight().y() - s/2, s, s),
            'bottom_left': QRectF(rect.bottomLeft().x() - s/2, rect.bottomLeft().y() - s/2, s, s),
            'bottom_right': QRectF(rect.bottomRight().x() - s/2, rect.bottomRight().y() - s/2, s, s),
        }

    def paint(self, painter, option, widget=None):
        """Paint the rectangle and its resize handles."""
        super().paint(painter, option, widget)
        if self.isSelected():
            pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.rect())
            # Draw handles
            painter.setBrush(QColor(255, 0, 0))
            for handle in self.handles.values():
                painter.drawRect(handle)

    def hoverMoveEvent(self, event):
        """Change cursor when hovering over resize handles."""
        cursor = Qt.ArrowCursor
        for key, handle in self.handles.items():
            if handle.contains(event.pos()):
                if 'left' in key:
                    if 'top' in key or 'bottom' in key:
                        cursor = Qt.SizeFDiagCursor
                elif 'right' in key:
                    if 'top' in key or 'bottom' in key:
                        cursor = Qt.SizeBDiagCursor
        self.setCursor(cursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        """Determine if a resize handle is pressed."""
        for key, handle in self.handles.items():
            if handle.contains(event.pos()):
                self._current_handle = key
                self._resizing = True
                break
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Resize the rectangle based on the handle being dragged."""
        if self._resizing and self._current_handle:
            rect = self.rect()
            delta = event.pos() - event.lastPos()
            if 'left' in self._current_handle:
                rect.setLeft(rect.left() + delta.x())
            if 'right' in self._current_handle:
                rect.setRight(rect.right() + delta.x())
            if 'top' in self._current_handle:
                rect.setTop(rect.top() + delta.y())
            if 'bottom' in self._current_handle:
                rect.setBottom(rect.bottom() + delta.y())
            self.setRect(rect)
            self.update_handles()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Reset resizing flags."""
        self._resizing = False
        self._current_handle = None
        super().mouseReleaseEvent(event)

class CustomGraphicsScene(QGraphicsScene):
    """Custom QGraphicsScene to handle drawing of bounding boxes."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.start_point = QPointF()
        self.current_rect_item = None

    def mousePressEvent(self, event):
        """Start drawing a new rectangle."""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.scenePos()
            self.current_rect_item = ResizableRectItem(QRectF(self.start_point, self.start_point))
            pen = QPen(QColor(255, 0, 0), 2)
            self.current_rect_item.setPen(pen)
            self.addItem(self.current_rect_item)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Update the size of the rectangle being drawn."""
        if self.drawing and self.current_rect_item:
            rect = QRectF(self.start_point, event.scenePos()).normalized()
            self.current_rect_item.setRect(rect)
            self.current_rect_item.update_handles()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Finish drawing the rectangle."""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.current_rect_item:
                rect = self.current_rect_item.rect()
                if rect.width() < 10 or rect.height() < 10:
                    self.removeItem(self.current_rect_item)
            self.current_rect_item = None
        super().mouseReleaseEvent(event)

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
        self.scene = CustomGraphicsScene(self)
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

        # Add the new buttons: Delete Selected Box and Delete All Boxes
        self.delete_selected_button = QPushButton("Delete Selected Box")
        self.delete_selected_button.clicked.connect(self.delete_selected_box)
        
        self.delete_all_button = QPushButton("Delete All Boxes")
        self.delete_all_button.clicked.connect(self.delete_all_boxes)

        self.buttons_layout.addWidget(self.prev_button)
        self.buttons_layout.addWidget(self.load_button)
        self.buttons_layout.addWidget(self.next_button)
        self.buttons_layout.addWidget(self.delete_selected_button)
        self.buttons_layout.addWidget(self.delete_all_button)

        self.main_layout.addLayout(self.buttons_layout)

        # Set initial state
        self.image_paths = []
        self.labels_path = ''
        self.current_index = 0

        # Keyboard shortcuts
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(self.show_previous_image)
        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(self.show_next_image)

    def open_directory(self):
        """Open a directory with 'P_DET_IMAGES' and 'P_DET_LABELS' folders."""
        directory = QFileDialog.getExistingDirectory(self, "Open Directory", "")
        if directory:
            images_dir = os.path.join(directory, 'P_DET_IMAGES')
            labels_dir = os.path.join(directory, 'P_DET_LABELS')
            
            if os.path.exists(images_dir) and os.path.exists(labels_dir):
                self.load_images_from_directory(images_dir)
                self.labels_path = labels_dir
                self.info_label.setText(f"Loaded images and labels from {directory}.")
                self.display_image(self.image_paths[self.current_index])
                self.load_bounding_boxes(self.image_paths[self.current_index])
            else:
                QMessageBox.warning(self, "Directory Error", "Selected folder must contain 'P_DET_IMAGES' and 'P_DET_LABELS' subfolders.")

    def load_images_from_directory(self, directory):
        """Load images from the 'P_DET_IMAGES' folder."""
        if not os.path.exists(directory):
            self.info_label.setText(f"Directory {directory} does not exist")
            return
        
        self.image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg','webp'))]
        if not self.image_paths:
            self.info_label.setText(f"No images found in {directory}")
            return
        
        self.current_index = 0
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
            # Clear the scene to remove any previous content except the pixmap
            self.scene.clear()

            # Add the image to the QGraphicsScene
            pixmap_item = self.scene.addPixmap(pixmap)
            pixmap_item.setZValue(-1)  # Ensure the image is at the back

            # Fit the image to the view
            self.view.fitInView(pixmap_item, Qt.KeepAspectRatio)
            self.info_label.setText(f"Displaying {os.path.basename(image_path)}")
            self.load_bounding_boxes(image_path)

    def load_bounding_boxes(self, image_path):
        """Load bounding boxes from the corresponding label file."""
        label_file = os.path.join(
            self.labels_path,
            os.path.basename(image_path).rsplit('.', 1)[0] + '.json'
        )
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                bounding_boxes = json.load(f).get('bounding_boxes', [])
                for box in bounding_boxes:
                    rect_item = ResizableRectItem(QRectF(box['x'], box['y'], box['width'], box['height']))
                    rect_item.setPen(QPen(QColor(255, 0, 0), 2))
                    self.scene.addItem(rect_item)
            self.info_label.setText(f"Loaded labels for {os.path.basename(image_path)}")
        else:
            self.info_label.setText(f"No labels found for {os.path.basename(image_path)}")

    def save_bounding_boxes(self, image_path):
        """Save bounding boxes to the corresponding label file."""
        label_file = os.path.join(
            self.labels_path,
            os.path.basename(image_path).rsplit('.', 1)[0] + '.json'
        )
        bounding_boxes = []
        
        for item in self.scene.items():
            if isinstance(item, ResizableRectItem):
                rect = item.rect()
                bounding_boxes.append({
                    'x': rect.x(),
                    'y': rect.y(),
                    'width': rect.width(),
                    'height': rect.height()
                })
        
        with open(label_file, 'w') as f:
            json.dump({'bounding_boxes': bounding_boxes}, f, indent=4)
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

    def delete_selected_box(self):
        """Delete the currently selected bounding box."""
        selected_items = self.scene.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "No Selection", "No box is selected to delete.")
            return
        for item in selected_items:
            if isinstance(item, ResizableRectItem):
                self.scene.removeItem(item)
        self.info_label.setText("Selected box(es) deleted.")

    def delete_all_boxes(self):
        """Delete all bounding boxes in the current image."""
        items_to_remove = [item for item in self.scene.items() if isinstance(item, ResizableRectItem)]
        if not items_to_remove:
            QMessageBox.information(self, "No Boxes", "There are no boxes to delete.")
            return
        for item in items_to_remove:
            self.scene.removeItem(item)
        self.info_label.setText("All boxes deleted.")

if __name__ == "__main__":
    app = QApplication([])
    labeler = ImageLabeler()
    labeler.show()
    app.exec_()
