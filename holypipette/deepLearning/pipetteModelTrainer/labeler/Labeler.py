import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFileDialog, QListWidget, QInputDialog, QMessageBox, QShortcut)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QColor, QKeySequence, QFont
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem
from PIL import Image
import xml.etree.ElementTree as ET

class BoundingBox(QGraphicsRectItem):
    def __init__(self, rect, label):
        super().__init__(rect)
        self.label = label
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setPen(QPen(Qt.red, 2, Qt.SolidLine))

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(Qt.white)
        painter.drawText(self.rect().topLeft(), self.label)

class ImageViewer(QGraphicsView):
    boxDrawn = pyqtSignal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setDragMode(QGraphicsView.NoDrag)
        self.start_pos = None
        self.current_rect = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = self.mapToScene(event.pos())
            self.current_rect = QGraphicsRectItem()
            self.current_rect.setPen(QPen(Qt.green, 2, Qt.DashLine))
            self.scene.addItem(self.current_rect)

    def mouseMoveEvent(self, event):
        if self.start_pos:
            end_pos = self.mapToScene(event.pos())
            rect = QRectF(self.start_pos, end_pos).normalized()
            self.current_rect.setRect(rect)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_pos:
            end_pos = self.mapToScene(event.pos())
            rect = QRectF(self.start_pos, end_pos).normalized()
            self.scene.removeItem(self.current_rect)
            self.current_rect = None
            self.start_pos = None
            if rect.width() > 5 and rect.height() > 5:
                self.boxDrawn.emit(rect)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            factor = 1.25
        else:
            factor = 0.8
        self.scale(factor, factor)

class ImageLabeler(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_dir = ""
        self.label_dir = ""
        self.initUI()
        self.current_image_index = -1
        self.image_files = []
        self.current_image_path = ""
        self.original_image_size = (0, 0)
        
    def initUI(self):
        self.setWindowTitle('Image Labeler')
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        main_layout = QVBoxLayout()

        self.image_name_label = QLabel("No Image Loaded")
        self.image_name_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.image_name_label)

        content_layout = QHBoxLayout()

        self.viewer = ImageViewer()
        self.viewer.boxDrawn.connect(self.on_box_drawn)
        content_layout.addWidget(self.viewer, 7)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout()
        
        self.prev_button = QPushButton('Previous Image')
        self.prev_button.clicked.connect(self.prev_image)
        
        self.next_button = QPushButton('Next Image')
        self.next_button.clicked.connect(self.next_image)
        
        save_button = QPushButton('Save Labels')
        save_button.clicked.connect(self.save_labels)
        
        select_image_dir_button = QPushButton('Select Image Directory')
        select_image_dir_button.clicked.connect(self.select_image_directory)
        
        select_label_dir_button = QPushButton('Select Label Directory')
        select_label_dir_button.clicked.connect(self.select_label_directory)

        self.label_list = QListWidget()
        self.label_list.itemClicked.connect(self.highlight_bounding_box)
        
        sidebar_layout.addWidget(select_image_dir_button)
        sidebar_layout.addWidget(select_label_dir_button)
        sidebar_layout.addWidget(self.prev_button)
        sidebar_layout.addWidget(self.next_button)
        sidebar_layout.addWidget(save_button)
        sidebar_layout.addWidget(QLabel('Labels:'))
        sidebar_layout.addWidget(self.label_list)
        sidebar.setLayout(sidebar_layout)
        
        content_layout.addWidget(sidebar, 3)
        main_layout.addLayout(content_layout)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.setup_shortcuts()

    def setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_image)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_image)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_labels)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete_selected_box)

    def on_box_drawn(self, rect):
        label, ok = QInputDialog.getText(self, "Input Label", "Enter label for the bounding box:")
        if ok and label:
            box = BoundingBox(rect, label)
            self.viewer.scene.addItem(box)
            self.label_list.addItem(label)

    def select_image_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if directory:
            self.image_dir = directory
            self.load_image_directory()

    def select_label_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Label Directory")
        if directory:
            self.label_dir = directory

    def load_image_directory(self):
        if not os.path.isdir(self.image_dir):
            QMessageBox.critical(self, "Error", f"The specified image directory does not exist: {self.image_dir}")
            return

        self.image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if not self.image_files:
            QMessageBox.warning(self, "Warning", "No supported image files found in the specified directory.")
            return
        
        self.image_files.sort()
        self.current_image_index = 0
        self.load_image(os.path.join(self.image_dir, self.image_files[self.current_image_index]))

    def load_image(self, file_path):
        try:
            self.current_image_path = file_path
            image = Image.open(file_path)
            self.original_image_size = image.size
            image = image.convert("RGBA")
            data = image.tobytes("raw", "RGBA")
            qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)

            self.viewer.scene.clear()
            self.viewer.scene.addPixmap(pixmap)
            self.viewer.setSceneRect(QRectF(pixmap.rect()))
            self.viewer.fitInView(self.viewer.sceneRect(), Qt.KeepAspectRatio)

            self.image_name_label.setText(os.path.basename(file_path))
            self.load_labels()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(os.path.join(self.image_dir, self.image_files[self.current_image_index]))

    def next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image(os.path.join(self.image_dir, self.image_files[self.current_image_index]))

    def highlight_bounding_box(self, item):
        label = item.text()
        for box in self.viewer.scene.items():
            if isinstance(box, BoundingBox) and box.label == label:
                box.setPen(QPen(Qt.yellow, 3, Qt.SolidLine))
            elif isinstance(box, BoundingBox):
                box.setPen(QPen(Qt.red, 2, Qt.SolidLine))

    def delete_selected_box(self):
        selected_items = self.viewer.scene.selectedItems()
        for item in selected_items:
            if isinstance(item, BoundingBox):
                self.viewer.scene.removeItem(item)
                for i in range(self.label_list.count()):
                    if self.label_list.item(i).text() == item.label:
                        self.label_list.takeItem(i)
                        break

    def save_labels(self):
        if not self.current_image_path or not self.label_dir:
            QMessageBox.warning(self, "Warning", "No image loaded or label directory not set.")
            return

        xml_file = os.path.join(self.label_dir, os.path.splitext(os.path.basename(self.current_image_path))[0] + '.xml')
        
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = os.path.basename(self.current_image_path)
        
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(self.original_image_size[0])
        ET.SubElement(size, "height").text = str(self.original_image_size[1])
        ET.SubElement(size, "depth").text = "3"

        for item in self.viewer.scene.items():
            if isinstance(item, BoundingBox):
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = item.label
                bndbox = ET.SubElement(obj, "bndbox")
                rect = item.sceneBoundingRect()
                ET.SubElement(bndbox, "xmin").text = str(int(rect.left()))
                ET.SubElement(bndbox, "ymin").text = str(int(rect.top()))
                ET.SubElement(bndbox, "xmax").text = str(int(rect.right()))
                ET.SubElement(bndbox, "ymax").text = str(int(rect.bottom()))

        tree = ET.ElementTree(root)
        tree.write(xml_file)
        
        QMessageBox.information(self, "Info", f"Labels saved to {xml_file}")

    def load_labels(self):
        xml_file = os.path.join(self.label_dir, os.path.splitext(os.path.basename(self.current_image_path))[0] + '.xml')
        
        if not os.path.exists(xml_file):
            return

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.findall('object'):
                label = obj.find('name').text
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                rect = QRectF(QPointF(xmin, ymin), QPointF(xmax, ymax))
                box = BoundingBox(rect, label)
                self.viewer.scene.addItem(box)
                self.label_list.addItem(label)
        except ET.ParseError as e:
            QMessageBox.warning(self, "Warning", f"Error parsing XML file: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageLabeler()
    ex.show()
    sys.exit(app.exec_())
