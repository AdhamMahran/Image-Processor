import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QWidget, QComboBox,
                             QTabWidget, QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

class HistogramCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axes = plt.subplots(1, 2, figsize=(width, height), dpi=dpi)
        super(HistogramCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def plot_histogram_and_cdf(self, image, title):
        self.axes[0].clear()
        self.axes[1].clear()

        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            # Extract channel data
            channel = image[:, :, i].flatten()

            # Compute histogram manually
            hist = np.zeros(256, dtype=int)
            for pixel in channel:
                hist[pixel] += 1  # Count occurrences of each pixel value

            # Compute cumulative distribution function (CDF)
            cdf = np.zeros(256, dtype=float)
            cdf[0] = hist[0]
            for j in range(1, 256):
                cdf[j] = cdf[j - 1] + hist[j]  # Cumulative sum

            # Normalize CDF for visualization
            cdf_normalized = cdf * float(hist.max()) / cdf.max()

            # Plot histogram and CDF
            self.axes[0].plot(range(256), hist, color=color, label=f'{color.upper()} Channel')
            self.axes[1].plot(range(256), cdf_normalized, color=color, label=f'{color.upper()} CDF')

        self.axes[0].set_xlim([0, 256])
        self.axes[1].set_xlim([0, 256])
        self.axes[0].set_title(f'{title} - Histogram')
        self.axes[1].set_title(f'{title} - CDF')
        self.axes[0].set_xlabel('Pixel Value')
        self.axes[1].set_xlabel('Pixel Value')
        self.axes[0].set_ylabel('Frequency')
        self.axes[1].set_ylabel('Cumulative Frequency')
        self.axes[0].legend()
        self.axes[1].legend()
        self.fig.tight_layout()
        self.draw()
class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.processed_image = None
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        controls_layout = QHBoxLayout()

        self.load_button = QPushButton('Load Image')
        self.load_button.clicked.connect(self.load_image)
        controls_layout.addWidget(self.load_button)

        self.operation_combo = QComboBox()
        self.operation_combo.addItems(['Manual Normalization', 'Histogram Equalization'])
        controls_layout.addWidget(self.operation_combo)

        self.process_button = QPushButton('Process Image')
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        controls_layout.addWidget(self.process_button)

        self.save_button = QPushButton('Save Result')
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        controls_layout.addWidget(self.save_button)

        main_layout.addLayout(controls_layout)

        self.tab_widget = QTabWidget()
        images_widget = QWidget()
        images_layout = QHBoxLayout()

        self.original_display = QLabel('Original Image')
        self.original_display.setAlignment(Qt.AlignCenter)
        images_layout.addWidget(self.original_display)

        self.processed_display = QLabel('Processed Image')
        self.processed_display.setAlignment(Qt.AlignCenter)
        images_layout.addWidget(self.processed_display)

        images_widget.setLayout(images_layout)
        histogram_widget = QWidget()
        histogram_layout = QHBoxLayout()

        self.original_histogram = HistogramCanvas(self, width=10, height=4, dpi=100)
        histogram_layout.addWidget(self.original_histogram)
        self.processed_histogram = HistogramCanvas(self, width=10, height=4, dpi=100)
        histogram_layout.addWidget(self.processed_histogram)

        histogram_widget.setLayout(histogram_layout)
        self.tab_widget.addTab(images_widget, "Images")
        self.tab_widget.addTab(histogram_widget, "Histograms & CDFs")
        main_layout.addWidget(self.tab_widget)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.setWindowTitle('Image Processor')
        self.setGeometry(100, 100, 1200, 700)
        self.show()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp *.tif)')
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.original_image, self.original_display)
            self.process_button.setEnabled(True)
            self.original_histogram.plot_histogram_and_cdf(self.original_image, 'Original Image')

    def process_image(self):
        if self.original_image is None:
            return
        operation = self.operation_combo.currentText()
        if operation == 'Manual Normalization':
            self.processed_image = self.manual_normalization(self.original_image.copy())
        else:
            self.processed_image = self.histogram_equalization(self.original_image.copy())
        self.display_image(self.processed_image, self.processed_display)
        self.save_button.setEnabled(True)
        self.processed_histogram.plot_histogram_and_cdf(self.processed_image, 'Processed Image')

    def manual_normalization(self, image):
        image = image.astype(np.uint8)
        mean_val = np.mean(image)
        std_val = np.std(image)

        norm_image = (image - mean_val) / std_val  # Normalize using mean and std
        return norm_image.astype(np.uint8)

    def histogram_equalization(self, image):
        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)

    def display_image(self, image, display_label):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        display_label.setPixmap(pixmap)

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', 'Image Files (*.png *.jpg *.jpeg *.bmp *.tif)')
        if file_path:
            cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    sys.exit(app.exec_())
