import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFileDialog, QWidget, QComboBox,
                             QTabWidget, QSizePolicy, QSlider)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class HistogramCanvas(FigureCanvas):
    """
    A custom Matplotlib canvas for plotting the histogram and CDF 
    of either a color or grayscale image.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axes = plt.subplots(1, 2, figsize=(width, height), dpi=dpi)
        super(HistogramCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def plot_histogram_and_cdf(self, image, title):
        self.axes[0].clear()
        self.axes[1].clear()

        if len(image.shape) == 2:
            # Grayscale image
            channel = image.flatten()
            hist, bins = np.histogram(channel, bins=256, range=(0, 256))
            cdf = np.cumsum(hist)
            cdf_normalized = cdf * float(hist.max()) / cdf.max()

            self.axes[0].plot(range(256), hist, color='black', label='Grayscale')
            self.axes[1].plot(range(256), cdf_normalized, color='black', label='CDF')
            self.axes[0].legend()
            self.axes[1].legend()

        else:
            # Color image
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                channel = image[:, :, i].flatten()
                hist, bins = np.histogram(channel, bins=256, range=(0, 256))
                cdf = np.cumsum(hist)
                cdf_normalized = cdf * float(hist.max()) / cdf.max()

                self.axes[0].plot(range(256), hist, color=color, label=f'{color.upper()} Channel')
                self.axes[1].plot(range(256), cdf_normalized, color=color, label=f'{color.upper()} CDF')

            self.axes[0].legend()
            self.axes[1].legend()

        self.axes[0].set_xlim([0, 256])
        self.axes[1].set_xlim([0, 256])
        self.axes[0].set_title(f'{title} - Histogram')
        self.axes[1].set_title(f'{title} - CDF')
        self.axes[0].set_xlabel('Pixel Value')
        self.axes[1].set_xlabel('Pixel Value')
        self.axes[0].set_ylabel('Frequency')
        self.axes[1].set_ylabel('Cumulative Frequency')
        self.fig.tight_layout()
        self.draw()

class ImageProcessor(QMainWindow):
    """
    Main application window for loading images, performing various image
    processing operations, and displaying results with histograms and CDFs.
    """
    def __init__(self):
        super().__init__()
        self.original_image = None
        self.second_image = None      # For hybrid images
        self.processed_image = None
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        controls_layout = QHBoxLayout()

        # Button to load the primary image
        self.load_button = QPushButton('Load Image')
        self.load_button.clicked.connect(self.load_image)
        controls_layout.addWidget(self.load_button)

        # Button to load a second image (used for hybrid images)
        self.load_button_2 = QPushButton('Load Second Image (Hybrid)')
        self.load_button_2.clicked.connect(self.load_second_image)
        controls_layout.addWidget(self.load_button_2)

        # Combo box with operations (including the new ones)
        self.operation_combo = QComboBox()
        self.operation_combo.addItems([
        'Manual Normalization', 'Histogram Equalization', 'Global Thresholding',
        'Local Thresholding', 'Convert to Grayscale', 'Low Pass Filter',
        'High Pass Filter', 'Hybrid Image', 'Add Gaussian Noise', 'Add Salt and Pepper Noise', 'Add Uniform Noise',
        'Average Filter', 'Gaussian Filter', 'Median Filter', 'Sobel Edge Detection', 'Roberts Edge Detection',
        'Prewitt Edge Detection', 'Canny Edge Detection'])

        self.operation_combo.currentIndexChanged.connect(self.process_image)  # Connect signal to process_image
        controls_layout.addWidget(self.operation_combo)

        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(1)    # Minimum standard deviation
        self.noise_slider.setMaximum(100)  # Maximum standard deviation
        self.noise_slider.setValue(25)     # Default value
        self.noise_slider.setTickInterval(5)
        self.noise_slider.setTickPosition(QSlider.TicksBelow)
        self.noise_slider.setVisible(False)  # Initially hidden
        self.noise_slider.valueChanged.connect(self.process_image)

        controls_layout.addWidget(self.noise_slider)  # Add it to the UI layout

        self.filter_slider = QSlider(Qt.Horizontal)
        self.filter_slider.setMinimum(3)
        self.filter_slider.setMaximum(11)  # Must be an odd number
        self.filter_slider.setValue(3)  # Default kernel size
        self.filter_slider.setSingleStep(2)  # Only odd numbers
        self.filter_slider.setTickInterval(2)
        self.filter_slider.setTickPosition(QSlider.TicksBelow)
        self.filter_slider.setVisible(False)
        self.filter_slider.valueChanged.connect(self.process_image)  # Reapply filter when changed
        controls_layout.addWidget(self.filter_slider)  # Add it to the UI layout


        # Process button
        self.process_button = QPushButton('Process Image')
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)
        controls_layout.addWidget(self.process_button)

        # Save button
        self.save_button = QPushButton('Save Result')
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)
        controls_layout.addWidget(self.save_button)

        main_layout.addLayout(controls_layout)

        # Tabs for Images and Histograms
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

        # Sliders Tab (NEW)
        sliders_widget = QWidget()
        sliders_layout = QHBoxLayout()

        images_layout = QVBoxLayout()
        self.slider_image_display = QLabel('Loaded Image')
        self.slider_image_display.setAlignment(Qt.AlignCenter)
        self.slider_image_display.setFixedSize(200, 150)
        images_layout.addWidget(self.slider_image_display)

        self.slider_processed_display = QLabel('Processed Image')
        self.slider_processed_display.setAlignment(Qt.AlignCenter)
        self.slider_processed_display.setFixedSize(200, 150)
        images_layout.addWidget(self.slider_processed_display)

        sliders_layout.addLayout(images_layout)

        sliders_controls_layout = QVBoxLayout()

        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(1)
        self.noise_slider.setMaximum(100)
        self.noise_slider.setValue(25)
        self.noise_slider.setTickInterval(5)
        self.noise_slider.setTickPosition(QSlider.TicksBelow)
        self.noise_slider.valueChanged.connect(self.process_image)
        sliders_controls_layout.addWidget(QLabel("Noise Intensity"))
        sliders_controls_layout.addWidget(self.noise_slider)

        self.filter_slider = QSlider(Qt.Horizontal)
        self.filter_slider.setMinimum(3)
        self.filter_slider.setMaximum(11)
        self.filter_slider.setValue(3)
        self.filter_slider.setSingleStep(2)
        self.filter_slider.setTickInterval(2)
        self.filter_slider.setTickPosition(QSlider.TicksBelow)
        self.filter_slider.valueChanged.connect(self.process_image)
        sliders_controls_layout.addWidget(QLabel("Filter Kernel Size"))
        sliders_controls_layout.addWidget(self.filter_slider)

        sliders_layout.addLayout(sliders_controls_layout)

        sliders_widget.setLayout(sliders_layout)

        # Add Tabs
        self.tab_widget.addTab(images_widget, "Images")
        self.tab_widget.addTab(histogram_widget, "Histograms & CDFs")
        self.tab_widget.addTab(sliders_widget, "Sliders")
        main_layout.addWidget(self.tab_widget)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        self.setWindowTitle('Image Processor')
        self.setGeometry(100, 100, 1300, 700)
        self.show()
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Image', '',
            'Image Files (*.png *.jpg *.jpeg *.bmp *.tif)'
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            default_width, default_height = 800, 600
            self.original_image = cv2.resize(self.original_image, (default_width, default_height))
            self.display_image(self.original_image, self.original_display)
            self.display_image(self.original_image, self.slider_image_display)
            self.process_button.setEnabled(True)
            self.original_histogram.plot_histogram_and_cdf(self.original_image, 'Original Image')

    def load_second_image(self):
        """
        Load a second image, used for creating hybrid images 
        (combining low-pass of the first image and high-pass of the second).
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Second Image', '', 
            'Image Files (*.png *.jpg *.jpeg *.bmp *.tif)'
        )


        if file_path:
            self.second_image = cv2.imread(file_path)
            self.second_image = cv2.cvtColor(self.second_image, cv2.COLOR_BGR2RGB)

        default_width, default_height = 800, 600
        self.second_image = cv2.resize(self.second_image, (default_width, default_height))

    def process_image(self):
        if self.original_image is None:
            return

        operation = self.operation_combo.currentText()

        if operation == 'Manual Normalization':
            self.processed_image = self.manual_normalization(self.original_image.copy())

        elif operation == 'Histogram Equalization':
            self.processed_image = self.histogram_equalization(self.original_image.copy())

        elif operation == 'Global Thresholding':
            self.processed_image = self.global_thresholding(self.original_image.copy())

        elif operation == 'Local Thresholding':
            self.processed_image = self.local_thresholding(self.original_image.copy())

        elif operation == 'Convert to Grayscale':
            self.processed_image = self.convert_to_grayscale(self.original_image.copy())

        elif operation == 'Low Pass Filter':
            self.processed_image = self.low_pass_filter(self.original_image.copy())

        elif operation == 'High Pass Filter':
            self.processed_image = self.high_pass_filter(self.original_image.copy())

        elif operation == 'Hybrid Image':
            if self.second_image is None:
                # If no second image is loaded, just skip
                print("Please load a second image for Hybrid operation.")
                return
            self.processed_image = self.create_hybrid_image(self.original_image, self.second_image)

        elif operation == 'Add Gaussian Noise':
            self.noise_slider.setVisible(True)
            noise_std = self.noise_slider.value()  # Get the slider's value
            self.processed_image = self.add_gaussian_noise(self.original_image.copy(), std=noise_std)

        elif operation == 'Add Salt and Pepper Noise':
            self.noise_slider.setVisible(True)
            noise_amount = self.noise_slider.value() / 100  # Convert to proportion   
            self.processed_image = self.add_salt_and_pepper_noise(self.original_image.copy(), amount=noise_amount)

        elif operation == 'Add Uniform Noise':
             self.noise_slider.setVisible(True)
             noise_intensity = self.noise_slider.value()  # Get the slider's value
             self.processed_image = self.add_uniform_noise(self.original_image.copy(), intensity=noise_intensity)  

        elif operation == 'Average Filter':
            self.processed_image = self.average_filter(self.original_image.copy(), kernel_size=self.filter_slider.value())

        elif operation == 'Gaussian Filter':
            self.processed_image = self.gaussian_filter(self.original_image.copy(), kernel_size=self.filter_slider.value())

        elif operation == 'Median Filter':
            self.processed_image = self.median_filter(self.original_image.copy(), kernel_size=self.filter_slider.value())

        elif operation == 'Sobel Edge Detection':
            edge_x, edge_y, edge_magnitude = self.sobel_edge_detection(self.original_image.copy())
            self.processed_image = edge_magnitude  # Display combined magnitude

        elif operation == 'Roberts Edge Detection':
            edge_x, edge_y, edge_magnitude = self.roberts_edge_detection(self.original_image.copy())
            self.processed_image = edge_magnitude  # Display combined magnitude

        elif operation == 'Prewitt Edge Detection':
            edge_x, edge_y, edge_magnitude = self.prewitt_edge_detection(self.original_image.copy())
            self.processed_image = edge_magnitude  # Display combined magnitude

        elif operation == 'Canny Edge Detection':
            self.processed_image = self.canny_edge_detection(self.original_image.copy())
    
         
    



        if self.processed_image is not None:
                self.display_image(self.processed_image, self.processed_display)
                self.display_image(self.processed_image, self.slider_processed_display)
                self.save_button.setEnabled(True)
                self.processed_histogram.plot_histogram_and_cdf(self.processed_image, 'Processed Image')

    # --------------------- Image Processing Methods --------------------- #

    def manual_normalization(self, image):
        """
        Subtract mean and divide by std, then clip/convert to uint8.
        """
        image = image.astype(np.float32)
        mean_val = np.mean(image)
        std_val = np.std(image) + 1e-5  # avoid division by zero
        norm_image = (image - mean_val) / std_val

        # Scale to 0-255 range
        norm_image = cv2.normalize(norm_image, None, 0, 255, cv2.NORM_MINMAX)
        return norm_image.astype(np.uint8)

    def histogram_equalization(self, image):
        """
        Convert to YCrCb, equalize the Y channel, and convert back to RGB.
        """
        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2RGB)

    def global_thresholding(self, image):
        """
        Use Otsu's threshold on a grayscale version of the image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Otsu's binarization
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def local_thresholding(self, image):
        """
        Adaptive thresholding (local thresholding) on a grayscale image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Adaptive Gaussian threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        return thresh

    def convert_to_grayscale(self, image):
        """
        Convert the color image to a single-channel grayscale image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray  # Keep it single-channel
    
    def add_gaussian_noise(self, image, std=25):
        """
        Add Gaussian noise using OpenCV and NumPy.
        :param image: Input image (RGB or Grayscale)
        :param std: Standard deviation of the Gaussian noise
        :return: Noisy image
        """
        mean = 0
        noise = np.random.normal(mean, std, image.shape).astype(np.int16)  # Generate Gaussian noise
        
        # Convert image to int16 to prevent overflow/underflow
        noisy_image = cv2.add(image.astype(np.int16), noise)

        # Clip values to valid range [0,255] and convert back to uint8
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return noisy_image
    
    def add_salt_and_pepper_noise(self, image, amount=0.02):
            """
            Add Salt & Pepper noise to an image.
            :param image: Input image (RGB or Grayscale)
            :param amount: Proportion of pixels to be affected (0 to 1)
            :return: Noisy image
            """
            noisy_image = image.copy()
            num_pixels = int(amount * image.size)

            # Add salt noise (white pixels)
            coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape[:2]]
            noisy_image[coords[0], coords[1]] = 255 if len(image.shape) == 2 else [255, 255, 255]

            # Add pepper noise (black pixels)
            coords = [np.random.randint(0, i - 1, num_pixels) for i in image.shape[:2]]
            noisy_image[coords[0], coords[1]] = 0

            return noisy_image
    
    def add_uniform_noise(self, image, intensity=30):
        """
        Add uniform noise to an image.
        :param image: Input image (RGB or Grayscale)
        :param intensity: Maximum deviation from original pixel values
        :return: Noisy image
        """
        noise = np.random.uniform(-intensity, intensity, image.shape).astype(np.int16)

        # Convert image to int16 to prevent overflow/underflow
        noisy_image = cv2.add(image.astype(np.int16), noise)

        # Clip values to valid range [0,255] and convert back to uint8
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return noisy_image
    
    def average_filter(self, image, kernel_size=3):
        self.filter_slider.show()
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return self.convolve_2d(grayscale_image, kernel)

    def gaussian_filter(self, image, kernel_size=3, sigma=1.0):
        self.filter_slider.show()
        ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        kernel /= np.sum(kernel)  # Normalize
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return self.convolve_2d(grayscale_image, kernel)
    
    def median_filter(self, image, kernel_size=3):
        """
        Apply a median filter by replacing each pixel with the median of its neighborhood.
        """
        self.filter_slider.show()
        padded_image = np.pad(image, pad_width=kernel_size // 2, mode='edge')
        output = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                region = padded_image[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.median(region)
        
        return output.astype(np.uint8)
    
    def convolve_2d(self, image, kernel):
        """
        Manually applies a 2D convolution operation.
        - image: 2D NumPy array (grayscale image).
        - kernel: 2D filter/kernel to apply.
        """
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape

        # Compute padding size
        pad_h = kernel_height // 2
        pad_w = kernel_width // 2

        # Pad image to keep original size after filtering
        padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

        # Create output array
        output = np.zeros((image_height, image_width), dtype=np.float32)

        # Perform convolution
        for i in range(image_height):
            for j in range(image_width):
                # Extract the region of interest (ROI)
                region = padded_image[i:i + kernel_height, j:j + kernel_width]

                # Apply element-wise multiplication and sum the result
                output[i, j] = np.sum(region * kernel)

        # Normalize and return the output image
        return np.clip(output, 0, 255).astype(np.uint8)
    def sobel_edge_detection(self, image):
        """Apply Sobel edge detection in both X and Y directions."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        sobel_x = np.array([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]])
        
        sobel_y = np.array([[-1, -2, -1], 
                            [0, 0, 0], 
                            [1, 2, 1]])

        edge_x = self.convolve_2d(gray, sobel_x)
        edge_y = self.convolve_2d(gray, sobel_y)

        edge_magnitude = np.hypot(edge_x, edge_y)
        edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255

        return edge_x.astype(np.uint8), edge_y.astype(np.uint8), edge_magnitude.astype(np.uint8)
    
    def roberts_edge_detection(self, image):
        """Apply Roberts edge detection in both X and Y directions."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        roberts_x = np.array([[1, 0], 
                            [0, -1]])
        
        roberts_y = np.array([[0, 1], 
                            [-1, 0]])

        edge_x = self.convolve_2d(gray, roberts_x)
        edge_y = self.convolve_2d(gray, roberts_y)

        edge_magnitude = np.hypot(edge_x, edge_y)
        edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255

        return edge_x.astype(np.uint8), edge_y.astype(np.uint8), edge_magnitude.astype(np.uint8)
    
    def prewitt_edge_detection(self, image):
        """Apply Prewitt edge detection in both X and Y directions."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        prewitt_x = np.array([[-1, 0, 1], 
                            [-1, 0, 1], 
                            [-1, 0, 1]])
        
        prewitt_y = np.array([[-1, -1, -1], 
                            [0, 0, 0], 
                            [1, 1, 1]])

        edge_x = self.convolve_2d(gray, prewitt_x)
        edge_y = self.convolve_2d(gray, prewitt_y)

        edge_magnitude = np.hypot(edge_x, edge_y)
        edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255

        return edge_x.astype(np.uint8), edge_y.astype(np.uint8), edge_magnitude.astype(np.uint8)
    
    def canny_edge_detection(self, image, low_threshold=50, high_threshold=150):
        """Apply Canny edge detection using OpenCV."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges



    # --------------------- Frequency-Domain Filters --------------------- #

    def low_pass_filter(self, image, radius=30):
        """
        Apply a low-pass filter by masking out high frequencies in the DFT.
        Operates on a grayscale version of the image for simplicity.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        # Create a mask with 1s in the center (low freq) and 0s elsewhere
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 1

        # Apply mask and inverse DFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Normalize to 0-255
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        return img_back.astype(np.uint8)

    def high_pass_filter(self, image, radius=30):
        """
        Apply a high-pass filter by masking out low frequencies in the DFT.
        Operates on a grayscale version of the image for simplicity.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        # Create a mask with 0s in the center (low freq) and 1s elsewhere
        mask = np.ones((rows, cols, 2), np.uint8)
        mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0

        # Apply mask and inverse DFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Normalize to 0-255
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        return img_back.astype(np.uint8)

    # --------------------- Hybrid Image --------------------- #

    def create_hybrid_image(self, img1, img2, radius=30):
        """
        Create a hybrid image by combining the low-frequency content of img1
        and the high-frequency content of img2.
        Both images are converted to grayscale for simplicity.
        """
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

        # 1) Low-pass filter on gray1
        dft1 = cv2.dft(np.float32(gray1), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift1 = np.fft.fftshift(dft1)
        rows, cols = gray1.shape
        crow, ccol = rows // 2, cols // 2

        mask_low = np.zeros((rows, cols, 2), np.uint8)
        mask_low[crow - radius:crow + radius, ccol - radius:ccol + radius] = 1

        fshift1 = dft_shift1 * mask_low
        f_ishift1 = np.fft.ifftshift(fshift1)
        img_back1 = cv2.idft(f_ishift1)
        low_pass = cv2.magnitude(img_back1[:, :, 0], img_back1[:, :, 1])

        # 2) High-pass filter on gray2
        dft2 = cv2.dft(np.float32(gray2), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift2 = np.fft.fftshift(dft2)

        mask_high = np.ones((rows, cols, 2), np.uint8)
        mask_high[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0

        fshift2 = dft_shift2 * mask_high
        f_ishift2 = np.fft.ifftshift(fshift2)
        img_back2 = cv2.idft(f_ishift2)
        high_pass = cv2.magnitude(img_back2[:, :, 0], img_back2[:, :, 1])

        # Combine low-pass and high-pass
        hybrid = low_pass + high_pass
        cv2.normalize(hybrid, hybrid, 0, 255, cv2.NORM_MINMAX)

        return hybrid.astype(np.uint8)

    # --------------------- Display & Save --------------------- #


    def display_image(self, image, display_label):
        """
        Convert NumPy array to QImage/QPixmap and display in QLabel.
        Handles both grayscale (2D) and color (3D) images.
        """
        image = cv2.resize(image, (200, 150))
        if len(image.shape) == 2:
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Indexed8)
        else:
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        display_label.setPixmap(pixmap)
        display_label.setFixedSize(200, 150)
    def save_image(self):
        """
        Save the processed image to disk. If the image is grayscale (2D), 
        save directly; if color (3D), convert back to BGR before saving.
        """
        if self.processed_image is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Image', '', 
            'Image Files (*.png *.jpg *.jpeg *.bmp *.tif)'
        )
        if file_path:
            if len(self.processed_image.shape) == 2:
                # Grayscale
                cv2.imwrite(file_path, self.processed_image)
            else:
                # Color (convert back to BGR)
                cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    sys.exit(app.exec_())
