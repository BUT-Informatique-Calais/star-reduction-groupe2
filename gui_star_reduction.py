# SAE - Star reduction
#
# Groupe 2 :
# - AMEDRO Louis (Osiris-Sio)
# - HERBAUX Jules (Lirei159)
# - PACE--BOULNOIS Lysandre (NovaChocolat)

import sys
import cv2 as cv
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

# --- Model ---
class StarModel:
    def __init__(self):
        self.original_image = None
        self.gray_image = None

    def load_fits_data(self, filepath):
        try:
            # Open and read the FITS file
            hdul = fits.open(filepath)
            # Access the data from the primary HDU
            data = hdul[0].data
            hdul.close()

            # Handle both monochrome and color images
            if data.ndim == 3:
                # Color image - need to transpose to (height, width, channels)
                if data.shape[0] == 3:
                    data = np.transpose(data, (1, 2, 0))
                
                # Global normalization to [0, 255] for OpenCV
                self.original_image = ((data - data.min()) / (data.max() - data.min()) * 255).astype("uint8")
                # Conversion RGB to BGR for OpenCV
                self.original_image = cv.cvtColor(self.original_image, cv.COLOR_RGB2BGR)
            else:
                # Convert to uint8 for OpenCV
                self.original_image = ((data - data.min()) / (data.max() - data.min()) * 255).astype("uint8")

            # Keep a grayscale version for mask calculation
            if len(self.original_image.shape) == 3:
                self.gray_image = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)
            else:
                self.gray_image = self.original_image
                
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)

    def process_image(self, params):
        if self.original_image is None:
            return None

        try:
            k_erosion = params['erosion_kernel']
            iter_erosion = params['erosion_iter']
            block_size = params['thresh_block']
            c_val = params['thresh_c']
            k_blur = params['blur_kernel']

            # Define a kernel for erosion
            kernel = np.ones((k_erosion, k_erosion), np.uint8)
            # Perform erosion
            eroded_image = cv.erode(self.original_image, kernel, iterations=iter_erosion)

            ###### Phase 2 :

            ### Step A: Create star mask
            mask = cv.adaptiveThreshold(
                self.gray_image, 
                255, 
                cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv.THRESH_BINARY, 
                block_size, 
                c_val # Keep only pixels significantly brighter than the average
            )

            ### Step B: Localized reduction
            # Blurred mask created with Gaussian kernel
            mask_blurred = cv.GaussianBlur(mask, (k_blur, k_blur), 0)

            # Use float32 to avoid image depth errors
            M = mask_blurred.astype(np.float32) / 255.0
            if len(self.original_image.shape) == 3:
                M = np.stack([M, M, M], axis=2)

            # Explicit conversion to float32 for calculation
            Ioriginal = self.original_image.astype(np.float32)
            Ierode = eroded_image.astype(np.float32)

            # Calculation of the final image
            final_image_float = (M * Ierode) + ((1.0 - M) * Ioriginal)
            # Conversion back to uint8 BEFORE saving to avoid Warnings
            final_image = np.clip(final_image_float, 0, 255).astype(np.uint8)

            return final_image
        except Exception as e:
            print(f"Error in processing: {e}")
            return self.original_image

# --- View ---
class StarView(QMainWindow):
    # Signal emitted when any parameter changes
    # Carries a dictionary of all current parameter values
    parameters_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Star Reduction - MVC Pattern")
        self.resize(1200, 800)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Image display area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.main_layout.addWidget(self.image_label, stretch=2)

        # Controls area
        self.controls_panel = QGroupBox("Paramètres")
        self.controls_layout = QVBoxLayout(self.controls_panel)
        self.main_layout.addWidget(self.controls_panel, stretch=1)

        self.sliders = {}
        self.labels = {}

        self.create_controls()

    # Create sliders and labels for parameters
    def create_controls(self):
        self.add_control("Taille du noyau d'érosion (impair)", 3, 51, 3, 2, "erosion_kernel")
        self.add_control("Itérations d'érosion", 1, 30, 4, 1, "erosion_iter")
        self.add_control("Taille de bloc seuil (impair)", 3, 251, 21, 2, "thresh_block")
        self.add_control("Constante seuil (C)", -100, 100, -10, 1, "thresh_c")
        self.add_control("Flou du masque (impair)", 1, 151, 15, 2, "blur_kernel")
        
        self.controls_layout.addStretch()

    # Add a single control (label + slider)
    def add_control(self, label_text, min_val, max_val, default_val, step, key):
        layout = QVBoxLayout()
        label = QLabel(f"{label_text}: {default_val}")
        
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)
        slider.setSingleStep(step)
        
        self.sliders[key] = slider
        self.labels[key] = (label, label_text) # Store label widget and base text

        # Connect signal
        slider.valueChanged.connect(lambda v: self.on_slider_change(v, key))
        
        layout.addWidget(label)
        layout.addWidget(slider)
        self.controls_layout.addLayout(layout)

    def on_slider_change(self, value, key):
        # Enforce odd numbers for specific kernels
        if key in ["erosion_kernel", "thresh_block", "blur_kernel"]:
            if value % 2 == 0:
                value += 1
                self.sliders[key].blockSignals(True) # Prevent recursive call
                self.sliders[key].setValue(value)
                self.sliders[key].blockSignals(False)

        # Update label text
        label_widget, base_text = self.labels[key]
        label_widget.setText(f"{base_text}: {value}")

        # Emit parameters to controller
        self.emit_parameters()

    def emit_parameters(self):
        params = {key: slider.value() for key, slider in self.sliders.items()}
        self.parameters_changed.emit(params)

    def display_image(self, img):
        if img is None: 
            return

        if len(img.shape) == 3:
            # OpenCV is BGR, Qt expects RGB
            rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            h, w = img.shape
            bytes_per_line = w
            qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        ))

# --- Controller ---
class StarController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

        # Connect View signals
        self.view.parameters_changed.connect(self.update_model)

        # Initial load
        self.model.load_fits_data("./examples/m31_star.fits")
        
        # Trigger initial update
        self.view.emit_parameters()

    def update_model(self, params):
        # Process image with new params
        result_image = self.model.process_image(params)
        # Update View
        self.view.display_image(result_image)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    model = StarModel()
    view = StarView()
    controller = StarController(model, view)
    
    view.show()
    sys.exit(app.exec())
