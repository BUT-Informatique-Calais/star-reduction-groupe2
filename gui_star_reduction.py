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
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QGroupBox, QFileDialog, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
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
            # Extract Phase 3 parameters
            block_size = params.get('thresh_block', 31)
            c_val = params.get('thresh_c', -2)
            
            k_opening = params.get('opening_kernel', 3)
            iter_dilate = params.get('dilate_iter', 3)
            
            inpaint_radius = params.get('inpaint_radius', 5)
            # Alpha is a percentage (0-100) in the interface, convert to 0.0-1.0
            alpha = params.get('reduction_alpha', 60) / 100.0
            
            k_blur = params.get('blur_kernel', 15)

            # 1. Star Mask Creation (Detection)
            mask = cv.adaptiveThreshold(
                self.gray_image, 
                255, 
                cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv.THRESH_BINARY, 
                block_size, 
                c_val
            )

            # 2. Mask Cleaning (Morphological Opening)
            kernel_m = np.ones((k_opening, k_opening), np.uint8)
            mask_cleaned = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_m)

            # 3. Mask Expansion (Dilation to cover halos)
            mask_dilated = cv.dilate(mask_cleaned, kernel_m, iterations=iter_dilate)

            # 4. Inpainting (Smart reconstruction of masked areas using original image)
            # Note: inpaint expects an 8-bit image
            inpainted_image = cv.inpaint(self.original_image, mask_dilated, inpaint_radius, cv.INPAINT_TELEA)

            # 5. Final Fusion (Alpha Blending)
            # Soften mask edges for smooth transition
            mask_blurred = cv.GaussianBlur(mask_dilated, (k_blur, k_blur), 0)

            # Convert mask to float (0.0 - 1.0)
            M = mask_blurred.astype(np.float32) / 255.0
            if len(self.original_image.shape) == 3:
                M = np.stack([M] * 3, axis=-1)

            Ioriginal = self.original_image.astype(np.float32)
            Iinpainted = inpainted_image.astype(np.float32)

            # Phase 3 Fusion Formula:
            # Weighted mix between original image and "repaired" (inpainted) image
            final_image_float = (M * alpha * Iinpainted) + (1.0 - (M * alpha)) * Ioriginal
            
            # Final conversion to uint8
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
    # Signal emitted when returning to launcher
    return_to_launcher = pyqtSignal()

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

        # Timer for debounce (delay processing while sliding)
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(200) # 200ms delay
        self.update_timer.timeout.connect(self.emit_parameters)

        self.create_controls()

    # Create sliders and labels for parameters
    def create_controls(self):
        # 1. Detection (Mask)
        self.add_control("Masque: Seuil Bloc (impair)", 3, 251, 31, 2, "thresh_block")
        self.add_control("Masque: Constante C", -50, 50, -2, 1, "thresh_c")
        
        # 2. Cleaning & Dilation
        self.add_control("Masque: Nettoyage (Ouverture)", 3, 21, 3, 2, "opening_kernel")
        self.add_control("Masque: Dilatation (Halos)", 0, 20, 3, 1, "dilate_iter")

        # 3. Inpainting
        self.add_control("Inpainting: Rayon", 1, 20, 5, 1, "inpaint_radius")
        
        # 4. Fusion
        self.add_control("Fusion: Intensité Reduction (%)", 0, 100, 60, 5, "reduction_alpha")
        self.add_control("Fusion: Flou Transition", 3, 101, 15, 2, "blur_kernel")
        
        self.controls_layout.addStretch()

        # Back button
        self.btn_back = QPushButton("Retour au menu")
        self.btn_back.setStyleSheet("background-color: #666; color: white; padding: 5px;")
        self.btn_back.clicked.connect(self.on_back_click)
        self.controls_layout.addWidget(self.btn_back)

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

    def on_back_click(self):
        self.return_to_launcher.emit()
        self.close()

    def on_slider_change(self, value, key):
        # Enforce odd numbers for specific kernels
        if key in ["thresh_block", "blur_kernel", "opening_kernel"]:
            if value % 2 == 0:
                value += 1
                self.sliders[key].blockSignals(True) # Prevent recursive call
                self.sliders[key].setValue(value)
                self.sliders[key].blockSignals(False)

        # Update label text
        label_widget, base_text = self.labels[key]
        label_widget.setText(f"{base_text}: {value}")

        # Restart processing timer (Debounce)
        # Only process when user stops moving slider for 200ms
        self.update_timer.start()

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
        self.load_image()

    def load_image(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self.view,
            "Ouvrir une image FITS", 
            "./examples", 
            "FITS Files (*.fits *.fit)"
        )

        if filepath:
            self.model.load_fits_data(filepath)
            # Trigger initial update
            self.view.emit_parameters()
        else:
            # User canceled, close app
            sys.exit(0)


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