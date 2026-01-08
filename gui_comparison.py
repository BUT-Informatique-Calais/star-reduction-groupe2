# SAE - Star reduction
#
# Groupe 2 :
# - AMEDRO Louis (Osiris-Sio)
# - HERBAUX Jules (Lirei159)
# - PACE--BOULNOIS Lysandre (NovaChocolat)

import cv2 as cv
import numpy as np
from astropy.io import fits
from skimage.metrics import structural_similarity, mean_squared_error
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QDialog)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

# --- Model ---
class ComparisonModel:
    def __init__(self):
        self.original_image = None
        self.processed_image = None

    def load_image(self, filepath):
        """Updates the image from a file (FITS or standard format)."""
        if filepath.lower().endswith(('.fits', '.fit')):
            try:
                hdul = fits.open(filepath)
                data = hdul[0].data
                hdul.close()
                if data.ndim == 3 and data.shape[0] == 3:
                    data = np.transpose(data, (1, 2, 0))
                
                # Normalization for display
                img = ((data - data.min()) / (data.max() - data.min()) * 255).astype("uint8")
                if len(img.shape) == 3:
                    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                return img
            except Exception as e:
                print(f"Error loading FITS: {e}")
                return None
        else:
            return cv.imread(filepath)

    def compute_difference(self):
        """Calculates difference and applies a heatmap (Red=Max Diff, Blue=Min Diff)."""
        if self.original_image is None or self.processed_image is None:
            return None
        
        h1, w1 = self.original_image.shape[:2]
        img2 = cv.resize(self.processed_image, (w1, h1)) # Match size
        
        # 1. Absolute difference
        diff = cv.absdiff(self.original_image, img2)
        
        # 2. Convert to grayscale (single intensity layer)
        if len(diff.shape) == 3:
            diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        else:
            diff_gray = diff

        # 3. Min-Max Normalization to use the full color spectrum
        # This allows visualizing relative differences like a weather map
        diff_norm = cv.normalize(diff_gray, None, 0, 255, cv.NORM_MINMAX)

        # 4. Apply JET Colormap (Blue=Cold/Low diff -> Red=Hot/High diff)
        heatmap = cv.applyColorMap(diff_norm, cv.COLORMAP_JET)
        
        return heatmap

    def calculate_metrics(self):
        """Calculates MSE and SSIM between the two images."""
        if self.original_image is None or self.processed_image is None:
            return None
        
        # Resize to match dimensions
        h1, w1 = self.original_image.shape[:2]
        img2 = cv.resize(self.processed_image, (w1, h1))
        
        # Convert to grayscale for metrics (Standard practice for SSIM)
        if len(self.original_image.shape) == 3:
            gray1 = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)
        else:
            gray1 = self.original_image
            
        if len(img2.shape) == 3:
            gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        else:
            gray2 = img2
            
        mse = mean_squared_error(gray1, gray2)
        ssim_val = structural_similarity(gray1, gray2, data_range=255)
        
        return mse, ssim_val

# --- View ---
class ComparisonView(QMainWindow):
    return_to_launcher = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.model = ComparisonModel()
        self.initUI()

    def initUI(self):
        """Initializes the user interface (Layouts and Widgets)."""
        self.setWindowTitle("Star Reduction - Mode Comparatif")
        self.resize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Controls
        controls_layout = QHBoxLayout()
        
        self.btn_load_orig = QPushButton("1. Charger Originale")
        self.btn_load_orig.clicked.connect(self.load_original)
        controls_layout.addWidget(self.btn_load_orig)

        self.btn_load_proc = QPushButton("2. Charger Érodée")
        self.btn_load_proc.clicked.connect(self.load_processed)
        controls_layout.addWidget(self.btn_load_proc)

        self.btn_diff = QPushButton("Voir les Différences")
        self.btn_diff.clicked.connect(self.show_difference)
        controls_layout.addWidget(self.btn_diff)
        
        main_layout.addLayout(controls_layout)

        # Images
        self.image_layout = QHBoxLayout()

        # Left Column (Original)
        self.col_orig = QVBoxLayout()
        self.label_orig = QLabel("Image Originale")
        self.label_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_orig.setStyleSheet("border: 1px solid #ccc; background-color: black; color: white;")
        self.col_orig.addWidget(self.label_orig)
        
        self.lbl_metrics_orig = QLabel("Reference")
        self.lbl_metrics_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.col_orig.addWidget(self.lbl_metrics_orig)
        
        self.image_layout.addLayout(self.col_orig)

        # Right Column (Processed)
        self.col_proc = QVBoxLayout()
        self.label_proc = QLabel("Image Érodée")
        self.label_proc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_proc.setStyleSheet("border: 1px solid #ccc; background-color: black; color: white;")
        self.col_proc.addWidget(self.label_proc)
        
        self.lbl_metrics_proc = QLabel("Similitude: -")
        self.lbl_metrics_proc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.col_proc.addWidget(self.lbl_metrics_proc)

        self.image_layout.addLayout(self.col_proc)

        main_layout.addLayout(self.image_layout)

        # Bottom
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        self.btn_back = QPushButton("Retour au menu")
        self.btn_back.setStyleSheet("padding: 8px 16px; background-color: #666; color: white;")
        self.btn_back.clicked.connect(self.on_back_click)
        bottom_layout.addWidget(self.btn_back)
        
        main_layout.addLayout(bottom_layout)

    def load_original(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Image Originale", "./examples", "Images (*.fits *.fit *.png *.jpg)")
        if filepath:
            img = self.model.load_image(filepath)
            if img is not None:
                self.model.original_image = img
                self.update_display()

    def load_processed(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Image Traitée", "./results", "Images (*.png *.jpg *.fits *.fit)")
        if filepath:
            img = self.model.load_image(filepath)
            if img is not None:
                self.model.processed_image = img
                self.update_display()

    def show_difference(self):
        diff = self.model.compute_difference()
        if diff is not None:
            # Add legend before display
            diff_with_legend = self.add_heatmap_legend(diff)
            self.show_popup_image(diff_with_legend, "Carte de Chaleur des Différences")
        else:
            QMessageBox.warning(self, "Attention", "Veuillez charger les deux images d'abord.")

    def add_heatmap_legend(self, heatmap):
        """Adds a vertical legend bar to the right of the image."""
        h, w = heatmap.shape[:2]
        legend_w = 50
        
        # 1. Create gradient (255 top -> 0 bottom) to match JET (Red -> Blue)
        gradient = np.linspace(255, 0, h).astype(np.uint8)
        # Horizontal repetition to create a bar
        gradient_bar = np.tile(gradient[:, np.newaxis], (1, legend_w))
        
        # 2. Apply colormap
        legend_color = cv.applyColorMap(gradient_bar, cv.COLORMAP_JET)
        
        # 3. Add text (White with black outline for readability)
        font = cv.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        white = (255, 255, 255)
        
        cv.putText(legend_color, "Max", (5, 20), font, scale, white, 1)
        cv.putText(legend_color, "Diff", (5, h // 2), font, scale, white, 1)
        cv.putText(legend_color, "Min", (5, h - 10), font, scale, white, 1)
        
        # 4. Assembly : Image | Spacer | Legend
        spacer = np.zeros((h, 10, 3), dtype=np.uint8) # 10px black strip
        combined = np.hstack((heatmap, spacer, legend_color))
        
        return combined

    def update_display(self):
        self.display_image(self.model.original_image, self.label_orig)
        self.display_image(self.model.processed_image, self.label_proc)
        
        # Calculate and display metrics
        metrics = self.model.calculate_metrics()
        if metrics:
            mse, ssim_val = metrics
            self.lbl_metrics_orig.setText("Originale (Réf)\nMSE: 0.00 | SSIM: 1.00")
            self.lbl_metrics_proc.setText(f"Modifiée\nMSE: {mse:.2f} | SSIM: {ssim_val:.4f}")
        else:
            self.lbl_metrics_orig.setText("Originale (Réf)")
            self.lbl_metrics_proc.setText("En attente de comparaison...")

    def display_image(self, img, label):
        if img is None:
            label.setText("Pas d'image")
            return

        # Resize for display efficiency
        h, w = img.shape[:2]
        max_h = 600
        if h > max_h:
            scale = max_h / h
            img = cv.resize(img, (int(w * scale), int(h * scale)))

        if len(img.shape) == 3:
            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            qimg = QImage(rgb.data, w, h, w * c, QImage.Format.Format_RGB888)
        else:
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)

        label.setPixmap(QPixmap.fromImage(qimg).scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def show_popup_image(self, img, title):
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(800, 600)
        layout = QVBoxLayout(dlg)
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Display logic similar to main labels
        h, w = img.shape[:2]
        max_h = 700
        if h > max_h:
            scale = max_h / h
            img = cv.resize(img, (int(w * scale), int(h * scale)))
        
        if len(img.shape) == 3:
            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            qimg = QImage(rgb.data, w, h, c * w, QImage.Format.Format_RGB888)
        else: 
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
            
        lbl.setPixmap(QPixmap.fromImage(qimg))
        layout.addWidget(lbl)
        dlg.show()

    def on_back_click(self):
        self.return_to_launcher.emit()
        self.close()
