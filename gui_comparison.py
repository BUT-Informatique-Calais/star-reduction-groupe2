# SAE - Star reduction
#
# Groupe 2 :
# - AMEDRO Louis (Osiris-Sio)
# - HERBAUX Jules (Lirei159)
# - PACE--BOULNOIS Lysandre (NovaChocolat)

import cv2 as cv
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QComboBox, QMessageBox, QSlider, QDialog)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

# --- Model ---
class ComparisonModel:
    def __init__(self):
        self.original_image = None
        self.processed_image = None

    def load_image(self, filepath):
        """Met à jour l'image depuis un fichier (FITS ou format classique)."""
        if filepath.lower().endswith(('.fits', '.fit')):
            try:
                hdul = fits.open(filepath)
                data = hdul[0].data
                hdul.close()
                if data.ndim == 3 and data.shape[0] == 3:
                    data = np.transpose(data, (1, 2, 0))
                
                # Normalisation pour affichage
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
        """Calcule la différence absolue entre l'image originale et le résultat."""
        if self.original_image is None or self.processed_image is None:
            return None
        
        h1, w1 = self.original_image.shape[:2]
        img2 = cv.resize(self.processed_image, (w1, h1)) # Match size
        
        diff = cv.absdiff(self.original_image, img2)
        return cv.multiply(diff, 3) # Amplify differences

# --- View ---
class ComparisonView(QMainWindow):
    return_to_launcher = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.model = ComparisonModel()
        self.initUI()

    def initUI(self):
        """Initialise l'interface utilisateur (Layouts et Widgets)."""
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

        
        self.label_orig = QLabel("Image Originale")
        self.label_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_orig.setStyleSheet("border: 1px solid #ccc; background-color: black; color: white;")
        self.image_layout.addWidget(self.label_orig)

        self.label_proc = QLabel("Image Érodée")
        self.label_proc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_proc.setStyleSheet("border: 1px solid #ccc; background-color: black; color: white;")
        self.image_layout.addWidget(self.label_proc)

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
            self.show_popup_image(diff, "Carte des différences (Amplifiée x3)")
        else:
            QMessageBox.warning(self, "Attention", "Veuillez charger les deux images d'abord.")

    def update_display(self):
        self.display_image(self.model.original_image, self.label_orig)
        self.display_image(self.model.processed_image, self.label_proc)

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
