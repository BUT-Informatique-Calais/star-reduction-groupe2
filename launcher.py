import sys
import os
import cv2 as cv
import numpy as np
from astropy.io import fits
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QSpacerItem,
    QSizePolicy,
    QFileDialog,
)
from PyQt6.QtCore import Qt
from gui_star_reduction import StarModel, StarView, StarController
from gui_comparison import ComparisonView


class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.realtime_window = None

    def initUI(self):
        self.setWindowTitle("Star Reduction - Accueil")
        self.resize(400, 300)

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)

        # Title
        title = QLabel("Star Reduction")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Groupe 2")
        subtitle.setStyleSheet("font-size: 14px; color: #666;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacerItem(
            QSpacerItem(
                20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
            )
        )

        # Buttons
        self.btn_realtime = QPushButton("Mode Temps Réel")
        self.btn_realtime.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_realtime.clicked.connect(self.launch_realtime)
        layout.addWidget(self.btn_realtime)

        self.btn_comparison = QPushButton("Mode Comparaison")
        self.btn_comparison.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_comparison.clicked.connect(self.launch_comparison)
        layout.addWidget(self.btn_comparison)

        self.btn_batch = QPushButton("Générer Images (Batch)")
        self.btn_batch.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_batch.clicked.connect(self.process_batch)
        layout.addWidget(self.btn_batch)

        layout.addSpacerItem(
            QSpacerItem(
                20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
            )
        )

        self.setLayout(layout)

    def launch_realtime(self):
        # Create MVC components
        self.model = StarModel()
        self.view = StarView()
        self.controller = StarController(self.model, self.view)

        # Connect return signal
        self.view.return_to_launcher.connect(self.show)

        # Show main window and close launcher
        self.view.show()
        self.close()

    def launch_comparison(self):
        self.comp_view = ComparisonView()
        self.comp_view.return_to_launcher.connect(self.show)
        self.comp_view.show()
        self.close()

    def process_batch(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner une image FITS",
            "./examples",
            "FITS Files (*.fits *.fit)",
        )
        if not filepath:
            return

        try:
            # Constants from prototype_phase3.py
            IMAGE_EROSION_SIZE = 3
            IMAGE_EROSION_ITER = 1
            MASK_BLOCK = 31
            MASK_C = -2
            OPENING_KERNEL_SIZE = 3
            MASK_DILATE_ITER = 3
            INPAINT_RADIUS = 5
            REDUCTION_ALPHA = 0.6
            BLUR_SIZE = 15

            if not os.path.exists("./results"):
                os.makedirs("./results")

            # Load
            hdul = fits.open(filepath)
            data = hdul[0].data
            hdul.close()

            # Normalize
            data_norm = (data - data.min()) / (data.max() - data.min())

            # Format handling
            if data.ndim == 3:
                if data.shape[0] == 3:
                    data_norm = np.transpose(data_norm, (1, 2, 0))
                image = (data_norm * 255).astype("uint8")
                image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            else:
                image = (data_norm * 255).astype("uint8")
                gray = image

            # Save Original
            cv.imwrite("./results/original.png", image)

            # 1. Erosion
            kernel_img = np.ones((IMAGE_EROSION_SIZE, IMAGE_EROSION_SIZE), np.uint8)
            image_eroded_step1 = cv.erode(
                image, kernel_img, iterations=IMAGE_EROSION_ITER
            )

            # 2. Mask
            mask = cv.adaptiveThreshold(
                gray,
                255,
                cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv.THRESH_BINARY,
                MASK_BLOCK,
                MASK_C,
            )
            kernel_m = np.ones((OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE), np.uint8)
            mask_cleaned = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_m)
            mask_dilated = cv.dilate(
                mask_cleaned, kernel_m, iterations=MASK_DILATE_ITER
            )

            # Save Mask
            cv.imwrite("./results/star_mask.png", mask_dilated)

            # 3. Inpaint
            eroded_final = cv.inpaint(
                image_eroded_step1, mask_dilated, INPAINT_RADIUS, cv.INPAINT_TELEA
            )

            # Save Eroded (Inpainted version as per prototype naming)
            cv.imwrite("./results/eroded.png", eroded_final)

            # 4. Fusion
            mask_blurred = cv.GaussianBlur(mask_dilated, (BLUR_SIZE, BLUR_SIZE), 0)
            M = mask_blurred.astype(np.float32) / 255.0
            if image.ndim == 3:
                M = np.stack([M] * 3, axis=-1)

            Ioriginal = image.astype(np.float32)
            Ieroded = eroded_final.astype(np.float32)

            final_image_float = (M * REDUCTION_ALPHA * Ieroded) + (
                1.0 - (M * REDUCTION_ALPHA)
            ) * Ioriginal
            final_image = np.clip(final_image_float, 0, 255).astype(np.uint8)

            # Save Final
            cv.imwrite("./results/final_phase3.png", final_image)

            QMessageBox.information(
                self,
                "Succès",
                "Les 4 images ont été générées dans le dossier ./results/",
            )

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Une erreur est survenue : {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    launcher = Launcher()
    launcher.show()
    sys.exit(app.exec())
