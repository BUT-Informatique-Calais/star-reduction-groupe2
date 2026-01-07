import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import Qt
from gui_star_reduction import StarModel, StarView, StarController

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

        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Buttons
        self.btn_realtime = QPushButton("Mode Temps Réel")
        self.btn_realtime.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_realtime.clicked.connect(self.launch_realtime)
        layout.addWidget(self.btn_realtime)

        self.btn_comparison = QPushButton("Mode Comparaison")
        self.btn_comparison.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_comparison.clicked.connect(self.launch_comparison)
        layout.addWidget(self.btn_comparison)

        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        self.setLayout(layout)

    def launch_realtime(self):
        # Create MVC components
        self.model = StarModel()
        self.view = StarView()
        self.controller = StarController(self.model, self.view)
        
        # Show main window and close launcher
        self.view.show()
        self.close()

    def launch_comparison(self):
        QMessageBox.information(self, "Information", "Le mode comparaison sera disponible prochainement.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    launcher = Launcher()
    launcher.show()
    sys.exit(app.exec())
