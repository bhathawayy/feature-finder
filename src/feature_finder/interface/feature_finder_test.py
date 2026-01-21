# This Python file uses the following encoding: utf-8
import sys

from PySide6.QtWidgets import QApplication, QWidget, QStyleFactory

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from ui_form import Ui_featureFinder

class featureFinder(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_featureFinder()
        self.ui.setupUi(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("fusion"))
    widget = featureFinder()
    widget.show()
    sys.exit(app.exec())
