import os.path
import sys
import pygetwindow as gw

import ctypes
import cv2
import logging
import math
import numpy as np
import winsound

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget, QStyleFactory, QHBoxLayout, QFileDialog
from qtrangeslider import QRangeSlider
from PySide6.QtCore import (QRectF, QSize, Slot)
from PySide6.QtGui import (QImage, QPainter)
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QSizePolicy)
from featureFinder.processing_support import convert_color_bit
from detection_methods import DefaultSettings

from app_ui import Ui_FeatureFinder


class FeatureFinder(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_FeatureFinder()
        self.ui.setupUi(self)

        # Define class variables
        self._display: Display = Display(self)
        self._logger: logging.Logger = logging.getLogger(__name__)
        self.detection_settings: DefaultSettings = DefaultSettings()
        self.drawn_image: np.ndarray = np.array([])
        self.rgb8_image: np.ndarray = np.array([])

        # Startup routines for GUI
        self._startup()

    def _add_logger(self):
        os.path.dirname(__file__)
        logger_path = os.path.join(os.path.dirname(__file__), "feature_finder_log.log")
        if os.path.exists(logger_path):
            os.remove(logger_path)
        file_handler = logging.FileHandler(logger_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def _attach_functions_to_widgets(self):
        # Detection method checkbox
        self.ui.crosshair_detection_check.clicked.connect(self._change_detection_method)

        # Gaussian blur controls
        self.ui.gauss_blur_slider.sliderReleased.connect(self._update_image)
        self.ui.gauss_blur_slider.valueChanged.connect(self._change_gauss_blur_slider)
        self.ui.gauss_blur_spin.lineEdit().returnPressed.connect(self._change_gauss_blur_spin)

        # Pixel threshold controls
        self.ui.threshold_slider.sliderReleased.connect(self._update_image)
        self.ui.threshold_slider.valueChanged.connect(self._change_threshold_slider)
        self.ui.threshold_spin.lineEdit().returnPressed.connect(self._change_threshold_spin)

        # Circularity controls
        self.ui.circularity_slider.sliderReleased.connect(self._update_image)
        self.ui.circularity_slider.valueChanged.connect(self._change_circularity_slider)
        self.ui.circularity_spin.lineEdit().returnPressed.connect(self._change_circularity_spin)

        # Blob size range slider
        blob_size_range_slider = self.ui.blob_size_range_slider.parent().findChildren(QRangeSlider)[0]
        blob_size_range_slider.sliderReleased.connect(self._update_image)
        blob_size_range_slider.valueChanged.connect(self._change_blob_size_range)

        # Feature size range slider
        feature_size_range_slider = self.ui.feature_size_range_slider.parent().findChildren(QRangeSlider)[0]
        feature_size_range_slider.sliderReleased.connect(self._update_image)
        feature_size_range_slider.valueChanged.connect(self._change_feature_size_range)

        # File browser
        self.ui.file_path_browse_button.clicked.connect(self._click_browse_file)

    def _click_browse_file(self, ):

        def import_image(file_path: str) -> np.ndarray:
            if os.path.isfile(file_path):
                raw_array = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                rgb8_image = convert_color_bit(raw_array, color_channels=3, out_bit_depth=8)
            else:
                rgb8_image = np.array([])

            return rgb8_image

        # Setup file browser dialog box
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open File")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)

        if file_dialog.exec():
            selected_file = file_dialog.selectedFiles()[0]
            image_array = import_image(selected_file)
            if image_array.size > 0:
                # Set the entry box to the valid image path
                self.ui.file_path_entry.setText(selected_file)

                # Store array internally
                self.rgb8_image = image_array
                self.drawn_image = image_array.copy()

                # Update the stream window to show imported image
                self._update_stream_window()
            else:
                self._dialog("Invalid file selected!\n\nPlease check the file path and integrity.", level=1)

    def _change_feature_size_range(self):
        new_val = self.feature_size_range
        self.ui.feature_size_min.setText(str(new_val[0]))
        self.ui.feature_size_max.setText(str(new_val[1]))

    def _change_blob_size_range(self):
        new_val = self.blob_size_range
        self.ui.blob_size_min.setText(str(new_val[0]))
        self.ui.blob_size_max.setText(str(new_val[1]))

    def _change_gauss_blur_slider(self):
        new_val = self.ui.gauss_blur_slider.value()
        self.ui.gauss_blur_spin.setValue(new_val)

    def _change_gauss_blur_spin(self):
        self.ui.gauss_blur_slider.setValue(self.gauss_blur)

    def _change_threshold_slider(self):
        new_val = self.ui.threshold_slider.value()
        self.ui.threshold_spin.setValue(new_val)

    def _change_threshold_spin(self):
        self.ui.threshold_slider.setValue(self.threshold)

    def _change_circularity_slider(self):
        new_val = self.ui.circularity_slider.value()
        self.ui.circularity_spin.setValue(new_val)

    def _change_circularity_spin(self):
        self.ui.circularity_slider.setValue(self.circularity)

    def _change_detection_method(self):
        blob_size_range_slider = self.ui.blob_size_range_slider.parent().findChildren(QRangeSlider)[0]
        blob_size_range_slider._maximum = self.detection_settings.range_slider_max * self.range_size_factor
        blob_size_range_slider._maximum = self.detection_settings.range_slider_max * self.range_size_factor
        blob_size_range_slider.setValue((int(min(self.detection_settings.blob_size) * self.range_size_factor),
                                         int(max(self.detection_settings.blob_size) * self.range_size_factor)))

        feature_size_range_slider = self.ui.feature_size_range_slider.parent().findChildren(QRangeSlider)[0]
        feature_size_range_slider._maximum = self.detection_settings.range_slider_max * self.range_size_factor
        feature_size_range_slider._maximum = self.detection_settings.range_slider_max * self.range_size_factor
        feature_size_range_slider.setValue((int(min(self.detection_settings.feature_size) * self.range_size_factor),
                                            int(max(self.detection_settings.feature_size) * self.range_size_factor)))

    def _dialog(self, message: str, button: hex = 0x0, level: int = 0) -> int:
        """
        Initiate a dialog box to prompt or inform the user.
        :param message: Message to display in the dialog box.
        :param button: Options: 0x0 = OK, 0x01 = OK/CANCEL, 0x03 = YES/NO/CANCEL, 0x04 = YES/NO
        :param level: Options: 0 = prompt, 1 = warning, 2 = error
        :return: Answer from the user.
        """
        # Set local variables
        message = bytes(message, 'utf-8')
        title = b"Action Required"
        icon = 0x40  # info icon
        if level == 1:
            title = b"Warning"
            icon = 0x30  # icon exclaim/warning
        elif level == 2:
            title = b"ERROR"
            icon = 0x10  # icon stop/error

        # Set widget as active window
        try:
            win = gw.getWindowsWithTitle(self.window().windowTitle())[0]
            win.activate()
        except (IndexError, RuntimeError):
            self._logger.warning(f"App not open. Could not display dialog: {message}")
            return -1
        except gw.PyGetWindowException:
            pass

        # Display dialog message
        if level != 2:
            winsound.Beep(800, 500)
        dialog_answer = ctypes.windll.user32.MessageBoxA(0, message, title, button | icon | 0x00001000)

        return dialog_answer

    def _setup_range_sliders(self, target_widget: QHBoxLayout):
        # Define what default values to use
        if "blob" in target_widget.objectName().lower():
            defaults = self.detection_settings.blob_size
        else:
            defaults = self.detection_settings.feature_size

        # Define default values
        low_val = int(min(defaults) * self.range_size_factor)
        high_val = int(max(defaults) * self.range_size_factor)

        # Set parameters for target range slider
        range_slider = QRangeSlider()
        range_slider.setOrientation(Qt.Horizontal)
        range_slider._minimum = 0
        range_slider._maximum = self.detection_settings.range_slider_max * self.range_size_factor
        range_slider.setValue((low_val, high_val))

        # Add configured slider to the application
        target_widget.addWidget(range_slider)

        # Update labels
        if "blob" in target_widget.objectName().lower():
            self.ui.blob_size_min.setText(str(low_val))
            self.ui.blob_size_max.setText(str(high_val))
        else:
            self.ui.feature_size_min.setText(str(low_val))
            self.ui.feature_size_max.setText(str(high_val))

    def _startup(self):
        # Init logger
        self._add_logger()

        # Set up custom range sliders
        self._setup_range_sliders(self.ui.feature_size_range_slider)
        self._setup_range_sliders(self.ui.blob_size_range_slider)

        # Attach functionality
        self._attach_functions_to_widgets()

    def _update_image(self):
        pass

    def _update_stream_window(self):
        if self.drawn_image.size > 0:
            s = self.drawn_image.shape
            q_image = QImage(self.drawn_image.tobytes(), s[1], s[0], 3 * s[1], QImage.Format.Format_RGB888)
            self._display.on_image_received(q_image)

    @property
    def circularity(self) -> float:
        return float(self.ui.circularity_spin.value())

    @property
    def blob_size_range(self) -> tuple:
        return self.ui.blob_size_range_slider.parent().findChildren(QRangeSlider)[0].value()

    @property
    def feature_size_range(self) -> tuple:
        return self.ui.feature_size_range_slider.parent().findChildren(QRangeSlider)[0].value()

    @property
    def gauss_blur(self) -> int:
        # Get raw value from UI
        val = int(self.ui.gauss_blur_spin.value())

        # Ensure return is greater than 1 and odd
        if val < 0:
            val = 1
        elif val % 2 == 0:
            val += 1

        return val

    @property
    def range_size_factor(self) -> float:
        if self.rect_detection_status:
            size_factor = 1 / 100
        else:
            size_factor = 1.0

        return size_factor

    def rect_detection_status(self) -> bool:
        return self.ui.crosshair_detection_check.isChecked()

    @property
    def threshold(self) -> int:
        return int(self.ui.threshold_spin.value())


class Display(QGraphicsView):

    def __init__(self, parent: FeatureFinder = None):
        """
        Graphics viewer object used to display frames from the camera.
        :param parent: Qt main window handle.
        """
        super().__init__(parent)

        self.setObjectName(u"stream_window")
        size_policy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        size_policy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(size_policy)
        self.setMinimumSize(QSize(445, 450))
        parent.ui.gridLayout.addWidget(self, 0, 0, 1, 1)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

        self.scene = CustomGraphicsScene(self)
        self.setScene(self.scene)

    @Slot(QImage)
    def on_image_received(self, image: QImage):
        """
        Action for when a new image is received.
        :param image: New image frame from camera.
        :return: None
        """
        self.scene.set_image(image)


class CustomGraphicsScene(QGraphicsScene):
    def __init__(self, parent: Display = None):
        """
        Graphics scene object used to display frames from the camera.
        :param parent: Qt main window handle.
        """
        super().__init__(parent)
        self.root = parent
        self.image = QImage()
        self.zoom = 1

    def wheelEvent(self, event):
        """
        Handle the wheel event for zooming in/out under the mouse cursor.
        """
        if self.root.underMouse():  # Ensure the mouse is over the widget
            # Get the zoom factor
            zoom_in_factor = 1.15
            zoom_out_factor = 1 / zoom_in_factor

            # Determine whether to zoom in or out
            if event.delta() > 0:
                scale_factor = zoom_in_factor
            else:
                scale_factor = zoom_out_factor

            # Apply transformation anchor under the mouse
            if 33 >= self.zoom * scale_factor >= 1:  # prevents unlimited zoom out/in
                self.root.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

                # Scale the view
                self.root.scale(scale_factor, scale_factor)

                # Update internal zoom level
                self.zoom *= scale_factor

    def set_image(self, image: QImage):
        """
        Set the image handle to be equal to the camera frame.
        :param image: New image frame from camera.
        :return: None
        """
        self.image = image
        self.update()

    def drawBackground(self, painter: QPainter, rect: QRectF):
        """
        Over-ride the internal drawBackground command.
        :param painter: Qt painter object used to "draw" image.
        :param rect: Qt rectangular object to define image size.
        :return: None
        """
        # Display size
        display_width = self.root.width()
        display_height = self.root.height()

        # Image size
        image_width = self.image.width()
        image_height = self.image.height()

        # Return if we don't have an image yet
        if image_width == 0 or image_height == 0:
            return

        # Calculate aspect ratio of display and image
        ratio1 = display_width / display_height
        ratio2 = image_width / image_height
        if ratio1 > ratio2:
            # The height with must fit to the display height.So h remains and w must be scaled down
            image_width = display_height * ratio2
            image_height = display_height
        else:
            # The image with must fit to the display width. So w remains and h must be scaled down
            image_width = display_width
            image_height = display_height / ratio2

        # Format scene/image size
        image_pos_x = -1.0 * (image_width / 2.0)
        image_pox_y = -1.0 * (image_height / 2.0)
        image_pos_x = math.trunc(image_pos_x)
        image_pox_y = math.trunc(image_pox_y)
        rect = QRectF(image_pos_x, image_pox_y, image_width, image_height)

        # Draw scene/image
        painter.drawImage(rect, self.image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("WindowsVista"))
    widget = FeatureFinder()
    widget.show()
    sys.exit(app.exec())
