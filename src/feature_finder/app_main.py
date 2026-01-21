import ctypes
import logging
import math
import os
import sys
from copy import deepcopy

import cv2
import numpy as np
import pygetwindow as gw
import winsound
from PySide6.QtCore import QRectF, QSize, Slot, QSignalBlocker, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QSizePolicy, QApplication, QWidget, QStyleFactory,
                               QHBoxLayout, QFileDialog)

from feature_finder.interface.ui_form import Ui_featureFinder
from feature_finder.detection_methods import DetectionBase, SFRDetection, CHDetection, DefaultSettings
from feature_finder.processing_support import convert_color_bit, check_path


class FeatureFinder(QWidget):
    """
    Main widget for the feature-finder application.
    """

    def __init__(self, parent=None):
        """
        Initialize the feature-finder widget.

        :param parent: Parent widget
        """
        super().__init__(parent)
        self.ui = Ui_featureFinder()
        self.ui.setupUi(self)

        # Define class variables
        self._display: Display = Display(self)
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._raw_image: np.ndarray = np.array([])
        self.detection_settings: DefaultSettings = DefaultSettings()
        self.detector: DetectionBase | None = None
        self.drawn_image: np.ndarray = np.array([])

        # Startup routines for GUI
        self._startup()

    def _add_logger(self):
        """
        Add a file handler to the logger.
        """
        logger_path = os.path.join(os.path.dirname(__file__), "resources", "feature_finder_log.log")
        if os.path.exists(logger_path):
            os.remove(logger_path)
        os.makedirs(os.path.dirname(logger_path), exist_ok=True)
        file_handler = logging.FileHandler(logger_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

    def _attach_functions_to_widgets(self):
        """
        Attach functions to various widgets in the UI.
        """
        # Single sliders
        self.ui.circularity_slider.sliderReleased.connect(self._update_image)
        self.ui.circularity_slider.valueChanged.connect(self._change_circularity_slider)

        self.ui.gauss_blur_slider.sliderReleased.connect(self._update_image)
        self.ui.gauss_blur_slider.valueChanged.connect(self._change_gauss_blur_slider)

        self.ui.threshold_slider.sliderReleased.connect(self._update_image)
        self.ui.threshold_slider.valueChanged.connect(self._change_threshold_slider)

        # Spin boxes
        self.ui.circularity_spin.lineEdit().returnPressed.connect(self._change_circularity_spin)
        self.ui.gauss_blur_spin.lineEdit().returnPressed.connect(self._change_gauss_blur_spin)
        self.ui.threshold_spin.lineEdit().returnPressed.connect(self._change_threshold_spin)

        # Range sliders
        blob_size_range_slider = self.ui.blob_size_range_slider.parent().findChildren(QRangeSlider)[0]
        blob_size_range_slider.sliderReleased.connect(self._update_image)
        blob_size_range_slider.valueChanged.connect(self._change_blob_size_range)

        feature_size_range_slider = self.ui.feature_size_range_slider.parent().findChildren(QRangeSlider)[0]
        feature_size_range_slider.sliderReleased.connect(self._update_image)
        feature_size_range_slider.valueChanged.connect(self._change_feature_size_range)

        # Buttons / Check boxes
        self.ui.crosshair_detection_check.clicked.connect(self._change_detection_method)
        self.ui.file_path_browse_button.clicked.connect(self._click_browse_file)
        self.ui.save_image_button.clicked.connect(self._click_save_drawing)

    def _change_feature_size_range(self):
        """
        Update the feature size range labels and image.
        """
        new_val = self.feature_size_range
        self.ui.feature_size_min.setText(str(new_val[0]))
        self.ui.feature_size_max.setText(str(new_val[1]))
        self._update_image()

    def _change_blob_size_range(self):
        """
        Update the blob size range labels and image.
        """
        new_val = self.blob_size_range
        self.ui.blob_size_min.setText(str(new_val[0]))
        self.ui.blob_size_max.setText(str(new_val[1]))
        self._update_image()

    def _change_gauss_blur_slider(self):
        """
        Update the Gaussian blur spin box value and image when the slider changes.
        """
        new_val = self.ui.gauss_blur_slider.value()
        with QSignalBlocker(self.ui.gauss_blur_spin):
            self.ui.gauss_blur_spin.setValue(new_val)
        self._update_image()

    def _change_gauss_blur_spin(self):
        """
        Update the Gaussian blur slider value when the spin box changes.
        """
        self.ui.gauss_blur_slider.setValue(self.gauss_blur)

    def _change_threshold_slider(self):
        """
        Update the threshold spin box value and image when the slider changes.
        """
        new_val = self.ui.threshold_slider.value()
        with QSignalBlocker(self.ui.threshold_spin):
            self.ui.threshold_spin.setValue(new_val)
        self._update_image()

    def _change_threshold_spin(self):
        """
        Update the threshold slider value when the spin box changes.
        """
        self.ui.threshold_slider.setValue(self.threshold)

    def _change_circularity_slider(self):
        """
        Update the circularity spin box value and image when the slider changes.
        """
        new_val = self.ui.circularity_slider.value()
        with QSignalBlocker(self.ui.circularity_spin):
            self.ui.circularity_spin.setValue(new_val / 100)
        self._update_image()

    def _change_circularity_spin(self):
        """
        Update the circularity slider value when the spin box changes.
        """
        self.ui.circularity_slider.setValue(self.circularity)

    def _change_detection_method(self):
        """
        Change the detection method and update the UI accordingly.
        """
        # Update blob size labels
        low_val = int(min(self.detection_settings.blob_size) * self.range_size_factor)
        high_val = int(max(self.detection_settings.blob_size) * self.range_size_factor)
        self.ui.blob_size_min.setText(str(low_val))
        self.ui.blob_size_max.setText(str(high_val))

        # Update feature size labels
        low_val = int(min(self.detection_settings.feature_size) * self.range_size_factor)
        high_val = int(max(self.detection_settings.feature_size) * self.range_size_factor)
        self.ui.feature_size_min.setText(str(low_val))
        self.ui.feature_size_max.setText(str(high_val))

        # Define detector type
        if self._raw_image.size > 0:
            if self.rect_detection_status:
                self.detector = SFRDetection(self._raw_image)
            else:
                self.detector = CHDetection(self._raw_image)

        # Update drawing
        self._update_image()

    def _click_browse_file(self):
        """
        Open a file dialog to browse and select an image file.
        """

        def import_image(file_path: str) -> np.ndarray:
            """
            Import an image file and convert it to an RGB8 numpy array.

            :param file_path: Path to the image file
            :return: Numpy array containing the image data
            """
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
                # Set the entry box to the valid image path and log
                self.ui.file_path_entry.setText(selected_file)
                self._logger.info(f"Imported image at: {selected_file}")

                # Store the image array
                self._raw_image = image_array
                self.drawn_image = convert_color_bit(image_array, color_channels=3, out_bit_depth=8)

                # Update range sliders
                feature_size_slider = self.ui.feature_size_range_slider.parent().findChildren(QRangeSlider)[0]
                feature_size_slider._maximum = self.range_slider_max * self.range_size_factor

                blob_size_slider = self.ui.blob_size_range_slider.parent().findChildren(QRangeSlider)[0]
                blob_size_slider._maximum = self.range_slider_max * self.range_size_factor

                # Init the appropriate detector
                if self.rect_detection_status:
                    self.detector = SFRDetection(self._raw_image)
                else:
                    self.detector = CHDetection(self._raw_image)

                # Update the stream window to show imported image
                self._update_image()
            else:
                self._dialog("Invalid file selected!\n\nPlease check the file path and integrity.", level=1)
                self._logger.warning(f"Ignoring invalid image at {selected_file}")

    def _click_save_drawing(self):
        """
        Save the current drawn image to a file.
        """
        if self._raw_image.size > 0:
            # Define file path
            file_path = os.path.join(os.path.dirname(self.ui.file_path_entry.toPlainText()), "ff_drawing.png")

            # Attempt to save the image
            if self.drawn_image.size > 0:
                # Check the image path
                checked_path = check_path(file_path)

                # Save the image with cv2
                try:
                    if not os.path.isdir(os.path.dirname(checked_path)):
                        os.makedirs(os.path.dirname(checked_path))
                    cv2.imwrite(checked_path, self.drawn_image)
                except PermissionError:
                    file_path = os.path.join(os.getcwd(), os.path.basename(file_path))
                    self._logger.warning(f"Lacking write permissions for this directory. Saving locally instead.")
                    cv2.imwrite(file_path, self.drawn_image)

            # Log location of image
            self._logger.info(f"Image saved at: {file_path}")

    def _dialog(self, message: str, button: hex = 0x0, level: int = 0) -> int:
        """
        Display a dialog box with the given message.

        :param message: Message to display in the dialog box
        :param button: Button options (0x0 = OK, 0x01 = OK/CANCEL, 0x03 = YES/NO/CANCEL, 0x04 = YES/NO)
        :param level: Dialog level (0 = prompt, 1 = warning, 2 = error)
        :return: User's response to the dialog.
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

    def _set_defaults(self):
        """
        Set default values for UI elements.
        """
        settings = self.detection_settings

        self.ui.threshold_slider.setValue(settings.threshold)
        self.ui.threshold_spin.setValue(settings.threshold)

        self.ui.gauss_blur_slider.setValue(settings.gauss)
        self.ui.gauss_blur_spin.setValue(settings.gauss)

        self.ui.circularity_spin.setValue(settings.circularity_min)
        self.ui.circularity_slider.setValue(settings.circularity_min * 100)

    def _setup_range_sliders(self, target_widget: QHBoxLayout):
        """
        Fill in the placeholders with the custom range sliders.

        :param target_widget: Placeholder handles.
        """
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
        range_slider.setOrientation(Qt.Orientation.Horizontal)
        range_slider._minimum = 0
        range_slider._maximum = self.range_slider_max * self.range_size_factor
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
        """
        Routines to run on startup of the GUI.
        """
        # Init logger
        self._add_logger()

        # Set up custom range sliders
        self._setup_range_sliders(self.ui.feature_size_range_slider)
        self._setup_range_sliders(self.ui.blob_size_range_slider)

        # Set default values of widget
        self._set_defaults()

        # Attach functionality
        self._attach_functions_to_widgets()

    def _update_image(self):
        """
        Update the drawn image internally.
        """
        if self._raw_image.size > 0:
            # Pre-process image
            update_next = self.detector.apply_gauss_blur(self.gauss_blur, update=(self.gauss_blur !=
                                                                                  self.detection_settings.gauss))
            update_next = self.detector.apply_threshold(self.threshold, update=(update_next or self.threshold !=
                                                                                self.detection_settings.threshold))
            if not self.rect_detection_status:
                self.detector.apply_hough_transform(self.threshold, self.feature_size_range[0],
                                                    update=(update_next or self.threshold !=
                                                            self.detection_settings.threshold or
                                                            self.feature_size_range[0] !=
                                                            self.detection_settings.feature_size[0]))

            # Find and draw features/detections
            self.drawn_image = self.detector.find_features_and_draw(self.blob_size_range, self.circularity,
                                                                    self.feature_size_range)

            # Update stored detection settings
            self.detection_settings.blob_size = deepcopy(self.blob_size_range)
            self.detection_settings.circularity_min = deepcopy(self.circularity)
            self.detection_settings.feature_size = deepcopy(self.feature_size_range)
            self.detection_settings.gauss = deepcopy(self.gauss_blur)
            self.detection_settings.threshold = deepcopy(self.threshold)

            # Log settings
            self._logger.info(f"Updated image with settings:\n"
                              f"   blob size = {self.detection_settings.blob_size}"
                              f"   circularity = {self.detection_settings.circularity_min}"
                              f"   feature size = {self.detection_settings.feature_size}"
                              f"   gauss kernel = {self.detection_settings.gauss}"
                              f"   threshold = {self.detection_settings.threshold}")

            # Update stream window
            self._update_stream_window()

    def _update_stream_window(self):
        """
        Update the stream window with the drawn image.
        """
        if self.drawn_image.size > 0:
            s = self.drawn_image.shape
            q_image = QImage(self.drawn_image.tobytes(), s[1], s[0], 3 * s[1], QImage.Format.Format_RGB888)
            self._display.on_image_received(q_image)

    @property
    def circularity(self) -> float:
        """
        Selected circularity limit for blob detection.

        :return: Circularity limit.
        """
        return float(self.ui.circularity_spin.value())

    @property
    def blob_size_range(self) -> tuple:
        """
        Selected range of detected blob sizes.

        :return: Blob size range.
        """
        raw_range = self.ui.blob_size_range_slider.parent().findChildren(QRangeSlider)[0].value()
        return tuple([i * self.range_size_factor for i in raw_range])

    @property
    def feature_size_range(self) -> tuple:
        """
        Selected range of detected feature (rectangles or crosshairs) sizes.

        :return: Feature size range.
        """
        raw_range = self.ui.feature_size_range_slider.parent().findChildren(QRangeSlider)[0].value()
        return tuple([i * self.range_size_factor for i in raw_range])

    @property
    def gauss_blur(self) -> int:
        """
        Selected gaussian blur kernel size.

        :return: Gaussian blur kernel.
        """
        # Get raw value from UI
        val = int(self.ui.gauss_blur_spin.value())

        # Ensure return is greater than 1 and odd
        if val < 0:
            val = 1
        elif val % 2 == 0:
            val += 1

        return val

    @property
    def range_slider_max(self) -> int:
        """
        Slider range maximum based on image size.

        :return: Slider range maximum.
        """
        if self._raw_image.size > 0:
            s = self._raw_image.shape
            return s[0] * s[1] * 2
        else:
            return 1000000

    @property
    def range_size_factor(self) -> float:
        """
        Factor applied to size sliders based on detection method.

        :return: Slider size factor.
        """
        if not self.rect_detection_status:
            size_factor = 1 / 100
        else:
            size_factor = 1.0

        return size_factor

    @property
    def rect_detection_status(self) -> bool:
        """
        Flag for rect. detection method.

        :return: Crosshair detection (false) or rectangle detection (true).
        """
        return self.ui.crosshair_detection_check.isChecked()

    @property
    def threshold(self) -> int:
        """
        Selected pixel threshold value.

        :return: Threshold value.
        """
        return int(self.ui.threshold_spin.value())


class Display(QGraphicsView):

    def __init__(self, parent: FeatureFinder = None):
        """
        Graphics viewer object used to display frames from the camera.
        :param parent: Qt main window handle.
        """
        super().__init__(parent)

        self.setObjectName(u"stream_window")
        size_policy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        size_policy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(size_policy)
        self.setMinimumSize(QSize(471, 411))
        parent.ui.gridLayout_2.addWidget(self, 0, 2, 1, 1)
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


def launch_gui():
    """
    Main functionality. Launches GUI for feature detection.
    :return:
    """
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("WindowsVista"))
    widget = FeatureFinder()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
