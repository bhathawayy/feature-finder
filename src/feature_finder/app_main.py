import logging
import math
import os
import sys
from copy import deepcopy

import cv2
import numpy as np
try:
    import pygetwindow as gw
except ImportError:
    gw = None
from PySide6.QtCore import QRectF, Slot, QSignalBlocker
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QSizePolicy, QApplication, QWidget, QStyleFactory,
                               QFileDialog, QMessageBox)

from feature_finder.detection_methods import DetectionBase, SFRDetection, CHDetection
from feature_finder.interface.ui_form import Ui_featureFinder
from feature_finder.processing_support import convert_color_bit, check_path, DefaultSettings


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
        
        :return: None
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
        
        :return: None
        """
        # Sliders
        self.ui.blob_max_size_slider.sliderReleased.connect(self._update_image)
        self.ui.blob_max_size_slider.valueChanged.connect(self._change_blob_size_slider)
        self.ui.blob_min_size_slider.sliderReleased.connect(self._update_image)
        self.ui.blob_min_size_slider.valueChanged.connect(lambda: self._change_blob_size_slider(False))
        self.ui.circularity_slider.sliderReleased.connect(self._update_image)
        self.ui.circularity_slider.valueChanged.connect(self._change_circularity_slider)
        self.ui.feature_max_size_slider.sliderReleased.connect(self._update_image)
        self.ui.feature_max_size_slider.valueChanged.connect(self._change_feature_size_slider)
        self.ui.feature_min_size_slider.sliderReleased.connect(self._update_image)
        self.ui.feature_min_size_slider.valueChanged.connect(lambda: self._change_feature_size_slider(False))
        self.ui.gauss_blur_slider.sliderReleased.connect(self._update_image)
        self.ui.gauss_blur_slider.valueChanged.connect(self._change_gauss_blur_slider)
        self.ui.hough_threshold_slider.sliderReleased.connect(self._update_image)
        self.ui.hough_threshold_slider.valueChanged.connect(self._change_hough_threshold_slider)
        self.ui.pixel_threshold_slider.sliderReleased.connect(self._update_image)
        self.ui.pixel_threshold_slider.valueChanged.connect(self._change_pixel_threshold_slider)

        # Spin boxes
        self.ui.blob_max_size_spin.lineEdit().returnPressed.connect(self._change_blob_size_spin)
        self.ui.blob_min_size_spin.lineEdit().returnPressed.connect(lambda: self._change_blob_size_spin(False))
        self.ui.circularity_spin.lineEdit().returnPressed.connect(self._change_circularity_spin)
        self.ui.feature_max_size_spin.lineEdit().returnPressed.connect(self._change_feature_size_spin)
        self.ui.feature_min_size_spin.lineEdit().returnPressed.connect(lambda: self._change_feature_size_spin(False))
        self.ui.gauss_blur_spin.lineEdit().returnPressed.connect(self._change_gauss_blur_spin)
        self.ui.pixel_threshold_spin.lineEdit().returnPressed.connect(self._change_pixel_threshold_spin)
        self.ui.hough_threshold_spin.lineEdit().returnPressed.connect(self._change_hough_threshold_spin)

        # Buttons / Check boxes
        self.ui.crosshair_detection_check.clicked.connect(self._change_detection_method)
        self.ui.file_path_browse_button.clicked.connect(self._click_browse_file)
        self.ui.save_image_button.clicked.connect(self._click_save_drawing)

    def _change_blob_size_slider(self, is_max_widget: bool = True):
        """
        Update the blob size slider value and image when the spin box changes.
        
        :param is_max_widget: Is max or min slider?
        :return: None
        """
        if is_max_widget:
            new_val = self.ui.blob_max_size_slider.value()
            min_val = self.ui.blob_min_size_slider.value()
            if new_val <= min_val:
                new_val = min_val + 1
                with QSignalBlocker(self.ui.blob_max_size_slider):
                    self.ui.blob_max_size_slider.setValue(new_val)
            spin_handle = self.ui.blob_max_size_spin
        else:
            new_val = self.ui.blob_min_size_slider.value()
            max_val = self.ui.blob_max_size_slider.value()
            if new_val >= max_val:
                new_val = max_val - 1
                with QSignalBlocker(self.ui.blob_min_size_slider):
                    self.ui.blob_min_size_slider.setValue(new_val)
            spin_handle = self.ui.blob_min_size_spin
        with QSignalBlocker(spin_handle):
            spin_handle.setValue(new_val)
        self._update_image()

    def _change_blob_size_spin(self, is_max_widget: bool = True):
        """
        Update the blob size spin box value and image when the slider changes.

        :param is_max_widget: Is max or min spin box?
        :return: None
        """
        if is_max_widget:
            new_val = self.ui.blob_max_size_spin.value()
            min_val = self.ui.blob_min_size_slider.value()
            if new_val <= min_val:
                new_val = min_val + 1
                with QSignalBlocker(self.ui.blob_max_size_spin):
                    self.ui.blob_max_size_spin.setValue(new_val)
            slider_handle = self.ui.blob_max_size_slider
        else:
            new_val = self.ui.blob_min_size_spin.value()
            max_val = self.ui.blob_max_size_slider.value()
            if new_val >= max_val:
                new_val = max_val - 1
                with QSignalBlocker(self.ui.blob_min_size_spin):
                    self.ui.blob_min_size_spin.setValue(new_val)
            slider_handle = self.ui.blob_min_size_slider
        slider_handle.setValue(new_val)
        self._update_image()

    def _change_feature_size_slider(self, is_max_widget: bool = True):
        """
        Update the feature size slider value and image when the spin box changes.

        :param is_max_widget: Is max or min slider?
        :return: None
        """
        if is_max_widget:
            new_val = self.ui.feature_max_size_slider.value()
            min_val = self.ui.feature_min_size_slider.value()
            if new_val <= min_val:
                new_val = min_val + 1
                with QSignalBlocker(self.ui.feature_max_size_slider):
                    self.ui.feature_max_size_slider.setValue(new_val)
            spin_handle = self.ui.feature_max_size_spin
        else:
            new_val = self.ui.feature_min_size_slider.value()
            max_val = self.ui.feature_max_size_slider.value()
            if new_val >= max_val:
                new_val = max_val - 1
                with QSignalBlocker(self.ui.feature_min_size_slider):
                    self.ui.feature_min_size_slider.setValue(new_val)
            spin_handle = self.ui.feature_min_size_spin
        with QSignalBlocker(spin_handle):
            spin_handle.setValue(new_val)
        self._update_image()

    def _change_feature_size_spin(self, is_max_widget: bool = True):
        """
        Update the feature size spin box value and image when the slider changes.

        :param is_max_widget: Is max or min spin box?
        :return: None
        """
        if is_max_widget:
            new_val = self.ui.feature_max_size_spin.value()
            min_val = self.ui.feature_min_size_slider.value()
            if new_val <= min_val:
                new_val = min_val + 1
                with QSignalBlocker(self.ui.feature_max_size_spin):
                    self.ui.feature_max_size_spin.setValue(new_val)
            slider_handle = self.ui.feature_max_size_slider
        else:
            new_val = self.ui.feature_min_size_spin.value()
            max_val = self.ui.feature_max_size_slider.value()
            if new_val >= max_val:
                new_val = max_val - 1
                with QSignalBlocker(self.ui.feature_min_size_spin):
                    self.ui.feature_min_size_spin.setValue(new_val)
            slider_handle = self.ui.feature_min_size_slider
        slider_handle.setValue(new_val)
        self._update_image()

    def _change_gauss_blur_slider(self):
        """
        Update the Gaussian blur spin box value and image when the slider changes.
        
        :return: None
        """
        new_val = self.ui.gauss_blur_slider.value()
        with QSignalBlocker(self.ui.gauss_blur_spin):
            self.ui.gauss_blur_spin.setValue(new_val)
        self._update_image()

    def _change_gauss_blur_spin(self):
        """
        Update the Gaussian blur slider value when the spin box changes.
        
        :return: None
        """
        self.ui.gauss_blur_slider.setValue(self.gauss_blur)

    def _change_hough_threshold_slider(self):
        """
        Update the Hough threshold spin box value and image when the slider changes.

        :return: None
        """
        new_val = self.ui.hough_threshold_slider.value()
        with QSignalBlocker(self.ui.hough_threshold_spin):
            self.ui.hough_threshold_spin.setValue(new_val)
        self._update_image()

    def _change_hough_threshold_spin(self):
        """
        Update the Hough threshold slider value when the spin box changes.

        :return: None
        """
        self.ui.hough_threshold_slider.setValue(self.hough_threshold)

    def _change_pixel_threshold_slider(self):
        """
        Update the threshold spin box value and image when the slider changes.
        
        :return: None
        """
        new_val = self.ui.pixel_threshold_slider.value()
        with QSignalBlocker(self.ui.pixel_threshold_spin):
            self.ui.pixel_threshold_spin.setValue(new_val)
        self._update_image()

    def _change_pixel_threshold_spin(self):
        """
        Update the threshold slider value when the spin box changes.
        
        :return: None
        """
        self.ui.pixel_threshold_slider.setValue(self.pixel_threshold)

    def _change_circularity_slider(self):
        """
        Update the circularity spin box value and image when the slider changes.
        
        :return: None
        """
        new_val = self.ui.circularity_slider.value()
        with QSignalBlocker(self.ui.circularity_spin):
            self.ui.circularity_spin.setValue(new_val / 100)
        self._update_image()

    def _change_circularity_spin(self):
        """
        Update the circularity slider value when the spin box changes.
        
        :return: None
        """
        self.ui.circularity_slider.setValue(self.circularity)

    def _change_detection_method(self):
        """
        Change the detection method and update the UI accordingly.
        
        :return: None
        """
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

        :return: None
        """

        def import_image(file_path: str) -> np.ndarray:
            """
            Import an image file and convert it to an RGB8 numpy array.

            :param file_path: Path to the image file
            :return: Numpy array containing the image data
            """
            ok = False
            rgb_image =  np.array([])

            # Try to import image
            if os.path.isfile(file_path):
                raw_array = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if raw_array.size > 0 and raw_array is not None:
                    rgb_image = convert_color_bit(raw_array, color_channels=3, out_bit_depth=8)
                    ok = True

            # Case for bad path
            if ok:
                self._logger.info(f"Imported image at: {file_path}")
            else:
                self._dialog_and_log(f"Invalid file selected!\n\nPlease check the file path and integrity at: "
                                     f"{file_path}", level=2)

            return rgb_image

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

                # Store the image array
                self._raw_image = image_array
                self.drawn_image = convert_color_bit(image_array, color_channels=3, out_bit_depth=8)

                # Update UI
                self.ui.save_image_button.setEnabled(True)

                # Init the appropriate detector
                if self.rect_detection_status:
                    self.detector = SFRDetection(self._raw_image)
                else:
                    self.detector = CHDetection(self._raw_image)

                # Update the stream window to show imported image
                self._update_image()

    def _click_save_drawing(self):
        """
        Save the current drawn image to a file.
        
        :return: None
        """
        if self._raw_image.size > 0:
            # Define file path
            file_path = os.path.join(os.path.dirname(self.ui.file_path_entry.toPlainText()), "ff_drawing.png")

            # Attempt to save the image
            if self.drawn_image.size > 0:
                # Check the image path
                checked_path = check_path(file_path)
                checked_dir = os.path.dirname(checked_path)

                # Save the image with cv2
                try:
                    if not os.path.isdir(checked_dir):
                        os.makedirs(checked_dir)
                    cv2.imwrite(checked_path, self.drawn_image)
                except PermissionError:
                    checked_dir = os.getcwd()
                    file_path = os.path.join(os.getcwd(), os.path.basename(file_path))
                    self._logger.warning(f"Lacking write permissions for this directory. Saving locally instead.")
                    cv2.imwrite(file_path, self.drawn_image)
                finally:
                    self._logger.info(f"Image saved at: {file_path}")

                # Save YAML files
                self.detection_settings.to_yaml(os.path.join(checked_dir, "detection_settings.yaml"))
                for f, feature in enumerate(self.detector.found_features):
                    write_mode = "w" if f == 0 else "a"
                    feature.to_yaml(os.path.join(checked_dir, "found_features.yaml"), write_mode=write_mode)
                self._logger.info(f"YAML files saved to: {file_path}")

    def _dialog_and_log(self, message: str, button: int = 0, level: int = 0, err_handle: Exception | None = None) -> int:
        """
        Display a dialog box with the given message.

        :param message: Message to display in the dialog box
        :param button: Button options (default = OK, 1 = OK/CANCEL, 3 = YES/NO/CANCEL, 4 = YES/NO)
        :param level: Dialog level (0 = prompt, 1 = warning, 2 = error)
        :return: User's response to the dialog.
        """
        # Set title and icon based on level
        msg_box = QMessageBox(self)
        msg_box.setText(message)
        if level == 1:
            msg_box.setWindowTitle("Warning")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            self._logger.warning(message)
        elif level == 2:
            msg_box.setWindowTitle("ERROR")
            msg_box.setIcon(QMessageBox.Icon.Critical)
            self._logger.error(f"{message}\n\n{str(err_handle)}")
        else:
            msg_box.setWindowTitle("Action Required")
            msg_box.setIcon(QMessageBox.Icon.Information)

        # Set buttons based on button parameter
        button_map = {
            1: QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            3: QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            4: QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        }
        msg_box.setStandardButtons(button_map.get(button, QMessageBox.StandardButton.Ok))

        # Activate window (cross-platform)
        self.activateWindow()
        self.raise_()

        return msg_box.exec()
        # # Set local variables
        # message = bytes(message, 'utf-8')
        # title = b"Action Required"
        # icon = 0x40  # info icon
        # if level == 1:
        #     title = b"Warning"
        #     icon = 0x30  # icon exclaim/warning
        # elif level == 2:
        #     title = b"ERROR"
        #     icon = 0x10  # icon stop/error
        #
        # # Set widget as active window
        # try:
        #     win = gw.getWindowsWithTitle(self.window().windowTitle())[0]
        #     win.activate()
        # except (IndexError, RuntimeError):
        #     self._logger.warning(f"App not open. Could not display dialog: {message}")
        #     return -1
        # except gw.PyGetWindowException:
        #     pass
        #
        # # Display dialog message
        # dialog_answer = ctypes.windll.user32.MessageBoxA(0, message, title, button | icon | 0x00001000)
        #
        # return dialog_answer

    def _set_defaults(self):
        """
        Set default values for UI elements.
        
        :return: None
        """
        settings = self.detection_settings

        self.ui.circularity_slider.setValue(settings.circularity_min * 100)
        self.ui.circularity_spin.setValue(settings.circularity_min)
        self.ui.gauss_blur_slider.setValue(settings.gauss)
        self.ui.gauss_blur_spin.setValue(settings.gauss)
        self.ui.pixel_threshold_slider.setValue(settings.pixel_threshold)
        self.ui.pixel_threshold_spin.setValue(settings.pixel_threshold)

    def _startup(self):
        """
        Routines to run on startup of the GUI.
        
        :return: None
        """
        # Init logger
        self._add_logger()

        # Set default values of widget
        self._set_defaults()

        # Attach functionality
        self._attach_functions_to_widgets()

    def _update_image(self):
        """
        Update the drawn image internally.
        
        :return: None
        """
        if self._raw_image.size > 0:
            # Pre-process image
            update_next = self.detector.apply_gauss_blur(self.gauss_blur, update=(self.gauss_blur !=
                                                                                  self.detection_settings.gauss))
            update_next = self.detector.apply_threshold(self.pixel_threshold,
                                                        update=(update_next or self.pixel_threshold !=
                                                                self.detection_settings.pixel_threshold))
            if not self.rect_detection_status:
                self.detector.apply_hough_transform(self.hough_threshold, self.feature_size_range[0],
                                                    update=(update_next or self.hough_threshold !=
                                                            self.detection_settings.hough_threshold or
                                                            self.feature_size_range[0] !=
                                                            self.detection_settings.feature_size[0]))

            # Find and draw features/detections
            self.detector.detect_features(self.feature_size_range, self.blob_size_range, self.circularity)
            self.drawn_image = self.detector.display_image

            # Update stored detection settings
            self.detection_settings.blob_size = deepcopy(self.blob_size_range)
            self.detection_settings.circularity_min = deepcopy(self.circularity)
            self.detection_settings.feature_size = deepcopy(self.feature_size_range)
            self.detection_settings.gauss = deepcopy(self.gauss_blur)
            self.detection_settings.hough_threshold = deepcopy(self.hough_threshold)
            self.detection_settings.pixel_threshold = deepcopy(self.pixel_threshold)

            # Log settings
            self._logger.info(f"Updated image with settings:\n"
                              f"\tblob size = {self.detection_settings.blob_size}"
                              f"\tcircularity = {self.detection_settings.circularity_min}"
                              f"\tfeature size = {self.detection_settings.feature_size}"
                              f"\tgauss kernel = {self.detection_settings.gauss}"
                              f"\though threshold = {self.detection_settings.hough_threshold}"
                              f"\tpixel threshold = {self.detection_settings.pixel_threshold}")

            # Update stream window
            self._update_stream_window()

    def _update_stream_window(self):
        """
        Update the stream window with the drawn image.
        
        :return: None
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
    def blob_size_range(self) -> tuple[float, float]:
        """
        Selected range of detected blob sizes.

        :return: Blob size range.
        """
        min_size = self.ui.blob_min_size_slider.value()
        max_size = self.ui.blob_max_size_slider.value()

        return min_size * self.range_size_factor, max_size * self.range_size_factor

    @property
    def feature_size_range(self) -> tuple[float, float]:
        """
        Selected range of detected feature (rectangles or crosshairs) sizes.

        :return: Feature size range.
        """
        min_size = self.ui.feature_min_size_slider.value()
        max_size = self.ui.feature_max_size_slider.value()

        return min_size * self.range_size_factor, max_size * self.range_size_factor

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
    def hough_threshold(self) -> int:
        """
        Selected Hough line threshold value.

        :return: Threshold value.
        """
        return int(self.ui.hough_threshold_spin.value())

    @property
    def pixel_threshold(self) -> int:
        """
        Selected pixel threshold value.

        :return: Threshold value.
        """
        return int(self.ui.pixel_threshold_spin.value())

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
        # self.setMinimumSize(QSize(471, 411))

        # Get stream position
        idx = parent.ui.gridLayout_2.indexOf(parent.ui.stream_window)
        row, col, row_span, col_span = parent.ui.gridLayout_2.getItemPosition(idx)
        parent.ui.gridLayout_2.addWidget(self, row, col, row_span, col_span)

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

        :param event: Event object.
        :return: None
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

    :return: None
    """
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("fusion"))
    widget = FeatureFinder()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_gui()
