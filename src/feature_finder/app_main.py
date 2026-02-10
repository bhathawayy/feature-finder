import logging
import math
import os
import sys
import yaml
from typing import Any, Callable

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Slot, QSignalBlocker
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QSizePolicy, QApplication, QWidget, QStyleFactory,
                               QFileDialog, QMessageBox, QSlider, QSpinBox, QDoubleSpinBox, QAbstractSpinBox)

from feature_finder.data_objects import DefaultSettings
from feature_finder.data_processing import convert_color_bit, check_path, DetectionBase
from feature_finder.interface.ui_form import Ui_featureFinder


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
        for slider in self.findChildren(QSlider):
            slider_name = slider.objectName().lower()
            slider.sliderReleased.connect(self._update_image)
            if ("_max" in slider_name or "_min" in slider_name) and "crosshair" not in slider_name:
                slider.valueChanged.connect(self._change_range_slider_or_spin)
            else:
                slider.valueChanged.connect(self._change_slider)

        # Spin boxes
        for spin_box in self.findChildren(QAbstractSpinBox):
            spin_box_name = spin_box.objectName().lower()
            if ("_max" in spin_box_name or "_min" in spin_box_name) and "crosshair" not in spin_box_name:
                spin_box.lineEdit().returnPressed.connect(self._change_range_slider_or_spin)

        spin_box_to_property_map = {
            self.ui.circularity_spin: type(self).circularity_min.fget.__name__,
            self.ui.crosshair_min_length_spin: type(self).crosshair_min_length.fget.__name__,
            self.ui.crosshair_slope_tilt_spin: type(self).crosshair_slope_tilt.fget.__name__,
            self.ui.distance_interval_spin: type(self).crosshair_distance.fget.__name__,
            self.ui.gauss_blur_spin: type(self).gauss_blur_kernel.fget.__name__,
            self.ui.hough_threshold_spin: type(self).crosshair_hough_threshold.fget.__name__,
            self.ui.pixel_threshold_spin: type(self).pixel_threshold.fget.__name__
        }
        for spin_box in spin_box_to_property_map:
            spin_box.lineEdit().returnPressed.connect(
                lambda: self._change_spin(spin_box, spin_box_to_property_map[spin_box]))

        # Buttons / Check boxes
        self.ui.elliptical_fit_check.clicked.connect(self._click_enable_circle_fitting)
        self.ui.crosshair_fit_check.clicked.connect(self._click_enable_crosshair_fitting)
        self.ui.file_path_browse_button.clicked.connect(self._click_browse_file)
        self.ui.rect_fit_check.clicked.connect(self._click_enable_rectangular_fitting)
        self.ui.save_image_button.clicked.connect(self._click_save_drawing)

    def _change_range_slider_or_spin(self):
        """
        Update the toggled widget and any bonded widgets.

        :return: None
        """
        # Define local variables
        toggled_widget, slider, spin_box = self._get_bonded_widget()
        if toggled_widget is None:
            return
        widget_name = toggled_widget.objectName().lower()
        new_val = toggled_widget.value()

        # Ensure range sliders react correctly to each other
        slider_min = self.ui.__getattribute__(slider.objectName().lower().replace("_max", "_min"))
        slider_max = self.ui.__getattribute__(slider.objectName().lower().replace("_min", "_max"))
        if "max" in widget_name:
            min_val = slider_min.value()
            if new_val <= min_val:
                new_val = min_val + 1
                with QSignalBlocker(toggled_widget):
                    toggled_widget.setValue(new_val)
        else:
            max_val = slider_max.value()
            if new_val >= max_val:
                new_val = max_val - 1
                with QSignalBlocker(toggled_widget):
                    toggled_widget.setValue(new_val)

        # Set bonded widget to new value
        if "_slider" in widget_name:
            with QSignalBlocker(spin_box):
                spin_box.setValue(new_val)
        else:
            slider.setValue(new_val)

        # Update image for any changes
        self._update_image()

    def _change_slider(self):
        """
        Action for changing a slider widget.

        :return: None
        """
        toggled_widget, slider, spin_box = self._get_bonded_widget()
        new_val = toggled_widget.value()
        with QSignalBlocker(spin_box):
            if "circularity" in spin_box.objectName().lower():
                new_val *= 1 / 100
            spin_box.setValue(new_val)
        self._update_image()

    def _change_spin(self, toggled_widget: QSpinBox, attribute_name: str):
        """
        Action for changing a spin-box widget.

        :param toggled_widget: Toggled spin-box (self.sender() doesn't work with lambda)
        :param attribute_name: Name of property call.
        :return: None
        """
        widget_name = toggled_widget.objectName().lower()
        slider = self.ui.__getattribute__(widget_name.replace("_spin", "_slider"))
        slider.setValue(int(getattr(self, attribute_name)))

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
            rgb_image = np.array([])

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
                # Store the image array
                self._raw_image = image_array
                self.drawn_image = convert_color_bit(image_array, color_channels=3, out_bit_depth=8)

                # Update UI
                self.ui.controls_frame.setEnabled(True)
                self.ui.file_path_entry.setText(selected_file)
                self.ui.fitting_frame.setEnabled(True)
                self.ui.save_image_button.setEnabled(True)

                # TODO: Init the appropriate detector
                self.detector = DetectionBase(self._raw_image)

                # Update the stream window to show imported image
                self._update_image()

    def _click_enable_circle_fitting(self):
        """
        Enable controls if fitting is enabled.

        :return: None
        """
        self.ui.elliptical_tab.setEnabled(self.ui.elliptical_fit_check.isChecked())
        self._update_image()

    def _click_enable_crosshair_fitting(self):
        """
        Enable controls if fitting is enabled.

        :return: None
        """
        self.ui.crosshair_tab.setEnabled(self.ui.crosshair_fit_check.isChecked())
        self._update_image()

    def _click_enable_rectangular_fitting(self):
        """
        Enable controls if fitting is enabled.

        :return: None
        """
        self.ui.rect_tab.setEnabled(self.ui.rect_fit_check.isChecked())
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
                with open(os.path.join(checked_dir, "found_features.yaml"), "w") as f:
                    features_dict = {}
                    for idx, feature in enumerate(self.detector.found_features):
                        feature_data = vars(feature).copy()
                        if isinstance(feature_data.get('centroid'), tuple):
                            feature_data['centroid'] = list(feature_data['centroid'])  # convert tuples to lists
                        features_dict[f"feature_{idx}"] = feature_data
                    yaml.dump(features_dict, f, default_flow_style=False, sort_keys=False)
                self._logger.info(f"YAML files saved to: {file_path}")

    def _dialog_and_log(self, message: str, button: int = 0, level: int = 0,
                        err_handle: Exception | None = None) -> int:
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

    def _get_bonded_widget(self) -> tuple[
        QWidget | QSlider | QSpinBox | QDoubleSpinBox, QSlider, QSpinBox | QDoubleSpinBox]:
        """
        Get the bonded widget i.e. slider if spin-box, or spin-box if slider.

        :return: (1) Toggled widget, (2) slider handle, and (3) spin-box handle.
        """
        # Handle case where sender is a line edit within a spin box
        toggled_widget = self.sender()
        if hasattr(toggled_widget, 'parent') and isinstance(toggled_widget.parent(), QSpinBox):
            toggled_widget = toggled_widget.parent()

        # Get widget handles
        widget_name = toggled_widget.objectName().lower()
        slider = self.ui.__getattribute__(widget_name.replace("_spin", "_slider"))
        spin_box = self.ui.__getattribute__(widget_name.replace("_slider", "_spin"))

        return toggled_widget, slider, spin_box

    def _set_defaults(self):
        """
        Set default values for UI elements.
        
        :return: None
        """

        def set_widget_value(widget_family_member: QWidget, setting_handle: Any, float_slider_factor: float = 1):

            def get_target_name(widget_name: str, target: str, option: str) -> str:
                return widget_name if target in widget_name else widget_name.replace(option, target)

            def set_value(widget_name: str, new_value: Any):
                if "slider" in widget_name:
                    self.ui.__getattribute__(widget_name).setValue(new_value * float_slider_factor)
                else:
                    self.ui.__getattribute__(widget_name).setValue(new_value)

            def set_min_and_max(widget_name: str):
                set_value(get_target_name(widget_name, "min", "max"), setting_handle[0])
                set_value(get_target_name(widget_name, "max", "min"), setting_handle[1])

            if isinstance(setting_handle, tuple):
                set_min_and_max(get_target_name(widget_family_member.objectName(), "slider", "spin"))
                set_min_and_max(get_target_name(widget_family_member.objectName(), "spin", "slider"))
            else:
                set_value(get_target_name(widget_family_member.objectName(), "slider", "spin"), setting_handle)
                set_value(get_target_name(widget_family_member.objectName(), "spin", "slider"), setting_handle)

        settings = self.detection_settings

        set_widget_value(self.ui.distance_interval_slider, settings.crosshair_distance)
        set_widget_value(self.ui.feature_max_size_slider, settings.feature_size_range)
        set_widget_value(self.ui.gauss_blur_slider, settings.gauss_blur_kernel)
        set_widget_value(self.ui.pixel_threshold_slider, settings.pixel_threshold)
        set_widget_value(self.ui.elliptical_max_size_slider, settings.feature_size_range)
        set_widget_value(self.ui.circularity_spin, settings.circularity_min, 100)
        set_widget_value(self.ui.hough_threshold_spin, settings.crosshair_hough_threshold)
        set_widget_value(self.ui.crosshair_min_length_spin, settings.crosshair_min_length)
        set_widget_value(self.ui.crosshair_slope_tilt_spin, settings.crosshair_slope_tilt)
        set_widget_value(self.ui.rect_max_size_slider, settings.feature_size_range)

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
            update_next = self.detector.apply_gauss_blur(
                self.gauss_blur_kernel,
                update=(self.gauss_blur_kernel != self.detection_settings.gauss_blur_kernel)
            )
            update_next = self.detector.apply_threshold(
                self.pixel_threshold,
                update=(update_next or self.pixel_threshold != self.detection_settings.pixel_threshold)
            )
            if self.ui.crosshair_fit_check.isChecked():
                self.detector.apply_hough_transform(
                    self.crosshair_hough_threshold,
                    self.crosshair_distance,
                    self.crosshair_min_length,
                    update=(update_next or self.crosshair_hough_threshold !=
                            self.detection_settings.crosshair_hough_threshold or
                            self.crosshair_min_length != self.detection_settings.crosshair_min_length))

            # Find and draw contours/detections
            self.detector.detect_features(
                self.feature_size_range,
                self.elliptical_size_range,
                self.rectangular_size_range,
                self.circularity_min,
                self.crosshair_slope_tilt,
                fit_ellipse=self.ui.elliptical_fit_check.isChecked(),
                fit_rect=self.ui.rect_fit_check.isChecked(),
                fit_crosshair=self.ui.crosshair_fit_check.isChecked()
            )
            self.drawn_image = self.detector.display_image

            # Update & log settings
            self._update_stored_settings()
            self._logger.info(
                f"Updated image with settings:\n"
                f"\tcircularity = {self.detection_settings.circularity_min}"
                f"\tcrosshair distance = {self.detection_settings.crosshair_distance}"
                f"\tcrosshair min. length = {self.detection_settings.crosshair_min_length}"
                f"\tcrosshair slope tilt = {self.detection_settings.crosshair_slope_tilt}"
                f"\telliptical size range = {self.detection_settings.elliptical_size_range}"
                f"\tfeature size range = {self.detection_settings.feature_size_range}"
                f"\tgauss blur kernel = {self.detection_settings.gauss_blur_kernel}"
                f"\though threshold = {self.detection_settings.crosshair_hough_threshold}"
                f"\tpixel threshold = {self.detection_settings.pixel_threshold}"
                f"\trectangular size range = {self.detection_settings.rectangular_size_range}"
            )

            # Update stream window
            self._update_stream_window()

    def _update_stored_settings(self):
        """
        Update stored settings based on UI.

        :return: None
        """
        for setting in self.detection_settings.__dict__:
            self.detection_settings.__setattr__(setting, self.__getattribute__(setting))

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
    def circularity_min(self) -> float:
        """
        Selected circularity limit for blob detection.

        :return: Circularity limit.
        """
        return float(self.ui.circularity_spin.value())

    @property
    def crosshair_hough_threshold(self) -> int:
        """
        Selected Hough line threshold value.

        :return: Threshold value.
        """
        return int(self.ui.hough_threshold_spin.value())

    @property
    def crosshair_min_length(self) -> int:
        """
        Selected allowable crosshair minimum length.

        :return: Crosshair minimum length
        """
        return int(self.ui.crosshair_min_length_spin.value())

    @property
    def crosshair_slope_tilt(self) -> float:
        """
        Selected crosshair slope definition rotation.

        :return: Crosshair slope rotation.
        """
        return float(self.ui.crosshair_slope_tilt_spin.value())

    @property
    def crosshair_distance(self) -> int:
        """
        Selected separation distance between crosshairs.

        :return: Distance between crosshairs.
        """
        return int(self.ui.distance_interval_spin.value())

    @property
    def elliptical_size_range(self) -> tuple[float, float]:
        """
        Selected range of detected elliptical sizes.

        :return: elliptical size range.
        """
        return self.ui.elliptical_min_size_slider.value(), self.ui.elliptical_max_size_slider.value()

    @property
    def feature_size_range(self) -> tuple[float, float]:
        """
        Selected range of detected contour (rectangles or crosshairs) sizes.

        :return: contour size range.
        """
        return self.ui.feature_min_size_slider.value(), self.ui.feature_max_size_slider.value()

    @property
    def gauss_blur_kernel(self) -> int:
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
    def rectangular_size_range(self) -> tuple[float, float]:
        """
        Selected range of detected elliptical sizes.

        :return: elliptical size range.
        """
        return self.ui.rect_min_size_slider.value(), self.ui.rect_max_size_slider.value()


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
