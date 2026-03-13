import logging
import math
import os
import sys
from typing import Any

import cv2
import numpy as np
import yaml
from PySide6.QtCore import QRectF, Slot, QSignalBlocker
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QSizePolicy, QApplication, QWidget, QStyleFactory,
                               QFileDialog, QMessageBox, QSlider, QSpinBox, QDoubleSpinBox, QAbstractSpinBox)

from feature_finder import resources
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
        self.drawn_image: np.ndarray = np.array([])

        # Startup routines for GUI
        startup_image = cv2.imread(os.path.join(next(iter(resources.__path__)), "sample.png"), cv2.IMREAD_UNCHANGED)
        self.detector: DetectionBase = DetectionBase(startup_image)
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
        # Slider(s)
        for slider in self.findChildren(QSlider):
            slider_name = slider.objectName().lower()
            slider.sliderReleased.connect(self._update_image)
            if ("_max" in slider_name or "_min" in slider_name) and "crosshair" not in slider_name:
                slider.valueChanged.connect(self._change_range_slider_or_spin)
            else:
                slider.valueChanged.connect(self._change_slider)

        # Spin box(es)
        for spin_box in self.findChildren(QAbstractSpinBox):
            spin_box_name = spin_box.objectName().lower()
            if ("_max" in spin_box_name or "_min" in spin_box_name) and "crosshair" not in spin_box_name:
                spin_box.lineEdit().returnPressed.connect(self._change_range_slider_or_spin)

        spin_box_to_property_map = {
            self.ui.circularity_spin: type(self).elliptical_circularity_min.fget.__name__,
            self.ui.clahe_clip_limit_spin: type(self).contrast_clahe_clip_limit.fget.__name__,
            self.ui.clahe_grid_size_spin: type(self).contrast_clahe_grid_size.fget.__name__,
            self.ui.crosshair_max_slope_spin: type(self).crosshair_max_slope.fget.__name__,
            self.ui.crosshair_min_length_spin: type(self).crosshair_min_length.fget.__name__,
            self.ui.crosshair_distance_spin: type(self).crosshair_max_line_gap.fget.__name__,
            self.ui.gauss_blur_spin: type(self).edge_gauss_blur_kernel.fget.__name__,
            self.ui.crosshair_hough_threshold_spin: type(self).crosshair_hough_threshold.fget.__name__,
            self.ui.pixel_threshold_spin: type(self).edge_pixel_threshold.fget.__name__,
            self.ui.top_hat_kernel_spin: type(self).contrast_top_hat_kernel.fget.__name__,
            self.ui.arch_u_score_spin: type(self).arch_min_u_score.fget.__name__,
            self.ui.winsor_percentile_spin: type(self).noise_winsor_percentile.fget.__name__
        }
        for spin_box in spin_box_to_property_map:
            spin_box.lineEdit().returnPressed.connect(
                lambda sb=spin_box, attr=spin_box_to_property_map[spin_box]: self._change_spin(sb, attr))

        # Button(s) / Check box(es)
        self.ui.arch_fit_check.clicked.connect(self._click_enable_arch_fitting)
        self.ui.crosshair_fit_check.clicked.connect(self._click_enable_crosshair_fitting)
        self.ui.elliptical_fit_check.clicked.connect(self._click_enable_elliptical_fitting)
        self.ui.file_path_browse_button.clicked.connect(lambda: self._click_import_file(browser=True))
        self.ui.rect_fit_check.clicked.connect(self._click_enable_rectangular_fitting)
        self.ui.reduce_noise_check.clicked.connect(self._click_reduce_noise)
        self.ui.save_image_button.clicked.connect(self._click_save_drawing)

        # Entry box(es)
        self.ui.file_path_entry.returnPressed.connect(self._click_import_file)

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
            if "circularity" in spin_box.objectName().lower() or "u_score" in spin_box.objectName().lower():
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

    def _check_for_setting_dif(self, prop: property, settings_handle) -> bool:
        """
        Check for difference in settings
        """
        property_name = prop.fget.__name__
        property_value = prop.fget(self)

        if property_value != settings_handle.__getattribute__(property_name):
            settings_handle.__setattr__(property_name, property_value)
            return True
        else:
            return False

    def _click_import_file(self, browser: bool = False):
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

        if browser:
            # Set up the file browser dialog box
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Open File")
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            file_dialog.setViewMode(QFileDialog.ViewMode.Detail)

            # Execute the file browser dialog box
            if file_dialog.exec():
                selected_file = file_dialog.selectedFiles()[0]
                image_array = import_image(selected_file)

                # Update UI
                self.ui.file_path_entry.setText(selected_file)
            else:
                image_array = np.array([])
        else:
            image_array = import_image(self.ui.file_path_entry.text())

        if image_array.size > 0:

            # Store the image array
            self._raw_image = image_array
            self.drawn_image = convert_color_bit(image_array, color_channels=3, out_bit_depth=8)

            # Update UI
            self.ui.classic_tab.setEnabled(True)
            self.ui.edge_detection_frame.setEnabled(True)
            self.ui.feature_fitting_frame.setEnabled(True)
            self.ui.edge_detection_tabs.setEnabled(True)
            self.ui.save_image_button.setEnabled(True)

            self.ui.arch_tab.setEnabled(self.ui.arch_fit_check.isChecked())
            self.ui.canny_tab.setEnabled(self.ui.canny_edge_check.isChecked())
            self.ui.crosshair_tab.setEnabled(self.ui.crosshair_fit_check.isChecked())
            self.ui.elliptical_tab.setEnabled(self.ui.elliptical_fit_check.isChecked())
            self.ui.noise_contrast_tab.setEnabled(self.ui.reduce_noise_check.isChecked())
            self.ui.noise_normalize_tab.setEnabled(self.ui.reduce_noise_check.isChecked())
            self.ui.rect_tab.setEnabled(self.ui.rect_fit_check.isChecked())

            if self.ui.arch_fit_check.isChecked(): self._click_enable_arch_fitting()
            if self.ui.canny_edge_check.isChecked(): self._click_canny_edge()
            if self.ui.crosshair_fit_check.isChecked(): self._click_enable_crosshair_fitting()
            if self.ui.elliptical_fit_check.isChecked(): self._click_enable_elliptical_fitting()
            if self.ui.invert_image_check.isChecked(): self._update_image()
            if self.ui.rect_fit_check.isChecked(): self._click_enable_rectangular_fitting()
            if self.ui.reduce_noise_check.isChecked(): self._click_reduce_noise()
            if self.ui.show_processed_image_check.isChecked(): self._update_image()

            # Import detector object
            self.detector = DetectionBase(self._raw_image)

            # Update the stream window to show imported image
            self._update_image()

    def _click_enable_arch_fitting(self):
        """
        Enable controls if fitting is enabled.

        :return: None
        """
        self.ui.arch_tab.setEnabled(self.ui.arch_fit_check.isChecked())
        self._update_image()

    def _click_enable_crosshair_fitting(self):
        """
        Enable controls if fitting is enabled.

        :return: None
        """
        self.ui.crosshair_tab.setEnabled(self.ui.crosshair_fit_check.isChecked())
        self._update_image()

    def _click_enable_elliptical_fitting(self):
        """
        Enable controls if fitting is enabled.

        :return: None
        """
        self.ui.elliptical_tab.setEnabled(self.ui.elliptical_fit_check.isChecked())
        self._update_image()

    def _click_enable_rectangular_fitting(self):
        """
        Enable controls if fitting is enabled.

        :return: None
        """
        self.ui.rect_tab.setEnabled(self.ui.rect_fit_check.isChecked())
        self._update_image()

    def _click_reduce_noise(self):
        """
        Enable controls if noise handling is enabled.

        :return: None
        """
        self.ui.noise_normalize_tab.setEnabled(self.ui.reduce_noise_check.isChecked())
        self.ui.noise_contrast_tab.setEnabled(self.ui.reduce_noise_check.isChecked())
        self._update_image()

    def _click_canny_edge(self):
        """
        Enable controls if Canny edging is enabled.

        :return: None
        """
        self.ui.canny_tab.setEnabled(self.ui.reduce_noise_check.isChecked())
        self._update_image()

    def _click_save_drawing(self):
        """
        Save the current drawn image to a file.
        
        :return: None
        """
        # Update UI
        self.ui.save_status_label.setText("Saving...")
        self.ui.save_status_label.setStyleSheet("color: black;")

        # Attempt to save the image
        if self._raw_image.size > 0 and self.drawn_image.size > 0:

            # Check the image path
            file_path = os.path.join(os.path.dirname(self.ui.file_path_entry.toPlainText()), "ff_drawing.png")
            checked_path = check_path(file_path)
            checked_dir = os.path.dirname(checked_path)

            # Save the image with cv2
            try:
                if not os.path.isdir(checked_dir):
                    os.makedirs(checked_dir)
                cv2.imwrite(checked_path, self.drawn_image)

            except PermissionError:  # handle permissions erro case
                checked_dir = os.getcwd()
                file_path = os.path.join(os.getcwd(), os.path.basename(file_path))
                self._logger.warning(f"Lacking write permissions for this directory. Saving locally instead.")
                cv2.imwrite(file_path, self.drawn_image)

            finally:  # log
                self._logger.info(f"Image saved at: {file_path}")

            # Save JSON settings file
            json_path = os.path.join(checked_dir, "detection_settings.json")
            self.detector.settings.to_file(json_path)
            self._logger.info(f"Detection settings file saved to: {json_path}")

            # Save YAML features file
            yaml_path = os.path.join(checked_dir, "found_features.yaml")
            with open(yaml_path, "w") as f:
                features_dict = {}
                for idx, feature in enumerate(self.detector.found_features):
                    feature_data = vars(feature).copy()
                    if isinstance(feature_data.get('centroid'), tuple):
                        feature_data['centroid'] = list(feature_data['centroid'])  # convert tuples to lists
                    features_dict[f"feature_{idx}"] = feature_data
                yaml.dump(features_dict, f, default_flow_style=False, sort_keys=False)
            self._logger.info(f"Found features file saved to: {yaml_path}")

            # Update UI
            self.ui.save_status_label.setText("Saved!")
            self.ui.save_status_label.setStyleSheet("color: black;")

        else:
            # Log and update
            self.ui.save_status_label.setText("No data!")
            self.ui.save_status_label.setStyleSheet("color: red;")
            self._logger.warning(f"No data to save.")

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

        # Define local variables
        edge_settings = self.detector.settings.edge_detection
        feature_settings = self.detector.settings.feature_fitting
        noise_settings = self.detector.settings.edge_detection.noise_handling

        # Set sliders ----------------------------------------------------------------------------------------------- #

        ## Feature fitting sliders
        set_widget_value(self.ui.arch_max_size_slider, feature_settings.arch.arch_size_range)
        set_widget_value(self.ui.arch_u_score_slider, feature_settings.arch.arch_min_u_score)
        set_widget_value(self.ui.circularity_spin, feature_settings.ellipse.elliptical_circularity_min,
                         float_slider_factor=100)
        set_widget_value(self.ui.crosshair_distance_slider, feature_settings.crosshair.crosshair_max_line_gap)
        set_widget_value(self.ui.crosshair_hough_threshold_spin, feature_settings.crosshair.crosshair_hough_threshold)
        set_widget_value(self.ui.crosshair_max_slope_spin, feature_settings.crosshair.crosshair_max_slope)
        set_widget_value(self.ui.crosshair_min_length_spin, feature_settings.crosshair.crosshair_min_length)
        set_widget_value(self.ui.elliptical_max_size_slider, feature_settings.ellipse.elliptical_size_range)
        set_widget_value(self.ui.rect_max_size_slider, feature_settings.rectangle.rectangular_size_range)

        ## Classic detection sliders
        set_widget_value(self.ui.feature_max_size_slider, edge_settings.contour_size_range)
        set_widget_value(self.ui.gauss_blur_slider, edge_settings.edge_gauss_blur_kernel)
        set_widget_value(self.ui.pixel_threshold_slider, edge_settings.edge_pixel_threshold)

        ## Other detection sliders
        set_widget_value(self.ui.canny_max_slider, edge_settings.edge_canny_range)
        set_widget_value(self.ui.clahe_clip_limit_slider, noise_settings.contrast_clahe_clip_limit,
                         float_slider_factor=100)
        set_widget_value(self.ui.clahe_grid_size_slider, noise_settings.contrast_clahe_grid_size)
        set_widget_value(self.ui.noise_percentile_max_slider, noise_settings.noise_percentile_range)
        set_widget_value(self.ui.top_hat_kernel_slider, noise_settings.contrast_top_hat_kernel)
        set_widget_value(self.ui.winsor_percentile_slider, noise_settings.noise_winsor_percentile)

        # Set check boxes ------------------------------------------------------------------------------------------- #

        ## Feature fitting check boxes
        self.ui.arch_fit_check.setChecked(feature_settings.rectangle.flag_fit_feature)
        self.ui.crosshair_fit_check.setChecked(feature_settings.crosshair.flag_fit_feature)
        self.ui.elliptical_fit_check.setChecked(feature_settings.ellipse.flag_fit_feature)
        self.ui.rect_fit_check.setChecked(feature_settings.rectangle.flag_fit_feature)

        ## Edge detection check boxes
        self.ui.canny_edge_check.setChecked(edge_settings.flag_canny_edged)
        self.ui.invert_image_check.setChecked(edge_settings.flag_invert_image)
        self.ui.reduce_noise_check.setChecked(edge_settings.flag_reduce_noise)
        self.ui.show_processed_image_check.setChecked(edge_settings.flag_show_processed)

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

    def _update_referenced_settings(self):

        """
        Update the internally tracked settings based on UI.

        :return: None
        """
        # Edge detection settings ----------------------------------------------------------------------------------- #

        ## Classic
        self.detector.settings.edge_detection.flag_invert_image = self.ui.invert_image_check.isChecked()
        self.detector.settings.edge_detection.flag_show_processed = self.ui.show_processed_image_check.isChecked()
        # self.detector.settings.edge_detection.gauss_blur_kernel = self.gauss_blur_kernel
        # self.detector.settings.edge_detection.pixel_threshold = self.pixel_threshold
        # self.detector.settings.edge_detection.size_range = self.feature_size_range

        ## Canny edge
        # self.detector.settings.edge_detection.canny_edge_range = self.canny_edge_range
        self.detector.settings.edge_detection.flag_canny_edged = self.ui.canny_edge_check.isChecked()

        ## Noise: Normalization
        self.detector.settings.edge_detection.flag_reduce_noise = self.ui.reduce_noise_check.isChecked()
        # self.detector.settings.edge_detection.noise_handling.percentile_range = self.noise_percentile_range
        # self.detector.settings.edge_detection.noise_handling.winsor_percentile = self.winsor_percentile

        ## Noise: Contrast boost
        # self.detector.settings.edge_detection.noise_handling.tophat_ksize = self.contrast_top_hat_kernel
        # self.detector.settings.edge_detection.noise_handling.clahe_grid_size = self.contrast_clahe_grid_size
        # self.detector.settings.edge_detection.noise_handling.clahe_clip = self.contrast_clahe_clip_limit

        # Feature fitting settings ---------------------------------------------------------------------------------- #

        ## Crosshair
        self.detector.settings.feature_fitting.crosshair.flag_fit_feature = self.ui.crosshair_fit_check.isChecked()
        # self.detector.settings.feature_fitting.crosshair.hough_threshold = self.crosshair_hough_threshold
        # self.detector.settings.feature_fitting.crosshair.max_line_gap = self.crosshair_distance
        # self.detector.settings.feature_fitting.crosshair.max_slope = self.crosshair_max_slope
        # self.detector.settings.feature_fitting.crosshair.min_length = self.crosshair_min_length

        ## Elliptical
        # self.detector.settings.feature_fitting.ellipse.circularity_min = self.circularity_min
        self.detector.settings.feature_fitting.ellipse.flag_fit_feature = self.ui.elliptical_fit_check.isChecked()
        # self.detector.settings.feature_fitting.ellipse.size_range = self.elliptical_size_range

        ## Rectangular
        self.detector.settings.feature_fitting.rectangle.flag_fit_feature = self.ui.rect_fit_check.isChecked()
        # self.detector.settings.feature_fitting.rectangle.size_range = self.rectangular_size_range

        ## Arched
        self.detector.settings.feature_fitting.arch.flag_fit_feature = self.ui.arch_fit_check.isChecked()
        # self.detector.settings.feature_fitting.arch.min_u_score = self.arch_u_score
        # self.detector.settings.feature_fitting.rectangle.size_range = self.arch_size_range

    def _update_image(self):
        """
        Update the drawn image internally.
        
        :return: None
        """

        if self._raw_image.size > 0:

            # Define local variables
            edge_settings = self.detector.settings.edge_detection
            noise_settings = self.detector.settings.edge_detection.noise_handling

            # Determine preprocessing
            if self.ui.reduce_noise_check.isChecked():
                update1 = self._check_for_setting_dif(type(self).noise_percentile_range, noise_settings)
                update2 = self._check_for_setting_dif(type(self).noise_winsor_percentile, noise_settings)
                update3 = self._check_for_setting_dif(type(self).contrast_top_hat_kernel, noise_settings)
                update4 = self._check_for_setting_dif(type(self).contrast_clahe_grid_size, noise_settings)
                update5 = self._check_for_setting_dif(type(self).contrast_clahe_clip_limit, noise_settings)
                update_noise = any([update1, update2, update3, update4, update5,
                                    edge_settings.flag_reduce_noise != True])
            else:
                update_noise = False
            if self.ui.invert_image_check.isChecked() != edge_settings.flag_invert_image and not update_noise:
                update_noise = True
            update_gauss = self._check_for_setting_dif(type(self).edge_gauss_blur_kernel, edge_settings) or update_noise
            update_threshold = self._check_for_setting_dif(type(self).edge_pixel_threshold,
                                                           edge_settings) or update_gauss
            if self.ui.crosshair_fit_check.isChecked():
                crosshair_settings = self.detector.settings.feature_fitting.crosshair
                update1 = self._check_for_setting_dif(type(self).crosshair_hough_threshold, crosshair_settings)
                update2 = self._check_for_setting_dif(type(self).crosshair_min_length, crosshair_settings)
                update3 = self._check_for_setting_dif(type(self).crosshair_max_line_gap, crosshair_settings)
                update_hough = update_threshold or any([update1, update2, update3])
            else:
                update_hough = False

            # Find and draw contours/detections
            self._update_referenced_settings()
            self.detector.detect_features(update_threshold=update_threshold,
                                          update_gauss=update_gauss,
                                          update_hough=update_hough)

            # Log settings
            self._logger.info(f"Updated image with settings:\n{self.detector.settings.__dict__}")

            # Update stream window
            self.drawn_image = self.detector.display_image
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
    def arch_size_range(self) -> tuple[int, int]:
        """
        Selected range of detected arch sizes.

        :return: Arch size range.
        """
        return int(self.ui.arch_min_size_spin.value()), int(self.ui.arch_max_size_spin.value())

    @property
    def arch_min_u_score(self) -> float:
        """
        Selected U-score (archness) minimum for blob detection.

        :return: U-score minimum.
        """
        return float(self.ui.arch_u_score_spin.value())

    @property
    def contour_size_range(self) -> tuple[int, int]:
        """
        Selected range of detected contour (rectangles or crosshairs) sizes.

        :return: Contour size range.
        """
        return int(self.ui.feature_min_size_spin.value()), int(self.ui.feature_max_size_spin.value())

    @property
    def contrast_clahe_grid_size(self) -> int:
        """
        Selected CLAHE kernel/grid size.

        :return: CLAHE kernel/grid size.
        """
        return int(self.ui.clahe_grid_size_spin.value())

    @property
    def contrast_clahe_clip_limit(self) -> float:
        """
        Selected CLAHE clip limit.

        :return: CLAHE clip limit.
        """
        return float(self.ui.clahe_clip_limit_spin.value())

    @property
    def contrast_top_hat_kernel(self) -> int:
        """
        Selected top hat kernel for small features.

        :return: Top hat kernel.
        """
        return int(self.ui.top_hat_kernel_spin.value())

    @property
    def crosshair_hough_threshold(self) -> int:
        """
        Selected Hough line threshold value.

        :return: Threshold value.
        """
        return int(self.ui.crosshair_hough_threshold_spin.value())

    @property
    def crosshair_min_length(self) -> int:
        """
        Selected allowable crosshair minimum length.

        :return: Crosshair minimum length
        """
        return int(self.ui.crosshair_min_length_spin.value())

    @property
    def crosshair_max_slope(self) -> float:
        """
        Selected crosshair slope definition rotation.

        :return: Crosshair slope rotation.
        """
        return float(self.ui.crosshair_max_slope_spin.value())

    @property
    def crosshair_max_line_gap(self) -> int:
        """
        Selected separation distance between crosshairs.

        :return: Distance between crosshairs.
        """
        return int(self.ui.crosshair_distance_spin.value())

    @property
    def edge_canny_range(self) -> tuple[int, int]:
        """
        Selected range of Canny edge limits.

        :return: Canny edge limits.
        """
        return int(self.ui.canny_min_spin.value()), int(self.ui.canny_max_spin.value())

    @property
    def edge_gauss_blur_kernel(self) -> int:
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
    def edge_pixel_threshold(self) -> int:
        """
        Selected pixel threshold value.

        :return: Threshold value.
        """
        return int(self.ui.pixel_threshold_spin.value())

    @property
    def elliptical_circularity_min(self) -> float:
        """
        Selected circularity minimum for blob detection.

        :return: Circularity minimum.
        """
        return float(self.ui.circularity_spin.value())

    @property
    def elliptical_size_range(self) -> tuple[int, int]:
        """
        Selected range of detected elliptical sizes.

        :return: Elliptical size range.
        """
        return int(self.ui.elliptical_min_size_spin.value()), int(self.ui.elliptical_max_size_spin.value())

    @property
    def noise_percentile_range(self) -> tuple[int, int]:
        """
        Selected range of noise percentiles.

        :return: Noise percentile range.
        """
        return int(self.ui.noise_percentile_min_spin.value()), int(self.ui.noise_percentile_max_spin.value())

    @property
    def noise_winsor_percentile(self) -> int:
        """
        Selected winsor percentile value.

        :return: Winsor percentile value.
        """
        return int(self.ui.winsor_percentile_spin.value())

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
    def rectangular_size_range(self) -> tuple[int, int]:
        """
        Selected range of detected rectangular sizes.

        :return: Rectangular size range.
        """
        return int(self.ui.rect_min_size_spin.value()), int(self.ui.rect_max_size_spin.value())


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
