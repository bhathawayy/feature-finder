import abc
from copy import deepcopy
from itertools import combinations

import circle_fit
from scipy.optimize import fsolve
from scipy.signal import argrelmin
from sklearn.metrics import r2_score

from detection_settings import *
from processing_support import *


# Parent Classes ---------------------------------------------------------------------------------------------------- #
class DefaultSettings:
    """
    Default detection settings. Note, the area or size parameters can typically be estimated using ImageJ. Fit a
    shape to your feature then "Measure" to return the area in pxl^2.

    USE THE HELPER GUI TO CONFIGURE!
    """
    # Parameters used for detection algorithms
    blob_size: tuple = (1000, 6000)  # Expected size of fiducial [(pxl^2, pxl^2)]
    circularity_min: float = 0.8  # The closer to 1, the more "perfect" the circle is
    deviation_cutoff: int = 100  # Cutoff distance between multiple detections [pxl]
    feature_size: tuple = (10000, 30000)  # Expected size of feature (non-fiducial) [(pxl^2, pxl^2)]
    gauss: int = 11  # Gaussian blur kernel size
    hough_min_length: int = 30  # Expected length of detected (CRH) line [pxl]
    threshold: int = 30  # Explicit edge thresholding
    range_slider_max: int = 100000


class DetectionBase(ImageLoader):
    """
    Base class for image detection, extending the functionality of ImageLoader.
    """

    def __init__(self, image_dir_or_file: str | np.ndarray, is_crosshair: bool = False):
        """
        Initialize the DetectionBase with the given parameters.

        :param image_dir_or_file: Path to the image directory, a single image file, or an image array.
        :param is_crosshair: Flag to indicate if the image contains a crosshair pattern for detection.
        """
        # Init the image loader
        super().__init__(image_dir_or_file, is_crosshair=is_crosshair)

        # Init dynamic hidden class variables
        self._clocking_angle: float | None = None  # Clocking rotation of image [deg]
        self._config_file_name: str | None = None
        self._crop: bool = False  # Crop raw image to speed up processing
        self._cropped_by: tuple = (0, 0)  # Width then height [pxl]
        self._detection_info: DetectionInfo = DetectionInfo()
        self._detection_settings: DefaultSettings = DefaultSettings()
        self._detection_settings_name: str = "DefaultDetection"
        self._do_compare: bool = False
        self._image_in_progress: str = ""
        self._orientation: list = [1, 1, 0]  # Orientation of reference points [X, Y, R]
        self._pivot_point: tuple | None = None

        # Init static hidden class variables
        self._color_blob: tuple = (255, 0, 0)  # Color for rects [BGR]
        self._color_found_pt: tuple = (0, 255, 0)  # Color for text or missing point [BGR]
        self._color_missing_pt: tuple = (0, 0, 255)  # Color for text or missing point [BGR]
        self._color_rect1: tuple = (0, 155, 255)  # Color counterpart for rects [BGR]
        self._color_rect2: tuple = (155, 0, 255)  # Color counterpart for rects [BGR]
        self._color_txt: tuple = (0, 0, 255)  # Color for text or missing point [BGR]
        self._dark_roi_size: tuple = (25, 25)  # Width then height [pxl]
        self._debug_save_detection: bool = True
        self._debug_save_rois: bool = False
        self._draw_contours: bool = False
        self._draw_fits: bool = False
        self._draw_size: int = 8  # Size of drawn features [pxl]
        self._roi_cushion: int = 15  # Distance between center and CRH ROI [pxl]
        self._roi_size: tuple = (35, 55)  # Width then height [pxl]
        self._sig_fig: int = 3

        # Set editable class variables
        self.debug_mode: bool = self._debug_dir is not None and self._debug_dir != ""
        self.is_olaf: bool = False
        self.is_mtf_processed: bool = False
        self.json_settings: dict = {}

        # Load image(s)
        self.load_image()

    @property
    def clocking_angle(self):
        """
        Get the clocking angle of the image.

        :return: The clocking angle in degrees, either from the detection settings or the internal value.
        """
        return self._detection_settings.default_clock_angle if self._clocking_angle is None else self._clocking_angle

    @property
    def config_name(self) -> str | None:
        """
        Get the name of the configuration file used.

        :return: The name of the configuration file or None if not set.
        """
        return self._config_file_name

    @property
    def debug_dir(self) -> str:
        """
        Get the directory path for saving debug information.

        :return: The path to the debug directory, creating it if necessary.
        """
        # Use image directory if none was provided
        if self._debug_dir is None or self._debug_dir == "":
            debug_path = os.getcwd()
        else:
            debug_path = self._debug_dir

        try:
            # Add folder to path
            if os.path.basename(debug_path) != "Images":
                if os.path.basename(debug_path) == "MTF":
                    debug_path = os.path.join(debug_path, "Images")
                else:
                    debug_path = os.path.join(debug_path, "MTF", "Images")

            # Make directory if it doesn't exist
            if not os.path.isdir(debug_path):
                os.makedirs(debug_path)

        except PermissionError:
            debug_path = os.path.join(os.getcwd(), "Debug")
            warnings.warn(f"Lacking write permissions. Using local instead: {debug_path}")

        # Set internal reference
        self._debug_dir = debug_path

        return self._debug_dir

    @property
    def detection_settings_name(self) -> str:
        """
        Get the name of the detection settings currently in use.

        :return: The name of the detection settings.
        """
        return self._detection_settings_name

    def do_detection(self) -> dict:
        """
        Perform detection on all images in the image_info_dict.

        This method iterates through each image, resets detection information,
        calls the detection algorithm, and saves the results.
        :return: the dictionary of image info
        """
        for image in self.image_info_dict:
            # Reset detection info
            self._detection_info = DetectionInfo()
            self._image_in_progress = image

            # Call detection algorithm
            self._detect()

            # Save to internal dict
            self.image_info_dict[image].detections = self._detection_info

        return self.image_info_dict

    @property
    def eye_side(self) -> str | None:
        """
        Get the side of the eye for the image currently in progress.

        :return: The side of the eye or None if the image is not found in the dictionary.
        """
        # TODO: do i need a setter for this? for non-standard name or array input
        if self._image_in_progress in self.image_info_dict.keys():
            return self.image_info_dict[self._image_in_progress].info.side
        else:
            return None

    @property
    def image_color8(self) -> np.ndarray:
        """
        Get the color image array in 8-bit format.

        :return: The color image array or an empty array if the image is not found.
        """
        if self._image_in_progress in self.image_info_dict.keys():
            return self.image_info_dict[self._image_in_progress].arrays.color8
        else:
            return np.array([])

    @property
    def image_date(self) -> str | None:
        """
        Get the date associated with the image currently in progress.

        :return: The date of the image or None if not found.
        """
        if self._image_in_progress in self.image_info_dict.keys():
            return self.image_info_dict[self._image_in_progress].info.date
        else:
            return None

    @property
    def image_mono8(self) -> np.ndarray:
        """
        Get the monochrome image array in 8-bit format.

        :return: The monochrome image array or an empty array if the image is not found.
        """
        if self._image_in_progress in self.image_info_dict.keys():
            return self.image_info_dict[self._image_in_progress].arrays.mono8
        else:
            return np.array([])

    @property
    def image_mono16(self) -> np.ndarray:
        """
        Get the monochrome image array in 16-bit format.

        :return: The monochrome image array or an empty array if the image is not found.
        """
        if self._image_in_progress in self.image_info_dict.keys():
            return self.image_info_dict[self._image_in_progress].arrays.mono16
        else:
            return np.array([])

    @property
    def pivot_point(self) -> tuple:
        """
        Get the pivot point for image orientation correction.

        :return: The pivot point coordinates or default values if not set.
        """
        if self._pivot_point is None:
            if self._detection_settings.default_pivot_point is None:
                image_shape = self.image_info_dict[self._image_in_progress].arrays.mono8.shape
                return int(image_shape[1] / 2 + self._cropped_by[0]), int(image_shape[0] / 2 + self._cropped_by[1])
            else:
                return self._detection_settings.default_pivot_point
        else:
            return self._pivot_point

    @property
    def test_system(self) -> str | None:
        """
        Get the test system acronym associated with the image currently in progress.

        :return: The test system or None if not found.
        """
        if self._image_in_progress in self.image_info_dict.keys():
            return self.image_info_dict[self._image_in_progress].info.test_system
        else:
            return None

    @staticmethod
    def _angle_to_horizontal_line(slope: float) -> float:
        """
        Calculate angle (clock angle) between line with a given slope and the horizontal plane.
        :param slope: Slope of line to compare horizontal to.
        :return: Clock angle [degrees].
        """
        # Calculate the angle in radians between the line and the horizontal
        angle_rad = np.arctan(slope)

        # Convert the angle to degrees
        angle_deg = np.rad2deg(angle_rad)

        return angle_deg

    def _after_detect(self):
        """
        Post-processing after detection, including comparison with reference points and saving results.
        """
        # Compare reference to detected points
        if self._do_compare:
            full_reference = self._compare_to_reference()
        else:
            full_reference = None

        # Save detections to image info dictionary
        self._detection_info.settings = self._detection_settings
        self._detection_info.crop_size_wh = self._cropped_by

        # Draw detection contours on image
        if self.debug_mode:
            if full_reference is not None:
                self._draw_reference(full_reference)
            self._draw_detection()
            self._detection_info.debug_image = self.image_color8

            # Save detection image
            if self.debug_mode:
                if self._is_olaf:
                    file_name_no_ext = "_".join(self._image_in_progress.split("\\")[-4:-1])
                else:
                    file_name = os.path.basename(self._image_in_progress)
                    file_name_no_ext = os.path.splitext(file_name)[0]

                if self._debug_save_detection:
                    save_image(os.path.join(self.debug_dir, f"{file_name_no_ext}_DETECTION.png"), self.image_color8,
                               overwrite=True)
                if self._debug_save_rois:
                    for p in self._detection_info.pois:
                        this_poi = self._detection_info.pois[p]
                        if not this_poi.is_fiducial and this_poi.shape.upper() != "NONE":
                            for roi in this_poi.mtf_rois.__annotations__.keys():
                                this_roi = eval(f"this_poi.mtf_rois.{roi}")
                                save_image(os.path.join(self.debug_dir,
                                                        f"{file_name_no_ext}_{p}{roi.upper()[0]}_ROI.png"),
                                           this_roi.array, overwrite=True)

            # # Show image (for debugging)
            # from MountOlympus.Processing.helper import show_image
            # show_image(self.image_color8)

    def _before_detect(self) -> np.ndarray:
        """
        Pre-processing before detection, setting detection settings and preprocessing the image.

        :return: Preprocessed image array.
        """
        # Set detection settings
        self._set_detection_settings()

        # Pre-process image
        preprocessed_image = self._preprocess_image()

        return preprocessed_image

    def _check_config_file(self, config_path: str | None):
        """
        Check and load the configuration file for detection settings.

        :param config_path: Path to the configuration file.
        """
        if config_path is not None:

            # Check the validity of file path/name
            resource_dir: str = os.path.join(os.path.abspath(os.path.join(__file__, "../../../../")), "Resources")
            if not os.path.exists(config_path):
                config_path: str = os.path.join(resource_dir, config_path)
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Invalid config file or name!\n{config_path}")

            # Attempt to load the file as a dictionary
            try:
                self.json_settings: dict = read_json_file(config_path)
                self._do_compare = True
                self._config_file_name = os.path.basename(config_path)
            except Exception:
                warnings.warn("Corrupt or invalid config file! Using defaults.")
                self._do_compare = False
                self.json_settings: dict = read_json_file(config_path)
        else:
            self._do_compare = False

    def _correct_for_orientation(self, rois: ROIInfo) -> ROIInfo:
        """
        Correct the orientation of ROIs based on eye side and detection definitions.

        :param rois: ROIInfo object containing ROI data.
        :return: Corrected ROIInfo object.
        """

        def flip_x(direction_dict: dict) -> dict:
            """
            Flip about x-axis.
            :param direction_dict: Input directional dict.
            :return: altered dictionary
            """
            flipped_dict = {
                'N': direction_dict['S'],
                'S': direction_dict['N'],
                'E': direction_dict['E'],
                'W': direction_dict['W']
            }

            return flipped_dict

        def flip_y(direction_dict: dict) -> dict:
            """
            FLip about y-axis.
            :param direction_dict: Input directional dict.
            :return: altered dictionary
            """
            flipped_dict = {
                'N': direction_dict['N'],
                'S': direction_dict['S'],
                'E': direction_dict['W'],
                'W': direction_dict['E']
            }

            return flipped_dict

        def rotate_x_degrees(direction_dict: dict, rotation: int) -> dict:
            """
            Rotate about the center.
            :param direction_dict: Input directional dict.
            :param rotation: Rotation in degrees.
            :return: altered dictionary
            """
            rotated_dict = direction_dict.copy()
            if rotation % 90 == 0 and rotation != 0:
                for _ in np.arange(int(np.abs(rotation / 90))):
                    if rotation < 0:
                        rotated_dict = {
                            'N': rotated_dict['W'],
                            'E': rotated_dict['N'],
                            'S': rotated_dict['E'],
                            'W': rotated_dict['S']
                        }
                    else:
                        rotated_dict = {
                            'N': rotated_dict['E'],
                            'E': rotated_dict['S'],
                            'S': rotated_dict['W'],
                            'W': rotated_dict['N']
                        }

            return rotated_dict

        # Create a temporary tracker
        temp_rois = {"N": rois.north, "E": rois.east, "S": rois.south, "W": rois.west}

        # Set orientation according to eye-side and detection definitions
        if self.eye_side == "L":
            x_flip, y_flip, rotate_deg = self._detection_settings.orientation_left
        else:
            x_flip, y_flip, rotate_deg = self._detection_settings.orientation_right

        # Check for old Tetons orientation
        if self.test_system == "TET" and int(self.image_date.split("-")[0]) < 240222 and self.eye_side[0] == "R":
            y_flip = not y_flip

        # Consider flips about X-axis
        if x_flip:
            temp_rois = flip_x(temp_rois)

        # Consider flips about Y-axis
        if y_flip:
            temp_rois = flip_y(temp_rois)

        # Consider any rotation
        temp_rois = rotate_x_degrees(temp_rois, rotate_deg)

        # Update new ROIs
        new_rois = ROIInfo()
        new_rois.north = temp_rois["N"]
        new_rois.east = temp_rois["E"]
        new_rois.south = temp_rois["S"]
        new_rois.west = temp_rois["W"]
        new_rois.dark = rois.dark

        return new_rois

    def _crop_around_center(self, image_array: np.ndarray, crop_factor: int = 2) -> np.ndarray:
        """
        Crop the center of the image array.
        :param image_array: Image as array.
        :param crop_factor: Scaled height and width of the crop.
        :return: Cropped image array.
        """
        img_height, img_width = image_array.shape[:2]
        if img_height != 0 and img_height != 0:
            # Get dimensions of the image
            if self._detection_settings.default_pivot_point is not None:
                image_center = self._detection_settings.default_pivot_point
            else:
                image_center = (int(img_width / 2), int(img_height / 2))
            crop_height = int(img_height / crop_factor)
            crop_width = int(img_width / crop_factor)

            # Calculate the coordinates of the top-left corner of the crop
            top = int(image_center[1] - crop_height / 2)
            left = int(image_center[0] - crop_width / 2)

            # Calculate the coordinates of the bottom-right corner of the crop
            bottom = int(top + crop_height)
            right = int(left + crop_width)

            # Crop the image
            self._cropped_by = (int(image_center[0] - crop_width / 2),
                                int(image_center[1] - crop_height / 2))
            cropped_image_array = image_array[top:bottom, left:right]
        else:
            cropped_image_array = image_array

        return cropped_image_array

    def _compare_to_reference(self) -> list:
        """
        Compare detected points with reference points from the configuration file.

        :return: List of reference points with comparison results.
        """
        # Set pivot point and clocking based on detected fiducials
        self._set_clock_and_pivot()

        # Extract reference points from the config file
        full_reference = self._get_reference_points()

        # Set local variables
        fiducials = [p for p in full_reference if p["Fiducial"]]
        features = [p for p in full_reference if not p["Fiducial"]]

        # Compare reference to detected
        organized = {}
        for p in self._detection_info.pois:
            # Set reference based on shape
            if self._detection_info.pois[p].is_fiducial:
                reference = fiducials
            else:
                reference = features

            # Determine the closest point in reference
            index, dist = get_nearest_point(self._detection_info.pois[p].center,
                                            [[p["X"], p["Y"]] for p in reference])
            if index is not None and dist <= self._detection_settings.deviation_cutoff:
                # If this point was already saved to the organized dict
                new_index = int(reference[index]["Order"])
                ref_index = [i for i, ref in enumerate(full_reference) if ref["Order"] == new_index][0]
                if new_index in organized.keys() and organized[new_index].ref_delta <= dist:
                    continue

                # Add details to point
                organized[new_index] = self._detection_info.pois[p]
                organized[new_index].ref_delta = round(dist, 3)
                organized[new_index].ref_regions = full_reference[ref_index]["Region"]

        # Fill empty keys
        for r, ref in enumerate(full_reference):
            if r not in organized.keys():
                organized[r] = POIInfo()

        self._detection_info.pois = organized

        return full_reference

    @abc.abstractmethod
    def _detect(self):
        """
        Abstract method to be implemented by subclasses for the actual detection algorithm.
        """
        pass

    def _draw_detection(self):
        """
        Draw detected features on the image for visualization.
        """
        for p in self._detection_info.pois:
            if self._detection_info.pois[p].shape.upper() != "NONE":
                # For easier referencing
                this_point = self._detection_info.pois[p]

                # Label variables
                center = tuple(this_point.center)
                label = f" {p}"

                # Label all ROI points
                if not this_point.is_fiducial:
                    for roi in this_point.mtf_rois.__annotations__.keys():
                        if roi in ["north", "south"]:
                            color = self._color_rect1
                        elif roi == "dark":
                            continue
                        else:
                            color = self._color_rect2
                        this_roi = eval(f"this_point.mtf_rois.{roi}")
                        if tuple(this_roi.center) != (0, 0):
                            cv2.circle(self.image_color8, this_roi.center, self._draw_size, color, self._draw_size)

                # Label center point
                if center != (0, 0):
                    cv2.circle(self.image_color8, center, self._draw_size, self._color_found_pt, self._draw_size)
                    cv2.putText(self.image_color8, label, center, cv2.FONT_HERSHEY_PLAIN, self._draw_size / 2,
                                self._color_found_pt, self._draw_size)

    def _draw_reference(self, full_reference: list | None = None):
        """
        Draw reference points on the image for comparison.

        :param full_reference: List of reference points to draw.
        """
        for p in self._detection_info.pois:
            ref_index = [i for i, ref in enumerate(full_reference) if ref["Order"] == p]
            if self._detection_info.pois[p].shape.upper() == "NONE" and len(ref_index) > 0:
                point = full_reference[ref_index[0]]
                center = (point["X"], point["Y"])
                label = f" {point['Order']}"
                cv2.circle(self.image_color8, center, self._draw_size, self._color_missing_pt, self._draw_size)
                cv2.putText(self.image_color8, label, center, cv2.FONT_HERSHEY_PLAIN, self._draw_size / 2,
                            self._color_missing_pt, self._draw_size)

    @staticmethod
    def _extend_lines(lines: np.ndarray | list, image_shape: tuple) -> list:
        """
        Extend the Hough lines to the edge of the image array.
        :param image_shape: Shape of input image.
        :param lines: An array of Hough lines.
        :return: An array of extended Hough lines.
        """
        # Set local variable
        extended_lines = []
        if len(image_shape) > 2:
            image_h, image_w = image_shape[:2]

            # Loop through extensions
            for line in lines:
                # Grab coordinates
                x1, y1 = line[0], line[1]
                x2, y2 = line[2], line[3]

                # Check for vertical line
                if x1 == x2:
                    extended_lines.append([x1, 0, x1, image_h - 1])
                    continue

                # Calculate slope and intercept
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1

                # Intersect with left/West edge
                points = []
                y_west = intercept
                if 0 <= y_west < image_h:
                    points.append((0, int(y_west)))

                # Intersect with right/East edge
                y_east = slope * (image_w - 1) + intercept
                if 0 <= y_east < image_h:
                    points.append((image_w - 1, int(y_east)))

                # Intersect with top/North edge
                if slope != 0:
                    x_north = -intercept / slope
                    if 0 <= x_north < image_w:
                        points.append((int(x_north), 0))

                    # Intersect with bottom/South edge
                    x_south = ((image_h - 1) - intercept) / slope
                    if 0 <= x_south < image_w:
                        points.append((int(x_south), image_h - 1))

                # Save points to extension
                extended_lines.append([points[0][0], points[0][1], points[1][0], points[1][1]])

        return np.array(extended_lines).tolist()

    def _filter_out_detections(self, check_point: POIInfo) -> tuple:
        """
        Indicates whether to remove, replace, or do nothing if the detected contour is redundant, obtuse, or invalid.
        :param check_point: Either square or fiducial point as dictionary.
        :return: (1) A status where [0: add point to array, 1: replace point in array, 2: do not change array], and
        (2) index in reference to replace.
        """
        # Initialize variables
        ok = 0
        index = None
        if check_point.is_fiducial:
            feature_area = np.mean(self._detection_settings.circle_size)
        else:
            feature_area = np.mean(self._detection_settings.rect_size)

        if len(self._detection_info.pois) > 0:
            centers = [[p, self._detection_info.pois[p].center] for p in self._detection_info.pois if
                       self._detection_info.pois[p].is_fiducial == check_point.is_fiducial]
        else:
            centers = []

        # Remove if rect is detected over fiducial
        # if not check_point.is_fiducial:
        #     fiducial_centers = [self.image_dict[image_name]["Detection"].pois[p]["ROIs"].center
        #                         for p in self.image_dict[image_name]["Detection"].pois if
        #                         self.image_dict[image_name]["Detection"].pois[p].is_fiducial]
        #     for ref in fiducial_centers:
        #         dist = get_point_distance(ref, check_point["ROIs"].center)
        #         if dist < (self.distance_cutoff / 2):
        #             ok = 2
        #             break

        # Remove if an obtuse feature is detected
        if ok == 0:
            for p, ref in centers:
                dist = get_point_distance(ref, check_point.center)
                if dist <= self._detection_settings.deviation_cutoff:

                    # Check if new detection is closer to target size
                    ref_point = self._detection_info.pois[p]
                    old_area = np.abs(ref_point.area - feature_area)
                    new_area = np.abs(check_point.area - feature_area)
                    if new_area < old_area:
                        if not check_point.is_fiducial:
                            old_sd = np.std(ref_point.size_wh)
                            new_sd = np.std(check_point.size_wh)
                            if old_sd > new_sd:
                                # Replace old point if new standard deviation (shape) is better/lower
                                ok = 1
                                index = p
                            else:
                                # Do not include this point
                                ok = 2
                        else:
                            # Replace old point if new area (size) is better/smaller
                            ok = 1
                            index = p
                    else:
                        # Do not include this point
                        ok = 2
                    break

        return ok, index

    @staticmethod
    def _fit_polynomial(points: np.ndarray, order: int = 1, flip_x_y: bool = False) -> tuple:
        """
        Fit polynomial function to dataset.
        :param order: Polynomial order.
        :param flip_x_y: Flip x and y (for vertical/sideways parabola fits)
        :param points: Dataset where the first column is x and y is the second.
        :return: (1) Polynomial coefficients, (2) R^2 of polynomial fit
        """
        # Define points to consider
        if flip_x_y and order >= 2:
            x = points[:, 1]
            y = points[:, 0]
            poly_var = "y"
        else:
            x = points[:, 0]
            y = points[:, 1]
            poly_var = "x"

        # Check if the line is vertical
        unique_x_values = np.unique(x)
        is_vertical_line = bool(len(unique_x_values) == 1 or (np.max(x) - np.min(x)) < 50)
        if np.all(x == x[0]) or is_vertical_line:
            coefficients = np.array([np.inf, float(np.mean(x))])
        else:
            # Fit with polyfit
            coefficients = np.polyfit(x, y, deg=order)

        # Calculate R^2
        poly_fit = np.poly1d(coefficients, variable=poly_var)
        predicted = poly_fit(x)
        if np.inf not in predicted:
            r_squared = r2_score(y, predicted)
        else:
            r_squared = 0

        return poly_fit, r_squared

    @staticmethod
    def _flip(point: tuple) -> tuple:
        """
        Flip the coordinates of a point.

        :param point: Tuple representing a point (x, y).
        :return: Flipped point (y, x).
        """
        return point[1], point[0]

    @staticmethod
    def _get_circularity(contour_area: float, perimeter: float):
        """
        Calculate the circularity of a contour.

        :param contour_area: Area of the contour.
        :param perimeter: Perimeter of the contour.
        :return: Circularity value.
        """
        return 4 * np.pi * (contour_area / (perimeter * perimeter))

    @staticmethod
    def _get_contours(image_thresh: np.ndarray):
        """
        Find contours in a thresholded image.

        :param image_thresh: Thresholded image array.
        :return: List of contours found in the image.
        """
        contours, _ = cv2.findContours(image_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        return contours

    @staticmethod
    def _get_corner_locations(roi_center: tuple | list | np.ndarray, roi_size_wh: tuple = (100, 100)) -> tuple:
        """
        Based on an ROI center, determine where the corners of this ROI are in pixel space.

        :param roi_center: Pixel coordinates of ROI center.
        :param roi_size_wh: ROI size (width x length) in pixels.
        :return: Tuple of top and bottom corner coordinates.
        """
        # Calculate corner locations
        top_left_x = int(roi_center[0] - roi_size_wh[0] // 2)
        top_left_y = int(roi_center[1] - roi_size_wh[1] // 2)
        bottom_right_x = int(roi_center[0] + roi_size_wh[0] // 2)
        bottom_right_y = int(roi_center[1] + roi_size_wh[1] // 2)

        # Save locations to corner outputs
        corner1 = (top_left_x, top_left_y)
        corner2 = (bottom_right_x, bottom_right_y)

        return corner1, corner2

    def _get_crosshair_pois(self, grouped_line_info: dict) -> dict:
        """
        Find where the detected cross-hair lines intersect.

        :param grouped_line_info: Output from self.group_hough_lines().
        :return: Dictionary of intersection information.
        """

        def empty_check(check_array: np.ndarray, limit: int = 10) -> bool:
            # Sort the pixel values in descending order
            flattened = check_array.flatten()
            sorted_pixels = np.sort(flattened)[::-1]

            # Calculate the number of pixels in the top 5%
            top_5_percent_count = int(len(sorted_pixels) * 0.05)

            # Compute the average of top 5% pixel values
            top_5_percent_pixels = sorted_pixels[:top_5_percent_count]
            top_5_percent_average = np.mean(top_5_percent_pixels)

            return top_5_percent_average >= limit

        # Set local variables
        intersection_combos: set = set()
        intersection_num: int = 1
        pois: dict = {}

        if len(self.image_mono8.shape) >= 2:
            image_h, image_w = self.image_mono8.shape[:2]

            # Precompute polynomial functions
            poly_funcs = {n: self._make_poly_function(grouped_line_info[n].fit) for n in grouped_line_info}

            # Iterate over unique pairs of lines
            for (n, m) in combinations(grouped_line_info.keys(), 2):
                if grouped_line_info[n].direction != grouped_line_info[m].direction:
                    poly1, poly2 = grouped_line_info[n].fit, grouped_line_info[m].fit
                    func1, func2 = poly_funcs[n], poly_funcs[m]

                    # Find intersection
                    intersection_x, intersection_y = np.nan, np.nan
                    if np.isinf(poly1.c[0]) and poly1.o == 1:  # Vertical line case
                        intersection_x = poly1.c[1]
                        intersection_y = np.polyval(poly2, intersection_x)
                    elif np.isinf(poly2.c[0]) and poly2.o == 1:  # Vertical line case
                        intersection_x = poly2.c[1]
                        intersection_y = np.polyval(poly1, intersection_x)
                    else:
                        roots = np.roots(poly1 - poly2).real
                        if len(roots) > 0:
                            if ((poly1.o == 2 and grouped_line_info[n].direction == "Vertical") or
                                    (poly2.o == 2 and grouped_line_info[m].direction == "Vertical")):
                                def equations(guess: tuple) -> list:
                                    x, y = guess
                                    if poly1.variable == "x":
                                        p1 = func1(x) - y
                                    else:
                                        p1 = func1(y) - x
                                    if poly2.variable == "x":
                                        p2 = func2(x) - y
                                    else:
                                        p2 = func2(y) - x
                                    return [p1, p2]

                                try:
                                    roots = fsolve(equations, np.array([0, 0]))
                                    intersection_x, intersection_y = roots[0], roots[1]
                                except IndexError:
                                    pass
                            else:
                                intersection_x = next((value for value in roots.flatten() if 0 <= value <= image_w),
                                                      np.nan)
                                intersection_y = np.polyval(poly1, intersection_x)

                    # Validate intersections within image bounds
                    if (not np.isnan(intersection_x) and 0 <= intersection_x <= image_w and
                            0 <= intersection_y <= image_h):
                        intersection = (int(intersection_x), int(intersection_y))

                        # Avoid duplicate intersections
                        if (m, n) not in intersection_combos and (n, m) not in intersection_combos:
                            crh_rois = self._get_crosshair_rois(intersection)
                            if empty_check(crh_rois.north.array):
                                poi_info: POIInfo = POIInfo()
                                poi_info.area = 0.0,
                                poi_info.center = intersection
                                poi_info.is_fiducial = False
                                poi_info.shape = "Crosshair"
                                poi_info.mtf_rois = crh_rois
                                poi_info.lines = {m: grouped_line_info[m], n: grouped_line_info[n]}

                                pois.update({intersection_num: poi_info})
                                intersection_num += 1
                                intersection_combos.add((n, m))

        return pois

    def _get_crosshair_rois(self, roi_center: tuple | list) -> ROIInfo:
        """
        Define the relevant geometric info for each ROI about the detected intersection.

        :param roi_center:Pixel coordinates of ROI center.
        :return: Dictionary with info on ROI center, cropped array, corners, width x height, and LSF direction.
        """
        # Define ROI centers
        dist_center = int(self._roi_size[1] / 2 + self._roi_cushion)
        opposite = int(dist_center * np.sin(np.deg2rad(self.clocking_angle)))
        adjacent = int(dist_center * np.cos(np.deg2rad(self.clocking_angle)))

        # Save to ROIs
        rois: ROIInfo = ROIInfo()

        rois.north.center = (roi_center[0] + opposite, roi_center[1] - adjacent)
        rois.north.mtf_direction = "Horizontal"
        rois.north.size_wh = self._roi_size
        rois.north.corners = self._get_corner_locations(rois.north.center, roi_size_wh=rois.north.size_wh)
        rois.north.array = crop_image(self.image_mono16, rois.north.center, roi_size_wh=rois.north.size_wh)

        rois.east.center = (roi_center[0] + adjacent, roi_center[1] + opposite)
        rois.east.mtf_direction = "Vertical"
        rois.east.size_wh = self._flip(self._roi_size)
        rois.east.corners = self._get_corner_locations(rois.east.center, roi_size_wh=rois.east.size_wh)
        rois.east.array = crop_image(self.image_mono16, rois.east.center, roi_size_wh=rois.east.size_wh)

        rois.south.center = (roi_center[0] - opposite, roi_center[1] + adjacent)
        rois.south.mtf_direction = "Horizontal"
        rois.south.size_wh = self._roi_size
        rois.south.corners = self._get_corner_locations(rois.south.center, roi_size_wh=rois.south.size_wh)
        rois.south.array = crop_image(self.image_mono16, rois.south.center, roi_size_wh=rois.south.size_wh)

        rois.west.center = (roi_center[0] - adjacent, roi_center[1] - opposite)
        rois.west.mtf_direction = "Vertical"
        rois.west.size_wh = self._flip(self._roi_size)
        rois.west.corners = self._get_corner_locations(rois.west.center, roi_size_wh=rois.west.size_wh)
        rois.west.array = crop_image(self.image_mono16, rois.west.center, roi_size_wh=rois.west.size_wh)

        rois.dark.center = (roi_center[0] + max(self._roi_size) + self._roi_cushion,
                            roi_center[1] + max(self._roi_size) + self._roi_cushion)
        rois.dark.size_wh = self._dark_roi_size
        rois.dark.corners = self._get_corner_locations(rois.dark.center, roi_size_wh=rois.dark.size_wh)
        rois.dark.array = crop_image(self.image_mono16, rois.dark.center, roi_size_wh=rois.dark.size_wh)

        # Consider orientation of image vs real
        rois = self._correct_for_orientation(rois)

        return rois

    def _get_circle_poi(self, circle: np.ndarray) -> POIInfo:
        """
        Get geometric information for a detected circle.

        :param circle: Array representing the circle.
        :return: POIInfo object with circle details.
        """
        geo_info: POIInfo = POIInfo()
        if circle.size > 0:
            xc, yc, rc, sig = circle_fit.least_squares_circle(circle)

            # Save geometry
            geo_info.area = np.pi * (rc ** 2)
            geo_info.center = (int(xc + self._cropped_by[0]), int(yc + self._cropped_by[1]))
            geo_info.is_fiducial = True
            geo_info.shape = "Blob"
            geo_info.width_height = (int(rc), int(rc))

        return geo_info

    def _group_hough_lines(self, hough_lines: np.ndarray, image_shape: tuple, r2_limit: float = 0.96) -> dict:
        """
        Extract relevant info from Hough lines like slope and projected endpoints, then organize the lines into groups.

        :param image_shape: Shape of input image.
        :param hough_lines: An array of Hough lines.
        :param r2_limit: Limit for R^2. Influences the polynomial fit orders.
        :return: Dictionary of geometrical info for cross-hairs.
        """
        # Set local variables
        group_num: int = 1
        group_info: dict = {group_num: LineInfo()}
        lines_dict: dict = {}

        # Make dict to keep track of slopes and mid-points of each detection
        hough_lines = hough_lines.reshape(-1, 4)
        for n, line in enumerate(hough_lines):
            hough_lines[n] = np.array([hough_lines[n][0] + self._cropped_by[0], hough_lines[n][1] + self._cropped_by[1],
                                       hough_lines[n][2] + self._cropped_by[0],
                                       hough_lines[n][3] + self._cropped_by[1]])
        extended_lines = self._extend_lines(hough_lines, image_shape)
        for n, _ in enumerate(hough_lines):
            # Avoid division by zero for vertical lines
            x1, y1, x2, y2 = hough_lines[n].tolist()
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = np.inf

            # Define direction of line based on the slope
            if abs(slope) < 1:
                direction = "Horizontal"
            else:
                direction = "Vertical"

            # Find the midpoint
            ex1, ey1, ex2, ey2 = extended_lines[n]
            midpoint = (int((ex1 + ex2) / 2), int((ey1 + ey2) / 2))

            # Save info for later
            lines_dict[n] = {
                "Direction": direction,
                "Endpoints": (ex1, ey1, ex2, ey2),
                "Group #": 0,
                "Midpoint": midpoint,
                "Probe Points": [(x1, y1), (x2, y2)],
                "Slope": slope
            }

            # Set clocking angle if not already
            if self.clocking_angle == 0 and direction == "Horizontal":
                self._clocking_angle = self._angle_to_horizontal_line(slope)

        # Organize detections based on proximity and slope
        for n in lines_dict:
            # Set temporary variables
            this_midpoint = lines_dict[n]["Midpoint"]
            this_probe = lines_dict[n]["Probe Points"]
            this_direction = lines_dict[n]["Direction"]

            # For first iteration
            if len(group_info[group_num].probes) == 0:
                group_info[group_num].direction = this_direction
                group_info[group_num].probes = [this_midpoint] + this_probe
            else:
                # Determine which group based on distance
                add_to_group = False
                for group in group_info:
                    index, dist = get_nearest_point(this_midpoint, group_info[group].probes)
                    if (dist < self._detection_settings.deviation_cutoff and
                            this_direction == group_info[group].direction):
                        add_to_group = True
                        group_num = group

                # Add point to a group
                if add_to_group:
                    group_info[group_num].probes += [this_midpoint] + this_probe

                # Create a new group for point
                else:
                    group_num = list(group_info.keys())[-1] + 1
                    group_info.update({group_num: LineInfo()})
                    group_info[group_num].probes = [this_midpoint] + this_probe
                    group_info[group_num].direction = this_direction

            # Save group number
            lines_dict[n]["Group #"] = group_num

        # Best fit lines
        for group_num in group_info:

            # Fit 1st or 2nd-order polynomial
            order = 1
            poly_fit = np.poly1d([])
            probes = np.array(group_info[group_num].probes)
            r_squared = 0
            while r_squared < r2_limit:
                # Determine R^2 and fit coefficients
                poly_fit, r_squared = self._fit_polynomial(probes, order=order,
                                                           flip_x_y=group_info[group_num].direction == "Vertical")

                # End tasks
                if order >= 2:
                    break
                elif r_squared < r2_limit:
                    order += 1

            # Get slope
            if order == 1:
                slope = poly_fit.c[0]
            else:
                p_derivative = poly_fit.deriv()
                slope = p_derivative(np.mean(probes, axis=0)[0])

            # Save coefficients
            group_info[group_num].r_squared = round(r_squared, self._sig_fig)
            group_info[group_num].fit = poly_fit
            group_info[group_num].slope = round(slope, self._sig_fig)

        # Save info to class variable
        return group_info

    def _get_rect_poi(self, rect: np.ndarray, image_array: np.ndarray) -> POIInfo:
        """
        Sort the detected rect points such that the cardinal edges and each corner are defined.

        :param image_array: Used to crop images to save ROIs.
        :param rect: Approximation for rect of interest.
        :return: Dictionary with location descriptors for each key point.
        """
        # Set local variables
        geo_info: POIInfo = POIInfo()

        if len(rect) == 4:
            # Set more variables
            rect = rect.reshape((4, 2))
            roi_size = (self._roi_size[0] / 100, self._roi_size[1] / 100)
            x_points = rect[:, 0]
            y_points = rect[:, 1]

            # Calculate width-height
            wh = (float(np.max(x_points) - np.min(x_points)), float(np.max(y_points) - np.min(y_points)))

            # Calculate the centroid of the box points
            center = np.mean(rect, axis=0).astype(int)

            # Assign corners
            corners: dict = {}
            sorted_points = sorted(rect, key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]))
            corners["NW"], corners["NE"], corners["SE"], corners["SW"] = sorted_points

            # Save ROI info
            rois: ROIInfo = ROIInfo()

            rois.north.center = get_pxl_midpoint(corners["NW"], corners["NE"])
            rois.north.corners = [corners["NW"], corners["NE"]]
            rois.north.size_wh = (int(wh[0] * roi_size[1]), int(wh[1] * roi_size[0]))
            rois.north.mtf_direction = "Vertical"
            rois.north.array = crop_image(image_array, rois.north.center, roi_size_wh=rois.north.size_wh)

            rois.east.center = get_pxl_midpoint(corners["SE"], corners["NE"])
            rois.east.corners = [corners["SE"], corners["NE"]]
            rois.east.size_wh = (int(wh[0] * roi_size[0]), int(wh[1] * roi_size[1]))
            rois.east.mtf_direction = "Horizontal"
            rois.east.array = crop_image(image_array, rois.east.center, roi_size_wh=rois.east.size_wh)

            rois.south.center = get_pxl_midpoint(corners["SE"], corners["SW"])
            rois.south.corners = [corners["SE"], corners["SW"]]
            rois.south.size_wh = (int(wh[0] * roi_size[1]), int(wh[1] * roi_size[0]))
            rois.south.mtf_direction = "Vertical"
            rois.south.array = crop_image(image_array, rois.south.center, roi_size_wh=rois.south.size_wh)

            rois.west.center = get_pxl_midpoint(corners["NW"], corners["SW"])
            rois.west.corners = [corners["NW"], corners["SW"]]
            rois.west.size_wh = (int(wh[0] * roi_size[0]), int(wh[1] * roi_size[1]))
            rois.west.mtf_direction = "Horizontal"
            rois.west.array = crop_image(image_array, rois.west.center, roi_size_wh=rois.west.size_wh)

            # Consider orientation of image vs real
            rois = self._correct_for_orientation(rois)

            # Update default
            geo_info.area = cv2.contourArea(rect)
            geo_info.center = center
            geo_info.is_fiducial = False
            geo_info.mtf_rois = rois
            geo_info.shape = "Rect"
            geo_info.width_height = wh

        return geo_info

    def _get_reference_points(self) -> list[dict]:
        """
        Retrieve reference points from the configuration file and convert them into pixel space.

        :return: List of dictionaries containing reference point information.
        """

        def check_for_tetons_fix() -> int:
            """
            Check date, test system, and eye side for bad Tetons orientation.
            :return: direction of flip
            """
            date = self.image_date
            if (date is not None and int(date.split("-")[0]) < 240222 and self.test_system == "TET" and
                    self.eye_side[0] == "R"):
                return -1
            else:
                return 1

        def rotate_about_pivot(point: tuple, rotation: float, offset: tuple) -> tuple:
            """
            Rotate coordinate about the pivot point.
            :param point: Coordinate to rotate.
            :param rotation: Rotation amount in degrees.
            :param offset: Offset from center.
            :return: new coordinates
            """
            new_x = int(offset[0] + np.cos(rotation) * point[0] - np.sin(rotation) * point[1])
            new_y = int(offset[1] + np.sin(rotation) * point[0] + np.cos(rotation) * point[1])

            return new_x, new_y

        # Set local variables
        side = get_long_name_side(self.eye_side)
        if side == "Left":
            orientation = self._detection_settings.orientation_left
        else:
            orientation = self._detection_settings.orientation_right

        # Pull angular field points from config file
        raw_reference = deepcopy(self.json_settings["ReticlePattern"][f"{side}Features"])
        reference_points: list = [(float(p["X"]), float(p["Y"])) for p in raw_reference]

        # Convert into pixel space
        if len(reference_points) > 0:

            # Define offset from (0, 0)
            zero_index = next((i for i, coord in enumerate(reference_points) if coord == (0, 0)), None)
            if zero_index is None:
                zero_index, _ = get_nearest_point((0, 0), reference_points)
            xy_offset = (reference_points[zero_index][0] + self.pivot_point[0],
                         reference_points[zero_index][0] + self.pivot_point[1])

            # Get flip and rotation factors
            x_flip_factor = 1 if not orientation[0] else -1
            y_flip_factor = 1 if not orientation[1] else -1
            rot_factor = np.deg2rad(self.clocking_angle) + np.deg2rad(orientation[2])

            # Apply flip and rotation factors
            for i, (x_ang, y_ang) in enumerate(reference_points):
                x_pxl = x_ang * self._detection_settings.ang2pxl * x_flip_factor
                y_pxl = y_ang * self._detection_settings.ang2pxl * y_flip_factor * check_for_tetons_fix()
                reference_points[i] = rotate_about_pivot((x_pxl, y_pxl), rot_factor, xy_offset)

        # Update values in reference
        for r, ref in enumerate(raw_reference):
            ref["X"], ref["Y"] = reference_points[r]
            ref["Region"] = [r for r in self.json_settings["DisplayRegionSquares"] if ref["Order"] in
                             self.json_settings["DisplayRegionSquares"][r]]

        return raw_reference

    @staticmethod
    def _make_poly_function(poly_fit: np.poly1d):
        """
        Create a lambda function from a polynomial fit.

        :param poly_fit: Polynomial fit object.
        :return: Lambda function representing the polynomial.
        """
        if poly_fit.order == 0:
            return lambda x: poly_fit.c[0]
        elif poly_fit.order == 1:
            return lambda x: poly_fit.c[0] * x + poly_fit.c[1]
        else:
            return lambda x: poly_fit.c[0] * x ** 2 + poly_fit.c[1] * x + poly_fit.c[2]

    def _preprocess_image(self, normalize_exposure: bool = True) -> np.ndarray:
        """
        Preprocess the image by cropping, normalizing exposure, applying Gaussian blur, and thresholding.

        :param normalize_exposure: Flag to normalize image exposure.
        :return: Preprocessed image array.
        """

        def reduce_noise(image_8bit: np.ndarray, sigma: int = 1500, brightness: float = 0.1):
            """
            Reduce noise by increasing dark patterns and ignoring background contributions.
            :param image_8bit: Image to process.
            :param sigma: Gaussian shape for brightness multiplication.
            :param brightness: additional brightness factor.
            :return: 'normalized' image
            """
            # Assumes input is 8-bit monochrome
            bit_max = 2 ** 8 - 1
            image = image_8bit / bit_max

            # Create a 2D Gaussian kernel
            h, w = image_8bit.shape
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            center_x, center_y = w // 2, h // 2
            gaussian_kernel = 1 - np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))

            # Normalize the kernel to sum to 1 for smoothing effect
            gaussian_kernel /= np.max(gaussian_kernel)

            # Find local minima (troughs) in the histogram
            histogram, bin_edges = np.histogram(image.flatten(), bins=(bit_max + 1), range=(0, bit_max))
            trough_indices = argrelmin(histogram)[0]  # Indices of local minima

            # Get the first trough index and its corresponding pixel intensity
            if len(trough_indices) > 0:
                first_trough_index = trough_indices[0]
                first_trough_intensity = bin_edges[first_trough_index]
                pixel_floor_percent = max(0.01, min(1, first_trough_intensity / bit_max))
            else:
                pixel_floor_percent = 0.05

            # Create a mask for darker squares
            mask = (image > pixel_floor_percent / 2) & (image < 0.5)

            # Apply the Gaussian kernel selectively to brighten darker squares
            normalized = image.copy()
            normalized[mask] = np.clip(normalized[mask] + gaussian_kernel[mask] * brightness, 0, 1)

            # Keep bright squares unchanged
            normalized[~mask] = image[~mask]

            # Scale back to 8-bit range
            normalized = np.clip(normalized * bit_max, 0, 255).astype(np.uint8)

            return normalized

        def apply_explicit_threshold(image: np.ndarray) -> np.ndarray:
            """
            Apply thresholding to image.
            :param image: Image to threshold.
            :return: thresholded image
            """
            _, threshed = cv2.threshold(image, self._detection_settings.threshold, 255, cv2.THRESH_BINARY)

            return threshed

        # Crop image around center
        if self._crop and self._cropped_by != (0, 0):
            image_mono8 = self._crop_around_center(self.image_mono8)
        else:
            image_mono8 = self.image_mono8

        # Normalize image between long and short exposures
        if normalize_exposure:
            image_mono8 = reduce_noise(image_mono8)

        # Apply Gaussian blur
        image_gauss = cv2.GaussianBlur(image_mono8, (self._detection_settings.gauss, self._detection_settings.gauss),
                                       sigmaX=1)

        # Apply adaptive or explicit thresholding
        image_thresh = apply_explicit_threshold(image_gauss)

        return image_thresh

    def _set_clock_and_pivot(self) -> tuple[tuple, float]:
        """
        Get the tilt angle based solely on detected fiducial points and not what is in the config file.

        :return: the found pivot point and clocking angle in degrees
        """

        def find_fiducials(coordinates: np.ndarray):

            def get_geometries(points: np.ndarray, tolerance: int = 0.5) -> tuple:
                """
                Get pivot point and clocking angle from the formed triangle.
                :param points: End points of triangle sides.
                :param tolerance: Standard deviation tolerance for detected triangle size.
                :return: (1) Conformity, (2) pivot point, (3) clocking angle
                """
                # Set list of distance between three points
                distances = []
                n = 1
                for p in points:
                    if n >= len(points):
                        n = 0
                    angle = get_point_angle(p, points[n])
                    mid_x, mid_y = get_pxl_midpoint(p, points[n])
                    distances.append([get_point_distance(p, points[n]), mid_x, mid_y, angle])
                    n += 1

                # Sort distances to identify the hypotenuse (longest side)
                distances = np.array(distances)
                distances = distances[distances[:, 0].argsort()]

                # The two smallest distances are the non-hypotenuse sides
                side1 = distances[0][0] / self._detection_settings.ang2pxl
                side2 = distances[1][0] / self._detection_settings.ang2pxl
                hypotenuse = distances[2][0] / self._detection_settings.ang2pxl

                # Define pivot and tilt
                pivot_point = (int(distances[2][1]), int(distances[2][2]))
                pivot_tilt = min(distances[:, -1], key=abs)
                if 35 < abs(pivot_tilt) < 55:
                    pivot_tilt = abs(pivot_tilt) - 45
                elif 80 <= abs(pivot_tilt) < 100:
                    pivot_tilt = abs(pivot_tilt) - 90

                # Define area
                area = 0.5 * (side1 + side2 + hypotenuse)

                # Check if size conforms to tolerance
                ok = False
                if np.std((area, self._detection_settings.fid_triangle_area)) <= tolerance:
                    if "MPBinocle" in self._detection_settings_name or "ML2" in self._detection_settings_name:
                        tolerance = 5
                    if np.std((side1, side2)) < tolerance:
                        ok = True

                return ok, pivot_point, pivot_tilt

            def get_triangle_angles(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple:
                """
                Calculate teh triangle angles from the side widths.
                :param a: Endpoints of side A.
                :param b: Endpoints of side B.
                :param c: Endpoints of side C.
                :return: Angles of formed triangle.
                """
                # Calculate the lengths of the sides
                ab = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
                ac = math.sqrt((c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2)
                bc = math.sqrt((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2)

                # Calculate angles using the cosine rule
                angle_a = math.acos((ab ** 2 + ac ** 2 - bc ** 2) / (2 * ab * ac)) * (180 / math.pi)
                angle_b = math.acos((ab ** 2 + bc ** 2 - ac ** 2) / (2 * ab * bc)) * (180 / math.pi)
                angle_c = 180 - angle_a - angle_b

                return angle_a, angle_b, angle_c

            def is_right_triangle(points: np.ndarray) -> np.ndarray:
                """
                Determine if triangle is approximately a right triangle.
                :param points: End points of triangle sides.
                :return: Which combinations of endpoints forma right triangle.
                """
                valid_triangles = []
                for a, b, c in combinations(points, 3):
                    angles = get_triangle_angles(a, b, c)
                    if any(80 <= angle <= 100 for angle in angles):
                        valid_triangles.append([a, b, c])
                valid_triangles = np.unique(np.array(valid_triangles), axis=0)

                return valid_triangles

            # Get possible triangular combinations of detected 'fiducials'
            possible_triangles = is_right_triangle(coordinates)

            # Filter based on size
            found = []
            for tuples in possible_triangles:
                is_ok, pivot, tilt = get_geometries(tuples)
                if is_ok:
                    if self.is_olaf:
                        index, dist = get_nearest_point(pivot, list(coordinates))
                        if np.abs(dist) < self._detection_settings.deviation_cutoff:
                            lowest_y_point = tuples[np.argmin(tuples[:, 1])]
                            filtered = [point for point in tuples if not np.array_equal(point, lowest_y_point)]
                            olaf_tuples = np.array([filtered[0], coordinates[index], filtered[1]])
                            self._detection_settings.fid_triangle_area *= 0.75
                            is_ok, pivot, tilt = get_geometries(olaf_tuples)
                            self._detection_settings.fid_triangle_area /= 0.75
                            if not is_ok:
                                continue
                            else:
                                tuples = olaf_tuples
                    found.append([tuples, pivot, tilt])

            return found

        # Set local variable(s)
        output_pivot, output_tilt = self.pivot_point, self.clocking_angle
        found_fiducials = []
        pois = self._detection_info.pois

        # Get fiducial coordinates
        try:
            detected_fiducials = [pois[p].center for p in pois if pois[p].is_fiducial]
            found_fiducials = find_fiducials(np.array(detected_fiducials))
        except KeyError:
            pass

        # Check found fiducials according to test system
        if not self.is_crosshair:
            if 0 < len(found_fiducials) <= 2:
                # Create a new list excluding the point with the minimum y-coordinate (for Olaf)
                if self.test_system == "OLF" and len(found_fiducials) == 2:
                    found_fiducials = [max(found_fiducials, key=lambda point: point[1][1])]

                # Define pivot point and clocking angle
                self._pivot_point = found_fiducials[0][1]
                if self.test_system in ["PIN", "OLF"]:
                    self._clocking_angle = 0
                else:
                    self._clocking_angle = found_fiducials[0][2]
            else:
                warnings.warn(f"Could not detect fiducials (or none present)! Using default values for pivot.")

        return output_pivot, output_tilt

    def _set_detection_settings(self):
        """
        Set the detection settings for the configured run.
        """
        # Format input
        if self.config_name is None:
            target_type = "UNKNOWN"
        else:
            target_type = os.path.basename(self.config_name).split("_")[0].upper()

        # Define handle to settings
        if "GALILEO" in target_type:
            handle = GalileoDetection
        elif "MIDAS" in target_type:
            handle = MidasDetection
        elif "OLAF" in target_type:
            handle = OlafDetection
        elif "HYDRA" in target_type:
            handle = HydraDetection
        elif "ML2" in target_type or "BINOCLE" in target_type:
            handle = ML2Detection
        elif "TRIOPTICS" in target_type:
            handle = TriOpticsDetection
        else:
            handle = DefaultDetection

        # Set class variables accordingly
        self._crop = handle.__name__ != "DefaultDetection"
        self._detection_settings = handle(test_system=self.test_system, is_crosshair=self.is_crosshair)
        self._detection_settings_name = handle.__name__
        self.is_olaf = "Olaf" in handle.__name__


# Child Classes ----------------------------------------------------------------------------------------------------- #
class BlobDetection(DetectionBase):
    """
    Child class for blob detection, extending the functionality of DetectionBase.
    """

    def __init__(self, image_dir_or_file: str | np.ndarray, config_path: str = None, is_olaf: bool = False,
                 debug_dir: str | None = None, **kwargs):
        """
        Initialize the BlobDetection with the given parameters.

        :param config_path: Path to the configuration file for detection settings.
        :param debug_dir: Directory to save debug information. If None, uses the image directory.
        :param image_dir_or_file: Path to the image directory, a single image file, or an image array.
        :param is_olaf: Flag to indicate if the image is from an Olaf system.
        :param kwargs: Additional keyword arguments for the KPI filtering.
        """
        super().__init__(image_dir_or_file, config_path=config_path, is_olaf=is_olaf, debug_dir=debug_dir, **kwargs)

    def _detect(self):
        """
        Blob or circular detection algorithm.
        """
        # Pre-process image
        preprocessed_image = self._before_detect()

        # Find edge contours
        contours_found = self._get_contours(preprocessed_image)

        # Detect fiducials and SFR patterns in image
        for contour in contours_found:
            # Get geometrical characteristics
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            contour_area = cv2.contourArea(contour)

            # Filter based on size
            if self._detection_settings.circle_size[0] <= contour_area <= self._detection_settings.circle_size[1]:
                # Get circularity of potential blob
                circularity = self._get_circularity(contour_area, perimeter)
                if self._draw_contours:
                    cv2.drawContours(self.image_color8, contour, -1, self._color_txt, self._draw_size)

                # Check shape to determine if fiducial or square
                key_point: POIInfo = POIInfo()
                if circularity >= self._detection_settings.circularity_min:
                    approx = cv2.approxPolyDP(contour, 1, True)
                    individual = np.array([pnt[0] for pnt in approx])
                    shape_area = cv2.contourArea(individual)
                    if (self._detection_settings.circle_size[0] <= shape_area <=
                            self._detection_settings.circle_size[1]):
                        key_point = self._get_circle_poi(individual)

                # Determine whether to save or ignore the given point
                if key_point.area > 0:
                    ok, replace_this_index = self._filter_out_detections(key_point)
                    if ok == 0:
                        index = self._detection_info.pois.__len__()
                        self._detection_info.pois[index] = key_point
                    elif ok == 1:
                        self._detection_info.pois[replace_this_index] = key_point
                    else:
                        pass

        # End tasks
        self._after_detect()


class CrosshairDetection(DetectionBase):
    """
    Child class for crosshair detection, extending the functionality of DetectionBase.
    """

    def __init__(self, image_dir_or_file: str | np.ndarray, config_path: str = None, is_olaf: bool = False,
                 debug_dir: str | None = None, **kwargs):
        """
        Initialize the CrosshairDetection with the given parameters.

        :param config_path: Path to the configuration file for detection settings.
        :param debug_dir: Directory to save debug information. If None, uses the image directory.
        :param image_dir_or_file: Path to the image directory, a single image file, or an image array.
        :param is_olaf: Flag to indicate if the image is from an Olaf system.
        :param kwargs: Additional keyword arguments for the KPI filtering.
        """
        super().__init__(image_dir_or_file, config_path=config_path, is_crosshair=True, is_olaf=is_olaf,
                         debug_dir=debug_dir, **kwargs)

    def _detect(self):
        """
        Crosshair detection algorithm.
        """
        # Pre-process image
        preprocessed_image = self._before_detect()

        # Use Hough Lines transform to fit contours
        lines = cv2.HoughLinesP(preprocessed_image, 1, np.pi / 180, maxLineGap=1,
                                threshold=self._detection_settings.threshold,
                                minLineLength=self._detection_settings.hough_min_length)

        # Sort detected lines
        if lines is not None:
            # Group Hough lines based on direction and proximity
            grouped_line_info = self._group_hough_lines(lines, self.image_color8.shape)
            self._detection_info.pois = self._get_crosshair_pois(grouped_line_info)

            # Draw detected lines
            if self._draw_contours:
                lines = lines.reshape(-1, 4)
                for end_points in lines:
                    # Calculate slope
                    x1, y1, x2, y2 = end_points
                    if x1 == x2:
                        slope = np.inf
                    else:
                        slope = abs((y2 - y1) / (x2 - x1))

                    # Determine color based on slope
                    if slope < 1:
                        color = self._color_rect1  # horizontal lines
                    else:
                        color = self._color_rect2  # vertical lines

                    # Draw lines
                    cv2.line(self.image_color8, (x1, y1), (x2, y2), color, self._draw_size)

        # End tasks
        self._after_detect()

    def _draw_detection(self):
        """
        Draw detected features on the image for visualization.
        """
        # Tracker for which line groups have already been drawn
        groups_drawn = []

        # Set colors for directionality
        color_h = self._color_rect1
        color_v = self._color_rect2

        # Draw detections
        for p in self._detection_info.pois:
            if self._detection_info.pois[p].shape.upper() != "NONE":
                # For easier referencing
                this_point = self._detection_info.pois[p]

                # Draw detected lines
                if self._draw_fits:
                    for i in this_point.lines:
                        if i not in groups_drawn:
                            # Set local variables
                            this_line = this_point.lines[i]
                            image_h, image_w = self.image_color8.shape[:2]

                            # Ensure we aren't drawing multiples of the same line
                            groups_drawn.append(i)

                            # Set color for line
                            color = color_h if this_line.direction == "Horizontal" else color_v

                            # For perfectly vertical line case (first-order)
                            if np.isinf(this_line.fit[0]) and this_line.fit.order == 1:
                                x = int(this_line.fit[1])
                                cv2.line(self.image_color8, (x, 0), (x, image_h), color, thickness=2)

                            # For all other cases
                            else:
                                poly_func = this_line.fit
                                x_end = image_w
                                x_start = 0

                                # For linear case (first-order)
                                if this_line.fit.order == 1:
                                    y_start = int(poly_func(x_start))
                                    y_end = int(poly_func(x_end))
                                    cv2.line(self.image_color8, (x_start, y_start), (x_end, y_end), color, thickness=2)

                                # For other nth-order cases
                                else:
                                    # For vertical line case (nth-order)
                                    if this_line.direction == "Vertical":
                                        x_end = image_h
                                        pt_output = np.linspace(x_start, x_end, num=x_end, dtype=int)
                                        pt_input = poly_func(pt_output).astype(int)

                                    # For horizontal line case (nth-order)
                                    else:
                                        pt_input = np.linspace(x_start, x_end, num=x_end, dtype=int)
                                        pt_output = poly_func(pt_input).astype(int)

                                    points = np.array([[[x, y]] for x, y in zip(pt_input, pt_output) if 0 <= y < x_end])
                                    cv2.polylines(self.image_color8, [points], isClosed=False, color=color, thickness=2)

                # Draw ROI around intersections
                for roi in this_point.mtf_rois.__annotations__.keys():
                    # For easier referencing
                    this_roi = eval(f"this_point.mtf_rois.{roi}")

                    # Determine color to use
                    if roi == "dark":
                        color = self._color_blob
                    else:
                        color = color_h if this_roi.mtf_direction == "Horizontal" else color_v

                    # Draw and label ROI corners
                    corners = this_roi.corners
                    cv2.rectangle(self.image_color8, corners[0], corners[1], color, self._draw_size)

                # Draw center of intersection
                if this_point.center != (0, 0):
                    cv2.circle(self.image_color8, this_point.center, self._draw_size, self._color_found_pt,
                               self._draw_size)
                    cv2.putText(self.image_color8, f" {p}", this_point.center,
                                cv2.FONT_HERSHEY_PLAIN, int(self._draw_size / 2), self._color_found_pt, self._draw_size)


class SFRDetection(DetectionBase):
    """
    Child class for SFR detection, extending the functionality of DetectionBase.
    """

    def __init__(self, image_dir_or_file: str | np.ndarray, config_path: str = None, is_olaf: bool = False,
                 debug_dir: str | None = None, **kwargs):
        """
        Initialize the SFRDetection with the given parameters.

        :param config_path: Path to the configuration file for detection settings.
        :param debug_dir: Directory to save debug information. If None, uses the image directory.
        :param image_dir_or_file: Path to the image directory, a single image file, or an image array.
        :param is_olaf: Flag to indicate if the image is from an Olaf system.
        :param kwargs: Additional keyword arguments for the KPI filtering.
        """
        super().__init__(image_dir_or_file, config_path=config_path, is_olaf=is_olaf, debug_dir=debug_dir, **kwargs)

    def _detect(self):
        """
        SFR pattern detection algorithm.
        """
        # Pre-process image
        preprocessed_image = self._before_detect()

        # Find contours
        contours_found = self._get_contours(preprocessed_image)

        # Detect fiducials and SFR patterns in image
        for contour in contours_found:

            # Show edge detection (if requested)
            if self._draw_contours:
                cv2.drawContours(self.image_color8, contour, -1, self._color_txt, self._draw_size)

            # Filter based on largest range of acceptable sizes
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            contour_area = cv2.contourArea(contour)
            min_size = min(self._detection_settings.circle_size[0], self._detection_settings.rect_size[0])
            max_size = max(self._detection_settings.circle_size[1], self._detection_settings.rect_size[1])
            if min_size <= contour_area <= max_size:

                # Get geometrical characteristics for filtering
                approx = cv2.approxPolyDP(contour, 1, True)
                circularity = self._get_circularity(contour_area, perimeter)
                max_circle_size = self._detection_settings.circle_size[1]
                min_rect_size = self._detection_settings.rect_size[0]

                # Check shape to determine if fiducial or square
                key_point: POIInfo = POIInfo()
                if circularity >= self._detection_settings.circularity_min and contour_area <= max_circle_size:
                    circle = np.array([pnt[0] for pnt in approx])
                    shape_area = cv2.contourArea(circle)
                    if self._detection_settings.circle_size[0] <= shape_area <= max_circle_size:
                        key_point = self._get_circle_poi(circle)

                elif contour_area >= min_rect_size:
                    # Fit a min-size box or polygon to detection
                    rect = cv2.minAreaRect(contour)
                    if abs(rect[2]) < 75:  # rotation angle
                        x, y, w, h = cv2.boundingRect(contour)
                        box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)
                        rect = cv2.minAreaRect(box)
                    box = cv2.boxPoints(rect).astype(int)
                    shape_area = cv2.contourArea(box)

                    # Define RECT key point
                    if min_rect_size <= shape_area <= self._detection_settings.rect_size[1]:
                        key_point = self._get_rect_poi(box, self.image_mono8)

                # Determine whether to save or ignore the given point
                if key_point.area != 0:
                    ok, replace_this_index = self._filter_out_detections(key_point)
                    if ok == 0:
                        index = self._detection_info.pois.__len__()
                        self._detection_info.pois[index] = key_point
                    elif ok == 1:
                        self._detection_info.pois[replace_this_index] = key_point
                    else:
                        pass

        # End tasks
        self._after_detect()


class SquareDetection(DetectionBase):
    """
    Child class for square detection, extending the functionality of DetectionBase.
    """

    def __init__(self, image_dir_or_file: str | np.ndarray, config_path: str = None, is_olaf: bool = False,
                 debug_dir: str | None = None, **kwargs):
        """
        Initialize the SquareDetection with the given parameters.

        :param config_path: Path to the configuration file for detection settings.
        :param debug_dir: Directory to save debug information. If None, uses the image directory.
        :param image_dir_or_file: Path to the image directory, a single image file, or a numpy array representing the image.
        :param is_olaf: Flag to indicate if the image is from an Olaf system.
        :param kwargs: Additional keyword arguments for the KPI filtering.
        """
        super().__init__(image_dir_or_file, config_path=config_path, is_olaf=is_olaf, debug_dir=debug_dir, **kwargs)

    def _detect(self):
        """
        Rectangular pattern detection algorithm.
        """
        # Pre-process image
        preprocessed_image = self._before_detect()

        # Find contours
        contours_found = self._get_contours(preprocessed_image)

        # Detect fiducials and SFR patterns in image
        for contour in contours_found:
            # Get geometrical characteristics
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            contour_area = cv2.contourArea(contour)

            # Filter based on size
            key_point = {}
            if self._detection_settings.rect_size[0] <= contour_area <= self._detection_settings.rect_size[1]:
                # Fit a min-size box or polygon to detection
                rect = cv2.minAreaRect(contour)
                if abs(rect[2]) < 75:  # rotation angle
                    x, y, w, h = cv2.boundingRect(contour)
                    box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)
                    rect = cv2.minAreaRect(box)
                box = cv2.boxPoints(rect).astype(int)
                shape_area = cv2.contourArea(box)
                if self._draw_contours:
                    cv2.drawContours(self.image_color8, contour, -1, self._color_txt, self._draw_size)

                # Define RECT key point
                if self._detection_settings.rect_size[0] <= shape_area <= self._detection_settings.rect_size[1]:
                    key_point = self._get_rect_poi(box, self.image_mono8)

            # Determine whether to save or ignore the given point
            if key_point.__len__() != 0:
                ok, replace_this_index = self._filter_out_detections(key_point)
                if ok == 0:
                    index = self._detection_info.pois.__len__()
                    self._detection_info.pois[index] = key_point
                elif ok == 1:
                    self._detection_info.pois[replace_this_index] = key_point
                else:
                    pass

        # End tasks
        self._after_detect()
