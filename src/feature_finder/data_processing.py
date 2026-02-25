import os
import warnings

import cv2
import numpy as np
import math

from typing_extensions import Any

from feature_finder.data_objects import FeatureInfo, DetectionSettings


class DetectionBase:

    def __init__(self, image_array: np.ndarray, detection_settings: DetectionSettings | str | None = None):
        """
        GUI to help the user determine proper detection values for what gets entered in detection_setting.py.

        :param image_array: Image array for processing.
        """
        # Set class variables
        self._color_blob: tuple[int, int, int] = (0, 255, 0)  # Color for blobs [BGR]
        self._color_edge: tuple[int, int, int] = (0, 0, 255)  # Color for edges [BGR]
        self._color_line_h: tuple[int, int, int] = (255, 0, 255)  # Color for horizontal lines [BGR]
        self._color_line_v: tuple[int, int, int] = (255, 255, 0)  # Color for vertical lines [BGR]
        self._color_rect: tuple[int, int, int] = (255, 0, 0)  # Color for rects [BGR]
        self._draw_size: int = 4  # Edge thickness of drawn features
        self._min_deviation: int = 10  # Min. deviation for filtering duplicates
        self._sig_fig: int = 4  # Significant digits used when rounding

        self._contours_all: list[tuple] = []
        self._contours_non_blobs: list[tuple] = []
        self._image_gauss: np.ndarray = np.array([])
        self._image_normal: np.ndarray = np.array([])
        self._image_thresh: np.ndarray = np.array([])
        self._lines: np.ndarray = np.array([])
        self._raw_array: np.ndarray = image_array
        self.display_image: np.ndarray = np.array([])
        self.found_features: list[FeatureInfo] = []

        # Import settings
        if isinstance(detection_settings, str) and os.path.isfile(detection_settings):
            self.settings = DetectionSettings().from_file(detection_settings)
        elif isinstance(detection_settings, DetectionSettings):
            self.settings = detection_settings
        elif detection_settings is None:
            from feature_finder import resources
            self.settings = DetectionSettings().from_file(os.path.join(next(iter(resources.__path__)),
                                                                       "detection_settings.json"))
        else:
            raise ValueError("No valid detection settings were passed!")

        # Initialize arrays
        if self._raw_array.size > 0:
            self._image_mono8: np.ndarray = convert_color_bit(self._raw_array, color_channels=1, out_bit_depth=8)
            self._image_rgb8: np.ndarray = convert_color_bit(self._raw_array, color_channels=3, out_bit_depth=8)
        else:
            raise FileNotFoundError("Improper image input!")

    def _find_ellipses(self, include: bool = True):
        """
        Once contours have found, search for blobs, then draw these on the debug image.

        :param include: Flag to include feature in findings or just use the non-blob contours.
        :return: None.
        """
        self._contours_non_blobs = []
        size_range = self.settings.features.ellipse.size_range
        for contour, approx, contour_area, contour_perimeter in self._contours_all:

            # Sort according to circularity
            circularity = contour_area / cv2.contourArea(cv2.convexHull(contour))
            if circularity >= self.settings.features.ellipse.circularity_min and len(approx) > 4:  # prevent circle fitting of rects

                # Filter based on circle size
                circle = np.array([pnt[0] for pnt in approx])
                shape_area = cv2.contourArea(circle)
                if size_range[0] <= shape_area <= size_range[1] and include:

                    # Fit & draw ellipse
                    (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
                    cv2.ellipse(self.display_image, ((cx, cy), (w, h), angle), self._color_blob, self._draw_size)

                    # Save to found
                    self.found_features.append(
                        FeatureInfo(
                            area=round(shape_area, self._sig_fig),
                            centroid=(int(cx), int(cy)),
                            height=round(h, self._sig_fig),
                            rotation=round(angle, self._sig_fig),
                            shape_type="circle" if abs(min(w, h) / max(w, h)) > 0.95 else "ellipse",
                            width=round(w, self._sig_fig)
                        )
                    )

            else:
                self._contours_non_blobs.append((contour, approx, contour_area, contour_perimeter))

    def _find_contours(self):
        """
        Find contours/edges in the image.

        :return: None.
        """
        # Find edges/contours
        contours_found, _ = cv2.findContours(self._image_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Filter edges/contours
        self._contours_all = []
        size_range = self.settings.edges.size_range
        for contour in contours_found:

            # Filter based on largest range of acceptable sizes
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            area = cv2.contourArea(contour)
            if size_range[0] < area <= size_range[1]:

                # Show edge detection
                approx = cv2.approxPolyDP(contour, 1, True)
                cv2.drawContours(self.display_image, [approx], 0, self._color_edge,
                                 min(1, int(self._draw_size / 2)))

                # Save detection info
                self._contours_all.append((contour, approx, area, perimeter))

    def _find_crosshairs(self):
        """
        Once contours have found, search for the appropriate shapes, then draw these on the debug image.

        :return: None
        """

        def sort_into_slope_category(line_end_points):
            x1, y1, x2, y2 = line_end_points

            # Calculate slope category for line1
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 1:  # vertical line
                tilt_angle = 90
            else:
                tilt_angle = np.degrees(np.arctan(dy / dx))

            # Classify as vertical or horizontal based on adjusted tilt_angle
            if abs(tilt_angle) > 45:  # closer to vertical
                color = self._color_line_v
            else:  # closer to horizontal
                color = self._color_line_h

            return (x1, y1), (x2, y2), dx, dy, tilt_angle, color

        def is_duplicate(new_feature):
            for existing in self.found_features:
                if (existing.area == new_feature.area and 
                    existing.width == new_feature.width and 
                    existing.height == new_feature.height and 
                    existing.rotation == new_feature.rotation):
                    dist = np.sqrt((existing.centroid[0] - new_feature.centroid[0])**2 + 
                                   (existing.centroid[1] - new_feature.centroid[1])**2)
                    if dist <= 50:
                        return True
            return False

        angular_cutoff = self.settings.features.crosshair.max_slope
        self._crosshair_centers = []
        for i, line1 in enumerate(self._lines):
            ep11, ep12, dx1, dy1, angle1, color1 = sort_into_slope_category(line1)  # ep = end-point

            for j, line2 in enumerate(self._lines[i + 1:], start=i + 1):
                ep21, ep22, dx2, dy2, angle2, color2 = sort_into_slope_category(line2)

                # Check if lines are perpendicular to each other
                angle_diff = abs(abs(angle1 - angle2) - 90)
                if angle_diff <= self._min_deviation:

                    # Find intersection point
                    denominator = dx1 * dy2 - dy1 * dx2
                    if abs(denominator) > 1e-10:  # Lines are not parallel
                        t = ((ep21[0] - ep11[0]) * dy2 - (ep21[1] - ep11[1]) * dx2) / denominator
                        ix = ep11[0] + t * dx1
                        iy = ep11[1] + t * dy1

                        line1_length = np.sqrt(dx1 ** 2 + dy1 ** 2)
                        line2_length = np.sqrt(dx2 ** 2 + dy2 ** 2)
                        pos1 = np.sqrt((ix - ep11[0]) ** 2 + (iy - ep11[1]) ** 2) / line1_length
                        pos2 = np.sqrt((ix - ep21[0]) ** 2 + (iy - ep21[1]) ** 2) / line2_length

                        # Only accept if intersection is in middle third of both lines (middle 30%)
                        if 0.3 <= pos1 <= 0.7 and 0.3 <= pos2 <= 0.7:

                            # Save to found
                            for line_length, angle in [[line1_length, angle1], [line2_length, angle2]]:
                                if angular_cutoff == 0 or abs(angle) < angular_cutoff:
                                    data = FeatureInfo(
                                            area=round(line_length, self._sig_fig),
                                            centroid=(int(ix), int(iy)),
                                            height=0 if angle < 1 else round(line_length, self._sig_fig),
                                            rotation=round(angle, self._sig_fig),
                                            shape_type=f"{'horizontal' if angle < 1 else 'vertical'} line",
                                            width=round(line_length, self._sig_fig) if angle < 1 else 0
                                        )
                                    if not is_duplicate(data):
                                        self.found_features.append(data)

                            # Draw shape
                            if angular_cutoff == 0 or abs(angle1) < angular_cutoff:
                                cv2.line(self.display_image, ep11, ep12, color1, self._draw_size)
                            if angular_cutoff == 0 or abs(angle2) < angular_cutoff:
                                            cv2.line(self.display_image, ep21, ep22, color2, self._draw_size)

    def _find_rects(self):
        """
        Once contours have found, search for the appropriate shapes, then draw these on the debug image.

        :return: None
        """
        def distance(p1: tuple[float | int, float | int], p2: tuple[float | int, float | int]) -> float:
            return round(math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2), self._sig_fig)

        size_range = self.settings.features.rectangle.size_range
        for contour, approx, contour_area, contour_perimeter in self._contours_non_blobs:

            # Filter based on rectangle size
            box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(int)
            shape_area = cv2.contourArea(box)
            if size_range[0] <= shape_area <= size_range[1] and len(approx) == 4:  # prevent rect fitting of crosshairs

                # Skip rectangular fitting for crosshairs
                (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
                if any(distance((cx, cy), p.centroid) < self._min_deviation for p in
                       self.found_features if p.shape_type == "line"):
                    continue

                # Draw shape
                cv2.drawContours(self.display_image, [box], 0, self._color_rect, self._draw_size)

                # Save to found
                self.found_features.append(
                    FeatureInfo(
                        area=round(shape_area, self._sig_fig),
                        centroid=(int(cx), int(cy)),
                        height=round(h, self._sig_fig),
                        rotation=round(angle, self._sig_fig),
                        shape_type="rectangular",
                        width=round(w, self._sig_fig)
                    )
                )

    @staticmethod
    def _make_odd(val: float | int, min_val: int = 1) -> int:
        """
        Ensure variable is greater than 0 and odd.

        :param val: Value to check.
        :param min_val: Minimum allowable value.
        :return: Odd integer, greater than 0.
        """
        if val < min_val:
            val = min_val
        elif int(val) % 2 == 0:
            val += 1

        return int(val)

    def _reset(self, reset_images: bool = True):
        """
        Reset variables that are internally referenced and built upon.

        :return: None
        """
        self._contours_all = []
        self._contours_non_blobs = []
        self._lines = np.array([])
        self.display_image = self._image_rgb8.copy()
        self.found_features = []

        if reset_images:
            self._image_gauss = np.array([])
            self._image_normal = np.array([])
            self._image_thresh = np.array([])

    def apply_gauss_blur(self, update: bool = True) -> bool:
        """
        Applies Gaussian blur filter.

        :param update: Update the shown image.
        :return: Whether the shown image needs further updates.
        """
        if update or len(self._image_gauss) == 0:
            # Check input
            gauss = self._make_odd(self.settings.edges.gauss_blur_kernel)

            # Process image
            self._image_gauss = cv2.GaussianBlur(self._image_mono8, (gauss, gauss), sigmaX=1)
            update_next = True
        else:
            update_next = False

        return update_next

    def apply_hough_transform(self, update: bool = True) -> bool:
        """
        Applies Hough Transform filter.

        :param update: Update the shown image.
        :return: Whether the shown image needs further updates.
        """
        if update or len(self._lines) == 0:
            lines: np.ndarray | None = cv2.HoughLinesP(self._image_thresh, 1, np.pi / 180,
                                                       maxLineGap=self.settings.features.crosshair.max_line_gap,
                                                       threshold=self.settings.features.crosshair.hough_threshold,
                                                       minLineLength=self.settings.features.crosshair.min_length)
            if lines is not None:
                self._lines = np.reshape(np.asarray(lines), (-1, 4))  # type: ignore[arg-type]
            update_next = True
        else:
            update_next = False

        return update_next

    def apply_threshold(self, update: bool = True) -> bool:
        """
        Applies threshold filter.

        :param update: Update the shown image.
        :return: Whether the shown image needs further updates.
        """
        if update or len(self._image_thresh) == 0:
            threshold = self.settings.edges.pixel_threshold
            _, self._image_thresh = cv2.threshold(self._image_gauss, threshold, 255, cv2.THRESH_BINARY)
            update_next = True
        else:
            update_next = False

        return update_next

    def detect_features(self, preprocess_image: bool = True) -> list[FeatureInfo]:
        """
        Detect features i.e. ellipses, rectangular objects, and/or crosshairs.

        :return: List of found features.
        """
        # Reset variables
        self._reset(reset_images=preprocess_image)

        # Preprocess image (order matters!)
        if preprocess_image:
            self.apply_gauss_blur()
            self.apply_hough_transform()
            if self.settings.features.crosshair.fit_feature:
                self.apply_threshold()

        # Detect edges/contours
        self._find_contours()

        # Detect ellipses
        self._find_ellipses(include=self.settings.features.ellipse.fit_feature)

        # Detect crosshairs
        if self.settings.features.crosshair.fit_feature:
            self._find_crosshairs()

        # Detect rects
        if self.settings.features.rectangle.fit_feature:
            self._find_rects()

        return self.found_features


def check_path(target_path: str, overwrite: bool = True) -> str:
    """
    Checks if directory and path exists, if not it creates one.

    :param overwrite: Overwrite the file (Ture) or not (False).
    :param target_path: Path to file.
    :return: Unique path.
    """
    # Get directory path
    target_split = os.path.splitext(target_path)
    if len(target_split[1]) > 0:
        target_dir = os.path.dirname(target_path)
        target_file = os.path.basename(target_path)
    else:
        target_dir = target_split[0]
        target_file = target_split[1]

    # Check validity of directory
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
        except PermissionError:
            target_dir = os.path.join(os.getcwd(), "Debug")
            warnings.warn(f"Lacking write permissions. Saving locally instead: {target_dir}")
        except FileExistsError:
            pass
    target_path = os.path.join(target_dir, target_file)

    # Check if path already exists, add (#) to name
    if not overwrite and os.path.isfile(target_path):
        count = 1
        temp = target_file.split(".")
        file_name, file_ext = (".".join(temp[:-1]), temp[-1])
        new_file_name = "%s (%i).%s" % (file_name, count, file_ext)
        while os.path.exists(os.path.join(target_dir, new_file_name)):
            count += 1
            new_file_name = "%s (%i).%s" % (file_name, count, file_ext)
        target_path = os.path.join(target_dir, new_file_name)

    return target_path


def convert_color_bit(image: np.ndarray | str, color_channels: int = None, out_bit_depth: int = None,
                      in_bit_depth: int = None) -> np.ndarray:
    """
    Converts image array into RGB/Monochrome with specified bit-depth.

    :param color_channels: Color descriptor options: 3 = RGB, 1 = Monochrome
    :param image: Image array to be processed.
    :param in_bit_depth: Bit-depth options: 8, 12, 16
    :param out_bit_depth: Bit-depth options: 8, 16
    :return: Converted image array.
    """
    # Check input image
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    converted_array = np.array(image).copy()

    if converted_array.size != 0:
        # Initialize local variables
        if in_bit_depth is None:
            in_bit_depth = converted_array.dtype.name
        else:
            in_bit_depth = str(in_bit_depth)
        current_shape = converted_array.shape

        # Convert to desired bit-depth
        if out_bit_depth is not None:
            if "16" in in_bit_depth and "8" in str(out_bit_depth):  # 16-bit to 8-bit
                converted_array = converted_array.astype(float)
                converted_array = (converted_array / (2 ** 8)).astype('uint8')
            elif "12" in in_bit_depth and "16" in str(out_bit_depth):  # 12-bit to 16-bit
                converted_array = converted_array.astype(float)
                converted_array = (converted_array / (2 ** 4)).astype('uint8')
            elif "8" in in_bit_depth and "16" in str(out_bit_depth):  # 8-bit to 16-bit
                converted_array = converted_array.astype(float)
                converted_array = (converted_array * (2 ** 8)).astype('uint16')

        # Convert to desired color
        if color_channels is not None:
            if len(current_shape) == 2 and color_channels == 3:
                converted_array = cv2.cvtColor(converted_array, cv2.COLOR_GRAY2RGB)
            elif len(current_shape) == 3 and color_channels == 1:
                if current_shape[-1] == 3:
                    converted_array = cv2.cvtColor(converted_array, cv2.COLOR_RGB2GRAY)
                else:
                    converted_array = cv2.cvtColor(converted_array, cv2.COLOR_RGBA2GRAY)
            elif len(current_shape) == 3 and color_channels == 3 and current_shape[-1] == 4:
                converted_array = cv2.cvtColor(converted_array, cv2.COLOR_RGBA2RGB)

    return converted_array
