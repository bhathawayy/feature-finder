import os
import warnings

import cv2
import numpy as np
from scipy.signal import argrelmin

from feature_finder.data_objects import FeatureInfo


class DetectionBase:

    def __init__(self, image_array: np.ndarray):
        """
        GUI to help the user determine proper detection values for what gets entered in detection_setting.py.

        :param image_array: Image array for processing.
        """
        # Set class variables
        self._color_blob: tuple[int, int, int] = (0, 255, 0)  # Color for blobs [BGR]
        self._color_edge: tuple[int, int, int] = (0, 0, 255)  # Color for edges [BGR]
        self._color_rect: tuple[int, int, int] = (255, 0, 0)  # Color for rects [BGR]
        self._color_line_v: tuple[int, int, int] = (255, 255, 0)  # Color for vertical lines [BGR]
        self._color_line_h: tuple[int, int, int] = (255, 0, 255)  # Color for horizontal lines [BGR]
        self._draw_size: int = 4  # Edge thickness of drawn features
        self._sig_fig: int = 4  # Significant digits used when rounding

        self._contours_all: list[tuple] = []
        self._contours_non_blobs: list[tuple] = []
        self._crosshair_centers: list[tuple[int, int]] = []
        self._image_gauss: np.ndarray = np.array([])
        self._image_normal: np.ndarray = np.array([])
        self._image_thresh: np.ndarray = np.array([])
        self._lines: list = []
        self._raw_array: np.ndarray = image_array

        self.display_image: np.ndarray = np.array([])
        self.found_features: list[FeatureInfo] = []

        if self._raw_array.size > 0:
            self._image_mono8: np.ndarray = convert_color_bit(self._raw_array, color_channels=1, out_bit_depth=8)
            self._image_rgb8: np.ndarray = convert_color_bit(self._raw_array, color_channels=3, out_bit_depth=8)
        else:
            raise FileNotFoundError("Improper image input!")

    def _find_ellipses(self, ellipse_size_range: tuple[float, float], circularity_min: float = 0.5):
        """
        Once contours have found, search for blobs, then draw these on the debug image.

        :param ellipse_size_range: Range of acceptable blob areas.
        :param circularity_min: Minimum acceptable blob circularity. The closer to 1, the more "circular".
        :return: None.
        """
        self._contours_non_blobs = []
        for contour, approx, contour_area, contour_perimeter in self._contours_all:

            # Sort according to circularity
            circularity = 4 * np.pi * (contour_area / (contour_perimeter * contour_perimeter))
            if circularity >= circularity_min and len(approx) > 8:  # use approx to prevent circle fitting of rects

                # Filter based on circle size
                circle = np.array([pnt[0] for pnt in approx])
                shape_area = cv2.contourArea(circle)
                if ellipse_size_range[0] <= shape_area <= ellipse_size_range[1]:

                    # Fit & draw ellipse
                    (cx, cy), (w, h), angle = cv2.fitEllipse(contour)
                    cv2.ellipse(self.display_image, ((cx, cy), (w, h), angle), self._color_blob, self._draw_size)

                    # Save to found
                    self.found_features.append(
                        FeatureInfo(
                            area=round(shape_area, self._sig_fig),
                            centroid=(int(cx), int(cy)),
                            height=round(h, self._sig_fig),
                            shape_type="circle" if min(w, h) / max(w, h) > 0.95 else "ellipse",
                            width=round(w, self._sig_fig)
                        )
                    )

            else:
                self._contours_non_blobs.append((contour, approx, contour_area, contour_perimeter))

    def _find_contours(self, feature_size_range: tuple[float, float]):
        """
        Find contours/edges in the image.

        :param feature_size_range: Range of acceptable contour areas.
        :return: None.
        """
        # Find edges/contours
        contours_found, _ = cv2.findContours(self._image_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Filter edges/contours
        self._contours_all = []
        for contour in contours_found:

            # Filter based on largest range of acceptable sizes
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            area = cv2.contourArea(contour)
            if feature_size_range[0] < area <= feature_size_range[1]:

                # Show edge detection
                approx = cv2.approxPolyDP(contour, 1, True)
                cv2.drawContours(self.display_image, [approx], 0, self._color_edge,
                                 min(1, int(self._draw_size / 2)))

                # Save detection info
                self._contours_all.append((contour, approx, area, perimeter))

    def _find_crosshairs(self, crosshair_rotation: float = 0.0, angular_cutoff: float | None = None):
        """
        Once contours have found, search for the appropriate shapes, then draw these on the debug image.

        :param crosshair_rotation: Rotation of slope definition about origin.
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

            # Adjust tilt_angle by rotation offset for color classification
            adjusted_angle = tilt_angle - crosshair_rotation
            while adjusted_angle > 90:
                adjusted_angle -= 180
            while adjusted_angle < -90:
                adjusted_angle += 180

            # Classify as vertical or horizontal based on adjusted tilt_angle
            if abs(adjusted_angle) > 45:  # closer to vertical
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

        self._crosshair_centers = []
        for i, line1 in enumerate(self._lines):
            ep11, ep12, dx1, dy1, angle1, color1 = sort_into_slope_category(line1)  # ep = end-point

            for j, line2 in enumerate(self._lines[i + 1:], start=i + 1):
                ep21, ep22, dx2, dy2, angle2, color2 = sort_into_slope_category(line2)

                # Check if lines are perpendicular to each other
                angle_diff = abs(abs(angle1 - angle2) - 90)
                if angle_diff <= 10:

                    # Find intersection point
                    denom = dx1 * dy2 - dy1 * dx2
                    if abs(denom) > 1e-10:  # Lines are not parallel
                        t = ((ep21[0] - ep11[0]) * dy2 - (ep21[1] - ep11[1]) * dx2) / denom
                        ix = ep11[0] + t * dx1
                        iy = ep11[1] + t * dy1

                        line1_length = np.sqrt(dx1 ** 2 + dy1 ** 2)
                        line2_length = np.sqrt(dx2 ** 2 + dy2 ** 2)

                        # Save to found
                        for line_length, angle in [[line1_length, angle1], [line2_length, angle2]]:
                            if angular_cutoff is None or abs(angle) < angular_cutoff:
                                data = FeatureInfo(
                                        shape_type="line",
                                        area=round(line_length, self._sig_fig),
                                        width=round(line_length, self._sig_fig) if angle < 1 else 0,
                                        height=0 if angle < 1 else round(line_length, self._sig_fig),
                                        centroid=(int(ix), int(iy)),
                                        rotation=round(angle, self._sig_fig)
                                    )
                                if not is_duplicate(data):
                                    self.found_features.append(data)
                                    self._crosshair_centers.append((int(ix), int(iy)))

                                    # Draw shape
                                    if angular_cutoff is None or abs(angle1) < angular_cutoff:
                                        cv2.line(self.display_image, ep11, ep12, color1, self._draw_size)
                                    if angular_cutoff is None or abs(angle2) < angular_cutoff:
                                        cv2.line(self.display_image, ep21, ep22, color2, self._draw_size)

    def _find_rects(self, rectangular_size_range: tuple[float, float]):
        """
        Once contours have found, search for the appropriate shapes, then draw these on the debug image.

        :param rectangular_size_range: Range of acceptable feature areas.
        :return: None
        """
        for contour, approx, contour_area, contour_perimeter in self._contours_non_blobs:

            # Filter based on rectangle size
            box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(int)
            shape_area = cv2.contourArea(box)
            if rectangular_size_range[0] <= shape_area <= rectangular_size_range[1]:

                # Skip rectangular fitting for crosshairs
                if any(cv2.pointPolygonTest(contour, center, False) >= 0 for center in self._crosshair_centers):
                    continue

                # Draw shape
                cv2.drawContours(self.display_image, [box], 0, self._color_rect, self._draw_size)

                # Save to found
                (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
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

    def _reduce_noise(self, sigma: int = 1500) -> np.ndarray:
        """
        Reduce noise by increasing dark patterns and ignoring background contributions.

        :param sigma: Gaussian shape for brightness multiplication.
        :return: 'brightened' image
        """
        # Set local variables
        bit_max = 2 ** 8 - 1
        image = self._image_mono8 / bit_max

        # Create a 2D Gaussian kernel
        h, w = image.shape
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
        brightened_image = image.copy()
        brightened_image[mask] = np.clip(brightened_image[mask] + gaussian_kernel[mask] * 0.1, 0, 1)

        # Keep bright squares unchanged
        brightened_image[~mask] = image[~mask]

        # Scale back to 8-bit range
        brightened_image = np.clip(brightened_image * bit_max, 0, 255).astype(np.uint8)

        return brightened_image

    def apply_gauss_blur(self, gauss: int, update: bool = False) -> bool:
        """
        Applies Gaussian blur filter.

        :param gauss: Gaussian kernel value.
        :param update: Update the shown image.
        :return: Whether the shown image needs further updates.
        """
        if update or len(self._image_gauss) == 0:
            # Check input
            gauss = self._make_odd(gauss)

            # Process image
            self._image_gauss = cv2.GaussianBlur(self._image_mono8, (gauss, gauss), sigmaX=1)
            update_next = True
        else:
            update_next = False

        return update_next

    def apply_hough_transform(self, threshold: int, min_distance: int, min_line_length: int,
                              update: bool = False) -> bool:
        """
        Applies Hough Transform filter.

        :param min_distance: Minimum allowable distance between detected lines.
        :param min_line_length: Allowable length of lines.
        :param threshold: Hough thresholding value.
        :param update: Update the shown image.
        :return: Whether the shown image needs further updates.
        """
        if update or len(self._lines) == 0:
            edged = cv2.Canny(self._image_thresh, 50, 150, apertureSize=3)
            lines: np.ndarray | None = cv2.HoughLinesP(edged, 1, np.pi / 180,
                                                       maxLineGap=min_distance,
                                                       threshold=threshold,
                                                       minLineLength=min_line_length)
            if lines is not None:
                self._lines = np.reshape(np.asarray(lines), (-1, 4))  # type: ignore[arg-type]
            update_next = True
        else:
            update_next = False

        return update_next

    def apply_threshold(self, threshold: int, update: bool = False) -> bool:
        """
        Applies threshold filter.

        :param threshold: Thresholding value.
        :param update: Update the shown image.
        :return: Whether the shown image needs further updates.
        """
        if update or len(self._image_thresh) == 0:
            _, self._image_thresh = cv2.threshold(self._image_gauss, threshold, 255, cv2.THRESH_BINARY)
            update_next = True
        else:
            update_next = False

        return update_next

    def detect_features(self, feature_size_range: tuple[float, float], ellipse_size_range: tuple[float, float] = (0, 0),
                        rectangular_size_range: tuple[float, float] = (0, 0), circularity_min: float = 0.5,
                        crosshair_rotation: float = 0.0, fit_ellipse: bool = True, fit_rect: bool = True,
                        fit_crosshair: bool = True) -> list[FeatureInfo]:
        """
        Detect features i.e. ellipses, rectangular objects, and/or crosshairs.

        :param ellipse_size_range: Allowable blob size range.
        :param circularity_min: Allowable minimum blob circularity.
        :param crosshair_rotation: Rotation of slope definition about origin.
        :param feature_size_range: Allowable feature size range.
        :param fit_ellipse: Flag to fit a circle to a blob or not.
        :param fit_crosshair: Flag to fit a crosshair to a blob or not.
        :param fit_rect: Flag to fit a rectangular object to a blob or not.
        :param rectangular_size_range: Range of acceptable rect areas.
        :return: List of found features.
        """
        # Reset drawn image
        self.display_image = self._image_rgb8.copy()

        # Detect edges/contours
        self._find_contours(feature_size_range)

        # Detect ellipses
        if fit_ellipse:
            self._find_ellipses(ellipse_size_range, circularity_min=circularity_min)
        else:
            self._contours_non_blobs = self._contours_non_blobs

        # Detect crosshairs
        if fit_crosshair:
            self._find_crosshairs(crosshair_rotation)
        else:
            self._crosshair_centers = []

        # Detect rects
        if fit_rect:
            self._find_rects(rectangular_size_range)

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
