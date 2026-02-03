import abc

import circle_fit
import cv2
import numpy as np
from scipy.signal import argrelmin

from feature_finder.processing_support import convert_color_bit, get_point_distance, get_midpoint, FeatureInfo


class DetectionBase(abc.ABC):

    def __init__(self, image_array: np.ndarray):
        """
        GUI to help the user determine proper detection values for what gets entered in detection_setting.py.

        :param image_array: Image array for processing.
        """
        # Set class variables
        self._color_blob: tuple[int, int, int] = (0, 255, 0)  # Color for blobs [BGR]
        self._color_edge_txt: tuple[int, int, int] = (0, 0, 255)  # Color for edges or text [BGR]
        self._color_rect: tuple[int, int, int] = (255, 0, 0)  # Color for rects [BGR]
        self._deviation_cutoff: int = 100
        self._draw_size: int = 6
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

    def _find_blobs(self, contours: list[tuple], blob_range: tuple[float, float], circularity_min: float,
                    fit_blob: bool = True) -> tuple[list[FeatureInfo], list[tuple]]:
        """
        Once contours have found, search for blobs, then draw these on the debug image.

        :param blob_range: Range of acceptable blob areas.
        :param circularity_min: Minimum acceptable blob circularity. The closer to 1, the more "circular".
        :param contours: Edges found.
        :param fit_blob: Flag to fit a circle to a blob or not.
        :return: (1) List of features found, and (2) list of remaining non-blob contours.
        """

        def get_centroid(contour_points) -> tuple:
            m = cv2.moments(contour_points)
            if m["m00"] != 0:
                return m["m10"] / m["m00"], m["m01"] / m["m00"]
            pts = contour.reshape(-1, 2)

            return tuple(pts.mean(axis=0))

        non_blob_contours = []
        blobs_found = []
        for contour, approx, contour_area, contour_perimeter in contours:

            # Sort according to circularity
            circularity = 4 * np.pi * (contour_area / (contour_perimeter * contour_perimeter))
            if circularity >= circularity_min:

                # Filter based on circle size
                circle = np.array([pnt[0] for pnt in approx])
                shape_area = cv2.contourArea(circle)
                if blob_range[0] <= shape_area <= blob_range[1]:
                    if fit_blob:

                        # Test if ellipse will work better
                        (cx, cy), (w, h), angle = cv2.fitEllipse(contour)

                        # Define shape
                        if min(w, h) / max(w, h) > 0.95: # circle case is acceptable
                            cx, cy, cr, sig = circle_fit.least_squares_circle(circle)
                            center = (int(cx), int(cy))
                            cv2.circle(self.display_image, center, int(cr), self._color_blob, self._draw_size)
                            shape = "circle"
                            w = h = cr
                        else:
                            cv2.ellipse(self.display_image, ((cx, cy), (w, h), angle), self._color_blob,
                                        self._draw_size)
                            shape = "ellipse"
                    else:
                        cx, cy = get_centroid(contour)
                        (_, _), (w, h), _ = cv2.fitEllipse(contour)
                        cv2.drawContours(self.display_image, [approx], 0, self._color_blob, self._draw_size)
                        shape = "blob"

                    # Save to found
                    blobs_found.append(FeatureInfo(shape_type=shape,
                                                   area=shape_area,
                                                   width=w,
                                                   height=h,
                                                   centroid=(cx, cy)))

            else:
                non_blob_contours.append((contour, approx, contour_area, contour_perimeter))

        return blobs_found, non_blob_contours

    def _find_contours(self, contour_range: tuple[float, float], draw_contours: bool = True) -> list[tuple]:
        """
        Find contours/edges in the image.

        :param contour_range: Range of acceptable contour areas.
        :param draw_contours: Flag for drawing contours.
        :return: List of contours and relevant info.
        """
        # Find edges
        contours_found, _ = cv2.findContours(self._image_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        contours_with_info = []
        for contour in contours_found:

            # Filter based on largest range of acceptable sizes
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            area = cv2.contourArea(contour)
            if contour_range[0] < area <= contour_range[1]:
                # Show edge detection
                approx = cv2.approxPolyDP(contour, 1, True)
                if draw_contours:
                    cv2.drawContours(self.display_image, [approx], 0, self._color_edge_txt,
                                     min(1, int(self._draw_size / 2)))

                # Save detection info
                contours_with_info.append((contour, approx, area, perimeter))

        return contours_with_info

    @abc.abstractmethod
    def _find_features(self, contours: list[tuple], feature_range: tuple[float, float]) -> list[FeatureInfo]:
        """
        Once contours have found, search for the appropriate shapes, then draw these on the debug image.

        :param contours: Non-blob contours
        :param feature_range: Range of acceptable feature areas.
        :return: List of features found
        """
        pass

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

    def apply_hough_transform(self, threshold: int, min_line_length: float, update: bool = False) -> bool:
        """
        Applies Hough Transform filter.

        :param threshold: Hough thresholding value.
        :param min_line_length: Allowable length of lines.
        :param update: Update the shown image.
        :return: Whether the shown image needs further updates.
        """
        if update or len(self._lines) == 0:
            lines = cv2.HoughLinesP(self._image_thresh, 1, np.pi / 180, maxLineGap=self._deviation_cutoff,
                                    threshold=threshold, minLineLength=int(min_line_length))
            if lines is not None:
                self._lines = np.reshape(lines, (-1, 4))
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

    def detect_features(self, feature_range: tuple[float, float], blob_range: tuple[float, float],
                        circularity_min: float, draw_contours: bool = False, fit_blob: bool = True,
                        fit_rect: bool = True, fit_crosshair: bool = True) -> list[FeatureInfo]:
        """
        Detect features i.e. blobs AND rectangular or crosshairs.

        :param blob_range: Allowable blob size range.
        :param circularity_min: Allowable minimum blob circularity.
        :param draw_contours: Flag to draw contours.
        :param feature_range: Allowable feature size range.
        :param fit_blob: Flag to fit a circle to a blob or not.
        :return: List of found features.
        """
        # Reset drawn image
        self.display_image = self._image_rgb8.copy()

        # Detect features
        contour_range = (min(blob_range[0], feature_range[0]), max(blob_range[1], feature_range[1]))
        contours = self._find_contours(contour_range, draw_contours=draw_contours)
        blobs_found, non_blob_contours = self._find_blobs(contours, blob_range, circularity_min, fit_blob=fit_blob)
        features_found = self._find_features(non_blob_contours, feature_range)

        # Save all findings for reference
        self.found_features = blobs_found + features_found

        return self.found_features


class SFRDetection(DetectionBase):

    def __init__(self, image_array: np.ndarray):
        """
        SFR-specific instance (also works for blob and/or square detection).

        :param image_array: Image array for processing.
        """
        super().__init__(image_array)

    def _find_features(self, contours: list[tuple], feature_range: tuple[int, int]) -> list[FeatureInfo]:
        """
        Once contours have found, search for the appropriate shapes, then draw these on the debug image.

        :param contours: Non-blob contours
        :param feature_range: Range of acceptable feature areas.
        :return: List of features found
        """
        # Find features
        features_found = []
        for contour, approx, contour_area, contour_perimeter in contours:

            # Filter based on rectangle size
            box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(int)
            shape_area = cv2.contourArea(box)
            if feature_range[0] <= shape_area <= feature_range[1]:
                # Draw shape
                cv2.drawContours(self.display_image, [box], 0, self._color_rect, self._draw_size)

                # Save to found
                (cx, cy), (w, h), angle = cv2.minAreaRect(contour)
                features_found.append(FeatureInfo(shape_type="rectangular",
                                                  area=shape_area,
                                                  centroid=(cx, cy),
                                                  width=w,
                                                  height=h,
                                                  slope_or_tilt=angle))

        return features_found


class CHDetection(DetectionBase):

    def __init__(self, image_array: np.ndarray):
        """
        Crosshair-specific instance.

        :param image_array: Image array for processing.
        """
        super().__init__(image_array)

    def _find_features(self, contours: list[tuple], feature_range: tuple[int, int]) -> list[FeatureInfo]:
        """
        Once contours have found, search for the appropriate shapes, then draw these on the debug image.

        :param contours: Non-blob contours (Unused here)
        :param feature_range: Range of acceptable feature areas.
        :return: List of features found
        """
        # Find features
        features_found = []
        for end_points in self._lines:

            # Filter based on size
            x1, y1, x2, y2 = end_points
            line_length = get_point_distance((x1, y1), (x2, y2))
            if feature_range[0] <= line_length <= feature_range[1]:

                # Determine color based on slope
                slope = np.inf if x1 == x2 else abs((y2 - y1) / (x2 - x1))
                if slope < 1:
                    color = self._color_blob  # horizontal lines
                    w = line_length
                    h = 0
                else:
                    color = self._color_edge_txt  # vertical lines
                    w = 0
                    h = line_length

                # Draw shape
                cv2.line(self.display_image, (x1, y1), (x2, y2), color, self._draw_size)

                # Save to found
                features_found.append(FeatureInfo(shape_type="line",
                                                  area=line_length,
                                                  width=w,
                                                  height=h,
                                                  centroid=get_midpoint((x1, y1), (x2, y2)),
                                                  slope_or_tilt=slope))

        return features_found
