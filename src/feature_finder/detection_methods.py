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
        self._draw_size: int = 8
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

    def _find_blobs(self, contours: list[tuple], blob_range: tuple[float, float], circularity_min: float) \
            -> tuple[list[FeatureInfo], list[tuple]]:
        """
        Once contours have found, fit the appropriate shape to them and draw these on the debug image.
        :return: None
        """
        non_blob_contours = []
        blobs_found = []
        for contour, approx, contour_area, contour_perimeter in contours:

            # Sort according to circularity
            circularity = 4 * np.pi * (contour_area / (contour_perimeter * contour_perimeter))
            if circularity >= circularity_min and contour_area <= blob_range[1]:

                # Filter based on circle size
                circle = np.array([pnt[0] for pnt in approx])
                shape_area = cv2.contourArea(circle)
                if blob_range[0] <= shape_area <= blob_range[1]:
                    # Draw shape
                    xc, yc, rc, sig = circle_fit.least_squares_circle(circle)
                    center = (int(xc), int(yc))
                    cv2.circle(self.display_image, center, int(rc), self._color_blob, self._draw_size)

                    # Save to found
                    blobs_found.append(FeatureInfo(shape_type="blob", area=shape_area, centroid=(xc, yc)))

            else:
                non_blob_contours.append((contour, approx, contour_area, contour_perimeter))

        return blobs_found, non_blob_contours

    def _find_contours(self, contour_range: tuple[float, float]) -> list[tuple]:
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
                cv2.drawContours(self.display_image, [approx], 0, self._color_edge_txt,
                                 int(self._draw_size / 2))

                # Save detection info
                contours_with_info.append((contour, approx, area, perimeter))

        return contours_with_info

    @abc.abstractmethod
    def _find_features(self, contours: list[tuple], feature_range: tuple[float, float]) -> list[FeatureInfo]:
        """
        Once contours have found, fit the appropriate shape to them and draw these on the debug image.
        :return: None
        """
        pass

    @staticmethod
    def _make_odd(val: float | int, min_val: int = 1) -> int:
        """
        Ensure variable is greater than 0 and odd.
        :param val: Value to check.
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
        if update or len(self._image_thresh) == 0:
            _, self._image_thresh = cv2.threshold(self._image_gauss, threshold, 255, cv2.THRESH_BINARY)
            update_next = True
        else:
            update_next = False

        return update_next

    def detect_features(self, feature_range: tuple[float, float], blob_range: tuple[float, float],
                        circularity_min: float) -> list[FeatureInfo]:
        # Reset drawn image
        self.display_image = self._image_rgb8.copy()

        # Detect features
        contour_range = (min(blob_range[0], feature_range[0]), max(blob_range[1], feature_range[1]))
        contours = self._find_contours(contour_range)
        blobs_found, non_blob_contours = self._find_blobs(contours, blob_range, circularity_min)
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
        Once contours have found, fit the appropriate shape to them and draw these on the debug image.
        :return: None
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
                (cx, cy), (_, _), angle = cv2.minAreaRect(contour)
                features_found.append(FeatureInfo(shape_type="rectangular",
                                                  area=shape_area,
                                                  centroid=(cx, cy),
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
        Once contours have found, fit the appropriate shape to them and draw these on the debug image.
        :return: None
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
                else:
                    color = self._color_edge_txt  # vertical lines

                # Draw shape
                cv2.line(self.display_image, (x1, y1), (x2, y2), color, self._draw_size)

                # Save to found
                features_found.append(FeatureInfo(shape_type="line",
                                                  area=line_length,
                                                  centroid=get_midpoint((x1, y1), (x2, y2)),
                                                  slope_or_tilt=slope))

        return features_found
