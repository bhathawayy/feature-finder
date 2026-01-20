import abc

import circle_fit
import cv2
import numpy as np
from scipy.signal import argrelmin

from feature_finder.processing_support import convert_color_bit, get_point_distance


class DefaultSettings:
    """
    Default detection settings. Note, the area or size parameters can typically be estimated using ImageJ. Fit a
    shape to your feature then "Measure" to return the area in pxl^2.

    USE THE HELPER GUI TO CONFIGURE!
    """
    # Parameters used for detection algorithms
    blob_size: tuple = (0, 220000)  # Expected size of fiducial [(pxl^2, pxl^2)]
    circularity_min: float = 0.8  # The closer to 1, the more "perfect" the circle is
    feature_size: tuple = (0, 700000)  # Expected size of feature (non-fiducial) [(pxl^2, pxl^2)]
    gauss: int = 21  # Gaussian blur kernel size
    range_slider_max: int = 100000  # max value of range sliders
    threshold: int = 84  # Explicit edge thresholding


class DetectionBase(abc.ABC):

    def __init__(self, image_array: np.ndarray):
        """
        GUI to help the user determine proper detection values for what gets entered in detection_setting.py.
        :param image_array: Image array for processing.
        """
        # Set class variables
        self._color_blob: tuple = (0, 255, 0)  # Color for blobs [BGR]
        self._color_edge_txt: tuple = (0, 0, 255)  # Color for edges or text [BGR]
        self._color_rect: tuple = (255, 0, 0)  # Color for rects [BGR]
        self._draw_size: int = 8
        self._deviation_cutoff: int = 100
        self._image_gauss: np.ndarray = np.array([])
        self._image_normal: np.ndarray = np.array([])
        self._image_thresh: np.ndarray = np.array([])
        self._lines: list = []
        self._raw_array: np.ndarray = image_array
        self.display_image: np.ndarray = np.array([])

        if self._raw_array.size > 0:
            self._image_mono8: np.ndarray = convert_color_bit(self._raw_array, color_channels=1, out_bit_depth=8)
            self._image_rgb8: np.ndarray = convert_color_bit(self._raw_array, color_channels=3, out_bit_depth=8)
        else:
            raise FileNotFoundError("Improper image input!")

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

    def apply_hough_transform(self, threshold: int, min_line_length: int, update: bool = False) -> bool:
        if update or len(self._lines) == 0:
            lines = cv2.HoughLinesP(self._image_thresh, 1, np.pi / 180, maxLineGap=self._deviation_cutoff,
                                    threshold=threshold, minLineLength=min_line_length)
            if lines is not None:
                self._lines = lines.reshape(-1, 4)
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

    @abc.abstractmethod
    def find_features_and_draw(self, blob_range: tuple, circularity_min: float, feature_range: tuple) -> np.ndarray:
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


class SFRDetection(DetectionBase):

    def __init__(self, image_array: np.ndarray):
        """
        SFR-specific instance (also works for blob and/or square detection).
        :param image_array: Image array for processing.
        """
        super().__init__(image_array)

    def find_features_and_draw(self, blob_range: tuple, circularity_min: float, feature_range: tuple) -> np.ndarray:
        """
        Once contours have found, fit the appropriate shape to them and draw these on the debug image.
        :return: None
        """
        # Rest drawn image
        self.display_image = self._image_rgb8.copy()

        # Find edges
        contours_found, _ = cv2.findContours(self._image_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours_found:

            # Filter based on largest range of acceptable sizes
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            contour_area = cv2.contourArea(contour)
            if min(blob_range[0], feature_range[0]) < contour_area <= max(blob_range[1], feature_range[1]):
                # Show edge detection first
                approx = cv2.approxPolyDP(contour, 1, True)
                cv2.drawContours(self.display_image, [approx], 0, self._color_edge_txt,
                                 int(self._draw_size / 2))

                # Sort according to circularity
                circularity = 4 * np.pi * (contour_area / (perimeter * perimeter))
                if circularity >= circularity_min and contour_area <= blob_range[1]:

                    # Filter based on circle size
                    circle = np.array([pnt[0] for pnt in approx])
                    shape_area = cv2.contourArea(circle)
                    if blob_range[0] <= shape_area <= blob_range[1]:
                        xc, yc, rc, sig = circle_fit.least_squares_circle(circle)
                        center = (int(xc), int(yc))
                        cv2.circle(self.display_image, center, int(rc), self._color_blob, self._draw_size)
                else:
                    # Filter based on rectangle size
                    box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(int)
                    shape_area = cv2.contourArea(box)
                    if feature_range[0] <= shape_area <= feature_range[1]:
                        cv2.drawContours(self.display_image, [box], 0, self._color_rect, self._draw_size)

        return self.display_image


class CHDetection(DetectionBase):

    def __init__(self, image_array: np.ndarray):
        """
        Crosshair-specific instance.
        :param image_array: Image array for processing.
        """
        super().__init__(image_array)

    def find_features_and_draw(self, blob_range: tuple, circularity_min: float, feature_range: tuple) -> np.ndarray:
        """
        Once contours have found, fit the appropriate shape to them and draw these on the debug image.
        :return: None
        """
        # Rest drawn image
        self.display_image = self._image_rgb8.copy()

        # Find fiducials
        contours_found, _ = cv2.findContours(self._image_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for contour in contours_found:

            # Filter based on largest range of acceptable sizes
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            contour_area = cv2.contourArea(contour)
            if blob_range[0] < contour_area <= blob_range[1]:
                # Show edge detection first
                approx = cv2.approxPolyDP(contour, 1, True)
                cv2.drawContours(self.display_image, [approx], 0, self._color_edge_txt,
                                 int(self._draw_size / 2))

                # Sort according to circularity
                circularity = 4 * np.pi * (contour_area / (perimeter * perimeter))
                if circularity >= circularity_min and contour_area <= blob_range[1]:

                    # Filter based on circle size
                    circle = np.array([pnt[0] for pnt in approx])
                    shape_area = cv2.contourArea(circle)
                    if blob_range[0] <= shape_area <= blob_range[1]:
                        xc, yc, rc, sig = circle_fit.least_squares_circle(circle)
                        center = (int(xc), int(yc))
                        cv2.circle(self.display_image, center, int(rc), self._color_blob, self._draw_size)

        # Find Hough lines
        for end_points in self._lines:
            # Calculate slope
            x1, y1, x2, y2 = end_points
            if x1 == x2:
                slope = np.inf
            else:
                slope = abs((y2 - y1) / (x2 - x1))

            # Determine color based on slope
            if slope < 1:
                color = self._color_blob  # horizontal lines
            else:
                color = self._color_edge_txt  # vertical lines

            # Draw lines
            if feature_range[0] <= get_point_distance((x1, y1), (x2, y2)) <= feature_range[1]:
                cv2.line(self.display_image, (x1, y1), (x2, y2), color, self._draw_size)

        return self.display_image
