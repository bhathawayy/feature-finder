import abc

import numpy as np
from MountOlympus.Processing.helper import (convert_color_bit, interpolate_array, find_outliers)
from MountOlympus.__init__ import *
from matplotlib import pyplot
from pyml.pyml import (Mat, RoiLevel)
from scipy.integrate import simpson
from scipy.signal.windows import hann
from sklearn.cluster import KMeans


# Parent Class ------------------------------------------------------------------------------------------------------ #
class MTFBase:
    """
    Base class for Modulation Transfer Function (MTF) calculations.
    """

    def __init__(self, roi_array: np.ndarray, pixel_size_mm: float = 0.00345, efl_mm: float = 12.297,
                 roi_name: str = "0NA", debug_dir: str | None = None):
        """
        Initialize the MTFBase class.

        :param roi_array: Region of Interest (ROI) array for MTF calculation.
        :param pixel_size_mm: Size of each pixel in millimeters.
        :param efl_mm: Effective focal length in millimeters.
        :param roi_name: Name of the ROI for identification.
        :param debug_dir: Directory path for saving debug plots.
        """
        self._csf_integral: float = 37.1722476
        self._debug_dir: str = debug_dir
        self._freq_length: int = 256
        self._nyquist_freq: float = 16
        self._pixel_pitch: float = np.rad2deg(np.arctan(pixel_size_mm / efl_mm))
        self._roi_name: str = roi_name
        self.debug_mode: bool = self._debug_dir is not None and self._debug_dir != ""
        self.half_nyquist_freq: float = self._nyquist_freq / 2
        self.roi_array = self._check_input_image(roi_array)

    @property
    def debug_dir(self) -> str:
        """
        Property to get or set the debug directory path.

        :return: Path to the debug directory.
        """
        # Use image directory if none was provided
        if self._debug_dir is None or self._debug_dir == "":
            debug_path = os.getcwd()
        else:
            debug_path = self._debug_dir

        # Add folder to path
        if os.path.basename(debug_path) != "Plots":
            if os.path.basename(debug_path) == "MTF":
                debug_path = os.path.join(debug_path, "Plots")
            else:
                debug_path = os.path.join(debug_path, "MTF", "Plots")

        # Make directory if it doesn't exist
        if not os.path.isdir(debug_path):
            try:
                os.makedirs(debug_path)
            except PermissionError:
                debug_path = os.path.join(os.getcwd(), "Debug")
                warnings.warn(f"Lacking write permissions. Using local instead: {debug_path}")

        # Set internal reference
        self._debug_dir = debug_path

        return self._debug_dir

    @abc.abstractmethod
    def get_mtf_and_acutance(self) -> tuple:
        """
        Abstract method to be implemented by child classes for calculating MTF and acutance.

        :return: Tuple containing frequency, MTF, acutance, and saturation status.
        """
        pass

    @staticmethod
    def _check_input_image(roi_array: np.ndarray):
        """
        Check and convert the input image to the required format.

        :param roi_array: Input image array.
        :return: Converted image array.
        """
        if roi_array.size > 0:
            if len(roi_array.shape) == 3 or "16" not in roi_array.dtype.name:
                roi_array = convert_color_bit(roi_array, color_channels=1, out_bit_depth=16)
            else:
                roi_array = roi_array
        else:
            raise ValueError("Input ROI is empty!")

        return roi_array

    @staticmethod
    def _check_saturation(mono16_array: np.ndarray, tolerance: float = 0.01) -> bool:
        """
        Test if the image is saturated.

        :param mono16_array: Image array to be processed.
        :param tolerance: Saturated pixel tolerance (%).
        :return: Whether the image is saturated (True) or not (False)
        """
        # Define saturation pixel amount and value limits
        total_pixels = mono16_array.shape[0] * mono16_array.shape[1]
        pixel_value_limit = int((2 ** 16 - 256) * 0.95)
        total_pixels_limit = round(tolerance / 100 * total_pixels)

        # Flatten frame to examine all entries
        flat_frame = np.array(mono16_array.flatten())

        # Determine if saturated or not based on conditions
        is_saturated = bool(len(flat_frame[flat_frame >= pixel_value_limit]) >= total_pixels_limit)

        return is_saturated


# Child Classes ----------------------------------------------------------------------------------------------------- #
class LineSpreadMTF(MTFBase):
    """
    Class for calculating MTF using the Line Spread Function (LSF) method.
    """

    def __init__(self, roi_array: np.ndarray, dark_roi_array: np.ndarray | None, slit_width_pxl: float = 1.1029,
                 pixel_size_mm: float = 0.00345, efl_mm: float = 12.297, roi_name: str = "0NA",
                 image_name: str = "", debug_dir: str | None = None):
        """
        Initialize the LineSpreadMTF class.

        :param roi_array: Region of Interest (ROI) array for MTF calculation.
        :param dark_roi_array: Dark frame ROI array for background subtraction.
        :param slit_width_pxl: Width of the slit in pixels.
        :param pixel_size_mm: Size of each pixel in millimeters.
        :param efl_mm: Effective focal length in millimeters.
        :param roi_name: Name of the ROI for identification.
        :param debug_dir: Directory path for saving debug plots.
        """
        super().__init__(roi_array, pixel_size_mm=pixel_size_mm, efl_mm=efl_mm, roi_name=roi_name, debug_dir=debug_dir)

        self._debug_save_aligned_lsf: bool = True
        self._debug_save_lsf: bool = False
        self._debug_save_sinc: bool = False
        self._image_name: str = image_name
        self._slit_width_pxl: float = slit_width_pxl
        self.dark_roi_array = self._check_input_image(dark_roi_array)

        # Determine LSF direction
        if roi_array.shape[0] >= roi_array.shape[1]:
            self._mtf_direction: str = "H"
        else:
            self._mtf_direction: str = "V"

    def get_mtf_and_acutance(self, ignore_saturation: bool = False) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Calculate MTF and acutance using the Line Spread Function method.

        :param ignore_saturation: If True, ignore saturation check.
        :return: Tuple containing frequency, MTF, acutance, and saturation status.
        """
        # Set local variables
        acutance: float = np.nan
        out_frequency: np.ndarray = np.ones(self._freq_length) * np.nan
        out_mtf: np.ndarray = np.ones(self._freq_length) * np.nan

        # Check saturation
        is_saturated = self._check_saturation(self.roi_array, tolerance=1)

        # Get LSF and apply FFT
        if not is_saturated or ignore_saturation:
            lsf = self._get_lsf()
            out_mtf, out_frequency = self._apply_fft_to_lsf(lsf)

            # Calculate acutance
            _, acutance = RoiLevel.Acutance(out_frequency, out_mtf, self._nyquist_freq, self._csf_integral)

        return out_frequency, out_mtf, acutance, is_saturated

    def _align_and_average_lsf(self, lsf_matrix: np.ndarray) -> np.ndarray:
        """
        Align the peaks of each LSF slice and then average to get a single LSF.

        :param lsf_matrix: Matrix of LSF curves.
        :return: The averaged LSF.
        """
        # Find the peak index for each array
        peak_indices = [np.argmax(array) for array in lsf_matrix]

        # Determine the reference peak position (mean of all peak indices)
        mean_peak_index = int(np.nanmean(peak_indices))

        # Align all arrays to the reference peak
        aligned_arrays = []
        for i, array in enumerate(lsf_matrix):
            shift = mean_peak_index - peak_indices[i]
            aligned_array = np.roll(array, shift)
            aligned_arrays.append(aligned_array)

        # Plot aligned LSFs
        if self.debug_mode and self._debug_save_aligned_lsf:
            pyplot.figure(figsize=(12, 6))
            for ary in aligned_arrays:
                pyplot.plot(ary)
            pyplot.title(f"Aligned Line Spread Functions (LSF) for ROI #{self._roi_name}")
            pyplot.xlabel("Pixel Number")
            pyplot.ylabel("Pixel Value")
            if self._mtf_direction == "V":
                pyplot.xlim((0, self.roi_array.shape[0]))
            else:
                pyplot.xlim((0, self.roi_array.shape[1]))
            pyplot.ylim((0, 0.5))
            pyplot.grid(True)
            pyplot.savefig(os.path.join(self.debug_dir, f"{self._image_name}_LSF_ROI_{self._roi_name}_ALIGNED.png"))
            pyplot.close()

        # Average the aligned arrays
        averaged_lsf = np.nanmean(aligned_arrays, axis=0)

        return averaged_lsf

    def _apply_fft_to_lsf(self, lsf: np.ndarray) -> tuple:
        """
        Apply the Fast Fourier Transform (FFT) to the averaged LSFs to get MTF.

        :param lsf: LSF matrix.
        :return: Tuple containing MTFs and associated frequencies.
        """
        # Init local variables
        out_frequency: list | np.ndarray = []
        out_mtf: list | np.ndarray = []

        # Apply Fast Fourier Transform
        fft_result = np.fft.fft(lsf)
        fft_freq = np.fft.fftfreq(len(lsf))
        fft_mtf = np.abs(fft_result)

        # Only consider the MTF values where frequency is >= 0
        for n, freq in enumerate(fft_freq):
            if freq >= 0:
                out_mtf.append(fft_mtf[n])
                out_frequency.append(fft_freq[n])
        out_mtf = np.array(out_mtf)
        out_frequency = np.array(out_frequency)

        # Define variables for sinc correction
        sinc_function_1 = np.sinc(np.multiply(out_frequency, self._slit_width_pxl))
        sinc_function_2 = np.sinc(out_frequency)
        sinc_correction = sinc_function_1 * sinc_function_2
        sinc_correction[sinc_correction == 0] = 1e-10

        # Plot the sinc corrections
        if self.debug_mode and self._debug_save_sinc:
            pyplot.figure(figsize=(12, 6))
            pyplot.plot(out_frequency, sinc_function_1, label="S1 = sinc(f*x)")
            pyplot.plot(out_frequency, sinc_function_2, label="S2 = sinc(f)")
            pyplot.plot(out_frequency, sinc_correction, label="Correction (S1*S2)")
            pyplot.title(f"Sinc Functions for Sampling Correction")
            pyplot.xlabel("Frequency (cyc/deg)")
            pyplot.ylabel("a.u.")
            pyplot.legend()
            pyplot.savefig(os.path.join(self.debug_dir, f"{self._image_name}_SINC_CORRECTION.png"))
            pyplot.close()

        # Correct for discrete scaling factor & normalize
        out_frequency /= self._pixel_pitch
        out_mtf /= sinc_correction
        out_mtf /= fft_mtf[0]

        # Interpolate the MTF and frequencies
        out_mtf = interpolate_array(out_mtf, self._freq_length)
        out_frequency = interpolate_array(out_frequency, self._freq_length)

        return out_mtf, out_frequency

    def _get_lsf(self, sd_min: float = 0.02):
        """
        Calculate the Line Spread Function (LSF) for every row/column in the image.

        :param sd_min: Defines the minimum standard deviation a collection of LSF curves need to proceed without
        outlier removal.
        :return: LSF matrix for cropped image.
        """
        # Init local variables
        lsf_matrix = []
        if self._mtf_direction.upper()[0] == "H":
            dimension = self.roi_array.shape[0]
        else:
            dimension = self.roi_array.shape[1]

        # Create plot (if needed)
        if self.debug_mode and self._debug_save_lsf:
            pyplot.figure(figsize=(12, 6))

        # Define LSF matrix
        for pxl_rc in range(1, dimension):
            if self._mtf_direction.upper()[0] == "H":
                lsf = self.roi_array[pxl_rc, :]
            else:
                lsf = self.roi_array[:, pxl_rc]

            # Background subtraction
            dark_val = np.nanmean(self.dark_roi_array)
            lsf = lsf.astype(float) - dark_val  # dark subtraction

            # Apodization/Band-Pass Filtering
            lsf = lsf * hann(len(lsf))
            lsf_area = simpson(lsf, x=np.arange(len(lsf)))

            # Normalize
            if lsf_area != 0:
                lsf = lsf / lsf_area
            lsf_matrix.append(lsf)

            # Add to plot (if needed)
            if self.debug_mode and self._debug_save_lsf:
                pyplot.plot(lsf, label=pxl_rc)
        lsf_matrix = np.array(lsf_matrix)

        # Account for missing CH
        sd = np.nanmean(np.nanstd(lsf_matrix, axis=0))
        if sd > sd_min:
            # print(f"Unusual LSF curve(s) detected for {roi_title}. Removing outliers...")
            cluster_num = 1
            majority_arrays = []
            while sd > sd_min:
                kmeans = KMeans(n_clusters=cluster_num)
                kmeans.fit(lsf_matrix)
                labels = kmeans.labels_
                majority_label = np.bincount(labels).argmax()
                majority_arrays = [array for array, label in zip(lsf_matrix, labels) if label == majority_label]
                cluster_num += 1
                sd = np.nanmean(np.std(majority_arrays, axis=0))
            lsf_matrix = np.array(majority_arrays)

        # Plot all LSFs
        if self.debug_mode and self._debug_save_lsf:
            pyplot.title(f"Line Spread Functions (LSF) for ROI #{self._roi_name}")
            pyplot.xlabel("Pixel Number")
            pyplot.ylabel("Pixel Value")
            if self._mtf_direction == "V":
                pyplot.xlim((0, self.roi_array.shape[0]))
            else:
                pyplot.xlim((0, self.roi_array.shape[1]))
            pyplot.ylim((0, 0.5))
            pyplot.grid(True)
            pyplot.savefig(os.path.join(self.debug_dir, f"{self._image_name}_LSF_ROI_{self._roi_name}.png"))
            pyplot.close()

        return self._align_and_average_lsf(lsf_matrix)


class SlantEdgeMTF(MTFBase):
    """
    Class for calculating MTF using the Slant Edge method.
    """

    def __init__(self, roi_array: np.ndarray, osf: float = 4, pixel_size_mm: float = 0.00345, efl_mm: float = 12.297,
                 roi_name: str = "0NA", debug_dir: str | None = None):
        """
        Initialize the SlantEdgeMTF class.

        :param roi_array: Region of Interest (ROI) array for MTF calculation.
        :param osf: Oversampling factor for edge detection.
        :param pixel_size_mm: Size of each pixel in millimeters.
        :param efl_mm: Effective focal length in millimeters.
        :param roi_name: Name of the ROI for identification.
        :param debug_dir: Directory path for saving debug plots.
        """
        super().__init__(roi_array, pixel_size_mm=pixel_size_mm, efl_mm=efl_mm, roi_name=roi_name, debug_dir=debug_dir)
        self._osf: float = osf

    def get_mtf_and_acutance(self, ignore_saturation: bool = False) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Calculate MTF and acutance using the Slant Edge method.

        :param ignore_saturation: If True, ignore saturation check.
        :return: Tuple containing frequency, MTF, acutance, and saturation status.
        """
        # Set local variables
        acutance: float = np.nan
        out_frequency: np.ndarray = np.ones(self._freq_length) * np.nan
        out_mtf: np.ndarray = np.ones(self._freq_length) * np.nan

        # Check saturation
        is_saturated = self._check_saturation(self.roi_array, tolerance=1)

        # Call PyML for calculations
        if not is_saturated or ignore_saturation:
            # Remove outliers?
            for outlier in find_outliers(self.roi_array, threshold=8):
                self.roi_array[outlier[0]][outlier[1]] = np.median(self.roi_array)

            # Calculate MTF
            roi_pyml_mat = Mat.from_array(np.array(self.roi_array))
            ok, freqs, mtfs, _ = RoiLevel.JSlantEdge(roi_pyml_mat, self._osf, self._pixel_pitch)
            out_frequency = interpolate_array(np.array(freqs), self._freq_length)
            out_mtf = interpolate_array(np.array(mtfs), self._freq_length)

            # Calculate acutance
            if ok and len(freqs) != 0 and len(mtfs) != 0:
                _, acutance = RoiLevel.Acutance(freqs, mtfs, self._nyquist_freq, self._csf_integral)

        return out_frequency, out_mtf, acutance, is_saturated
