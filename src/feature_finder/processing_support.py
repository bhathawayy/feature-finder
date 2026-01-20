import os
import warnings

import cv2
import numpy as np


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


def get_point_distance(point1: tuple | list | np.ndarray, point2: tuple | list | np.ndarray) -> float:
    """
    Get pixel distance between two coordinate points.
    :param point1: First point coordinates.
    :param point2: Second point coordinates.
    :return: Distance between points and whether this distance is less than cutoff.
    """
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    return distance
