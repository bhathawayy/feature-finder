import os
import warnings
from typing import Type, TypeVar, Optional

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, ValidationError, Field

T = TypeVar("T", bound="YamlConfig")


class YamlConfig(BaseModel):

    @classmethod
    def from_yaml(cls: Type[T], path2yaml: str) -> Optional[T]:
        """
        Converts the YAML file contents to a data object.

        :param path2yaml: Path to the YAML file.
        :return: instance of the derived class (or None on failure)
        """
        # Create file if it does not exist (empty YAML -> {})
        if not os.path.exists(path2yaml):
            # Ensure parent folder exists
            parent = os.path.dirname(path2yaml)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            with open(path2yaml, "w", encoding="utf-8") as f:
                f.write("{}\n")

        msg: Optional[str] = None
        try:
            with open(path2yaml, "r", encoding="utf-8") as file:
                raw = yaml.safe_load(file) or {}
                # Pydantic v2 idiom: classmethod validate
                # (If you prefer __init__ kwargs, cls(**raw) also works.)
                return cls.model_validate(raw)
        except ValidationError as er2:
            error_info = er2.errors()[0] if er2.errors() else {"type": "validation_error", "loc": ("<unknown>",)}
            msg = f"{error_info['type'].capitalize()} field(s) in {er2.title} YAML: {error_info['loc']}"
        except yaml.YAMLError as ye:
            msg = f"YAML parse error in {path2yaml}: {ye}"
        except PermissionError:
            msg = f"File is not accessible (may be opened elsewhere): {path2yaml}"
        except FileNotFoundError:
            msg = f"File could not be found at: {path2yaml}"
        finally:
            if msg is not None:
                raise Exception(msg)

        return None

    def to_yaml(self, path2yaml: str, write_mode: str = "w") -> None:
        """
        Converts the data object to a YAML file.

        :param path2yaml: Path to the YAML file.
        :param write_mode: File write mode: "w" = write, "a" = append.
        :return: None
        """
        # Ensure parent folder exists
        parent = os.path.dirname(path2yaml)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(path2yaml) and write_mode == "a":
            write_mode = "w"

        # Use JSON mode so datetime/Decimal/UUID serialize cleanly
        data = self.model_dump(mode="json", by_alias=True, exclude_none=False)

        try:
            with open(path2yaml, write_mode, encoding="utf-8") as file:
                yaml.safe_dump(
                    data,
                    file,
                    sort_keys=False,
                    allow_unicode=True,
                    indent=2,
                    default_flow_style=False,
                )
        except PermissionError:
            raise PermissionError(f"File is not writable (may be opened elsewhere): {path2yaml}")


class DefaultSettings(YamlConfig):
    blob_size: tuple = Field(default=(0, 220000))  # Expected size of fiducial [(pxl^2, pxl^2)]
    circularity_min: float = Field(default=0.8)  # The closer to 1, the more "perfect" the circle is
    feature_size: tuple = Field(default=(0, 700000))  # Expected size of feature (non-fiducial) [(pxl^2, pxl^2)]
    gauss: int = Field(default=21)  # Gaussian blur kernel size
    range_slider_max: int = Field(default=100000)  # max value of range sliders
    pixel_threshold: int = Field(default=84)  # Explicit edge thresholding
    hough_threshold: int = Field(default=80)  # Explicit Hough line thresholding


class FeatureInfo(YamlConfig):
    area: float = Field(default=0.0)
    centroid: tuple = Field(default=(0, 0))
    shape_type: str = Field(default="NA")
    slope_or_tilt: float = Field(default=0.0)


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


def get_midpoint(point1: tuple | list | np.ndarray, point2: tuple | list | np.ndarray) -> tuple[float, float]:
    """
    Return the midpoint between two 2D coordinates.

    :param point1: First point coordinates.
    :param point2: Second point coordinates.
    :return: Midpoint between two 2D coordinates.
    """
    return (point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2


def get_point_distance(point1: tuple | list | np.ndarray, point2: tuple | list | np.ndarray) -> float:
    """
    Get pixel distance between two coordinate points.

    :param point1: First point coordinates.
    :param point2: Second point coordinates.
    :return: Distance between points and whether this distance is less than cutoff.
    """
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    return distance
