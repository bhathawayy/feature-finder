import json
import os
from typing import Type, TypeVar, Optional

import numpy as np
from pydantic import BaseModel, ValidationError, Field

T = TypeVar("T", bound="JSONConfig")


class CrosshairSettings(BaseModel):
    fit_feature: bool = Field(default=False)  # Flag for fitting feature to detections
    hough_threshold: int = Field(default=15)  # Explicit Hough line thresholding
    max_line_gap: int = Field(default=5)  # Allowable distance between detections
    max_slope: float = Field(default=0.0)  # Max allowed slope of crosshair
    min_length: int = Field(default=10)  # Allowable length of Hough lines


class EdgeSettings(BaseModel):
    gauss_blur_kernel: int = Field(default=21)  # Gaussian blur kernel size
    pixel_threshold: int = Field(default=150)  # Explicit edge thresholding
    size_range: tuple[int, int] = Field(default=(0, 700000))  # Expected size of ANY feature [(pxl^2, pxl^2)]


class EllipseSettings(BaseModel):
    circularity_min: float = Field(default=0.6)  # The closer to 1, the more "perfect" the circle is
    fit_feature: bool = Field(default=False)  # Flag for fitting feature to detections
    size_range: tuple[int, int] = Field(default=(0, 220000))  # Expected size of elliptical object [(pxl^2, pxl^2)]


class FeatureInfo(BaseModel):
    area: float = Field(default=np.nan)
    centroid: tuple[int, int] = Field(default=(np.nan, np.nan))
    height: float = Field(default=np.nan)
    rotation: float = Field(default=np.nan)
    shape_type: str = Field(default="NA")
    width: float = Field(default=np.nan)


class JSONConfig(BaseModel):

    @classmethod
    def from_file(cls: Type[T], file_path: str) -> Optional[T]:
        """
        Converts the JSON file contents to a data object.

        :param file_path: Path to the JSON file.
        :return: instance of the derived class (or None on failure)
        """
        # Create file if it does not exist (empty JSON -> {})
        if not os.path.exists(file_path):
            # Ensure parent folder exists
            parent = os.path.dirname(file_path)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("{}\n")

        msg: Optional[str] = None
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                raw = json.load(file)
                if raw is None:
                    raw = {}
                if not isinstance(raw, dict):
                    raise ValueError(f"Top-level JSON must be an object/dict, got: {type(raw).__name__}")
                return cls.model_validate(raw)

        except ValidationError as err_v:
            error_info = err_v.errors()[0] if err_v.errors() else {"type": "validation_error", "loc": ("<unknown>",)}
            msg = f"{error_info['type'].capitalize()} field(s) in {err_v.title} JSON: {error_info['loc']}"
        except json.JSONDecodeError as err_j:
            msg = f"JSON parse error in {file_path}: {err_j}"
        except PermissionError:
            msg = f"File is not accessible (may be opened elsewhere): {file_path}"
        except FileNotFoundError:
            msg = f"File could not be found at: {file_path}"
        finally:
            if msg is not None:
                raise Exception(msg)

        return None

    def to_file(self, file_path: str, write_mode: str = "w") -> None:
        """
        Converts the data object to a JSON file.

        :param file_path: Path to the JSON file.
        :param write_mode: File write mode: "w" = write, "a" = append.
        :return: None
        """
        # Ensure parent folder exists
        parent = os.path.dirname(file_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(file_path) and write_mode == "a":
            write_mode = "w"

        # Use JSON mode so datetime/Decimal/UUID serialize cleanly
        data = self.model_dump(mode="json", by_alias=True, exclude_none=False)

        try:
            with open(file_path, write_mode, encoding="utf-8") as file:
                file.write(json.dumps(data, ensure_ascii=False, indent=2))

        except PermissionError:
            raise PermissionError(f"File is not writable (may be opened elsewhere): {file_path}")


class NoiseSettings(BaseModel):
    lower_percentile: int = Field(default=1)  # Lower normalization cutoff as percentile
    normalize: bool = Field(default=True)  # Flag for reducing noise by normalization
    upper_percentile: int = Field(default=99)  # Upper normalization cutoff as percentile
    winsor_percentile: int = Field(default=85)  # Caps very bright pixels at the specified percentile


class RectSettings(BaseModel):
    fit_feature: bool = Field(default=False)  # Flag for fitting feature to detections
    size_range: tuple[int, int] = Field(default=(0, 220000))  # Expected size of rectangular object [(pxl^2, pxl^2)]


# Level 1 Dependency ------------------------------------------------------------------------------------------------ #

class FeatureSettings(BaseModel):
    crosshair: CrosshairSettings = Field(default=CrosshairSettings(), alias="crosshair")
    ellipse: EllipseSettings = Field(default=EllipseSettings(), alias="ellipse")
    rectangle: RectSettings = Field(default=RectSettings(), alias="rectangle")


# Level 2 Dependency ------------------------------------------------------------------------------------------------ #

class DetectionSettings(JSONConfig):
    edges: EdgeSettings = Field(default=EdgeSettings(), alias="edges")
    features: FeatureSettings = Field(default=FeatureSettings(), alias="features")  # Container for feature settings
    noise_handling: NoiseSettings = Field(default=NoiseSettings(), alias="noise_handling")


if __name__ == "__main__":
    from feature_finder import resources

    # Reset settings file
    DetectionSettings().to_file(os.path.join(next(iter(resources.__path__)), "detection_settings.json"))
