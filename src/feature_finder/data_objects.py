import os
from typing import Type, TypeVar, Optional

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
    """
    Needs to be the same as property names in FeatureFinder app!
    """
    # Feature detection settings
    feature_size_range: tuple = Field(default=(0, 700000))  # Expected size of any feature [(pxl^2, pxl^2)]
    gauss_blur_kernel: int = Field(default=21)  # Gaussian blur kernel size
    pixel_threshold: int = Field(default=19)  # Explicit edge thresholding

    # Feature fitting settings
    circularity_min: float = Field(default=0.6)  # The closer to 1, the more "perfect" the circle is
    crosshair_distance: int = Field(default=10)
    crosshair_hough_threshold: int = Field(default=40)  # Explicit Hough line thresholding
    crosshair_min_length: int = Field(default=10)  # Allowable length of Hough lines
    crosshair_slope_tilt: float = Field(default=0.0)  # Rotation of slope definition about origin.
    elliptical_size_range: tuple = Field(default=(0, 220000))  # Expected size of elliptical object [(pxl^2, pxl^2)]
    rectangular_size_range: tuple = Field(default=(0, 220000))  # Expected size of rectangular object [(pxl^2, pxl^2)]


class FeatureInfo(YamlConfig):
    area: float = Field(default=np.nan)
    centroid: tuple = Field(default=(np.nan, np.nan))
    height: float = Field(default=np.nan)
    rotation: float = Field(default=np.nan)
    shape_type: str = Field(default="NA")
    width: float = Field(default=np.nan)
