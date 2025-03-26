from dataclasses import (dataclass, field)

import numpy as np

from detection_methods import DefaultSettings


@dataclass
class MTFKPI:
    acutance: float = field(default_factory=lambda: np.nan)
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    half_nyquist_freq: float = field(default_factory=lambda: np.nan)
    half_nyquist_mtf: float = field(default_factory=lambda: np.nan)
    is_saturated: bool | None = field(default_factory=lambda: None)
    mtfs: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ROI:
    array: np.ndarray = field(default_factory=lambda: np.array([]))
    center: tuple[int] = field(default_factory=lambda: (0, 0))
    corners: list[tuple] = field(default_factory=lambda: [(0, 0), (0, 0)])  # Alt: endpoints
    mtf_direction: str = field(default_factory=lambda: "Horizontal")
    mtf_kpi: MTFKPI = field(default_factory=MTFKPI)
    size_wh: tuple[int] = field(default_factory=lambda: (0, 0))


@dataclass
class ROIInfo:
    dark: ROI = field(default_factory=ROI)
    east: ROI = field(default_factory=ROI)
    north: ROI = field(default_factory=ROI)
    south: ROI = field(default_factory=ROI)
    west: ROI = field(default_factory=ROI)


@dataclass
class LineInfo:
    direction: str = field(default_factory=lambda: "Horizontal")
    fit: np.polyfit = field(default_factory=lambda: np.poly1d([]))
    r_squared: float = field(default_factory=lambda: np.nan)
    slope: float = field(default_factory=lambda: np.nan)
    probes: list = field(default_factory=lambda: [])


@dataclass
class POIInfo:
    area: float = field(default_factory=lambda: np.nan)
    center: tuple[int] = field(default_factory=lambda: (0, 0))
    is_fiducial: bool = field(default_factory=lambda: False)
    lines: dict[int, LineInfo] = field(default_factory=lambda: {})  # Only for cross-hairs
    mtf_rois: ROIInfo = field(default_factory=ROIInfo)
    shape: str = field(default_factory=lambda: "NONE")
    size_wh: tuple[int] = field(default_factory=lambda: (0, 0))  # Alt: (radius, radius)


@dataclass
class DetectionInfo:
    crop_size_wh: tuple[int] = field(default_factory=lambda: (0, 0))
    debug_image: np.ndarray = field(default_factory=lambda: np.array([]))
    pois: dict[int, POIInfo] = field(default_factory=lambda: {})
    settings: DefaultSettings = field(default_factory=DefaultSettings)


@dataclass
class ImageInfo:
    date: str | None = field(default_factory=lambda: None)
    directory: str = field(default_factory=lambda: "")
    file_name: str = field(default_factory=lambda: "")


@dataclass
class SupportedArrays:
    color8: np.ndarray = field(default_factory=lambda: np.array([]))
    mono16: np.ndarray = field(default_factory=lambda: np.array([]))
    mono8: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ProcessedInfo:
    arrays: SupportedArrays = field(default_factory=SupportedArrays)
    info: ImageInfo = field(default_factory=ImageInfo)
    detections: DetectionInfo | None = field(default_factory=lambda: None)
