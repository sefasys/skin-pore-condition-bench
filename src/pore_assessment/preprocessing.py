"""Deterministic image preprocessing used in the controlled comparison."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image
from skimage import color, exposure
from skimage.filters import gaussian

PREPROCESSING_METHODS = ("raw", "clahe", "retinex", "adaptive_gamma")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Frozen parameters for every deterministic input method."""

    clahe_clip_limit: float = 0.01
    clahe_kernel_size: int = 28
    retinex_sigmas: tuple[float, ...] = (15.0, 60.0, 180.0)
    retinex_low_percentile: float = 1.0
    retinex_high_percentile: float = 99.0
    gamma_weight: float = 0.5

    def to_dict(self) -> dict:
        result = asdict(self)
        result["retinex_sigmas"] = list(self.retinex_sigmas)
        return result


DEFAULT_CONFIG = PreprocessingConfig()


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _rgb_image(array: np.ndarray) -> Image.Image:
    pixels = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _replace_lab_luminance(rgb: np.ndarray, luminance: np.ndarray) -> Image.Image:
    lab = color.rgb2lab(rgb)
    lab[..., 0] = np.clip(luminance, 0.0, 1.0) * 100.0
    # Large luminance changes can push a small number of LAB pixels outside the
    # sRGB gamut. skimage clips them correctly but otherwise emits one warning
    # per image, which would overwhelm multi-epoch training logs.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Conversion from CIE-LAB.*",
            category=UserWarning,
        )
        converted = color.lab2rgb(lab)
    return _rgb_image(converted)


def luminance_clahe(
    image: Image.Image,
    config: PreprocessingConfig = DEFAULT_CONFIG,
) -> Image.Image:
    """Apply CLAHE to CIELAB luminance without independently altering RGB."""

    rgb = _rgb_array(image)
    luminance = color.rgb2lab(rgb)[..., 0] / 100.0
    enhanced = exposure.equalize_adapthist(
        luminance,
        kernel_size=(config.clahe_kernel_size, config.clahe_kernel_size),
        clip_limit=config.clahe_clip_limit,
        nbins=256,
    )
    return _replace_lab_luminance(rgb, enhanced)


def multiscale_retinex(
    image: Image.Image,
    config: PreprocessingConfig = DEFAULT_CONFIG,
) -> Image.Image:
    """Apply fixed-scale single-channel MSR and retain the original chroma."""

    rgb = _rgb_array(image)
    luminance = color.rgb2lab(rgb)[..., 0] / 100.0
    stabilized = np.clip(luminance, 0.0, 1.0) + (1.0 / 255.0)
    responses = []
    for sigma in config.retinex_sigmas:
        illumination = gaussian(
            stabilized,
            sigma=sigma,
            mode="reflect",
            preserve_range=True,
        )
        responses.append(np.log(stabilized) - np.log(illumination + 1e-6))
    response = np.mean(responses, axis=0)

    low, high = np.percentile(
        response,
        (config.retinex_low_percentile, config.retinex_high_percentile),
    )
    if high <= low + 1e-8:
        enhanced = luminance
    else:
        enhanced = np.clip((response - low) / (high - low), 0.0, 1.0)
    return _replace_lab_luminance(rgb, enhanced)


def adaptive_gamma_correction(
    image: Image.Image,
    config: PreprocessingConfig = DEFAULT_CONFIG,
) -> Image.Image:
    """Apply adaptive gamma correction with a weighted luminance CDF."""

    rgb = _rgb_array(image)
    luminance = color.rgb2lab(rgb)[..., 0] / 100.0
    levels = np.clip(np.rint(luminance * 255.0), 0, 255).astype(np.uint8)
    histogram = np.bincount(levels.ravel(), minlength=256).astype(np.float64)
    probability = histogram / max(histogram.sum(), 1.0)
    minimum = probability.min()
    maximum = probability.max()

    if maximum <= minimum:
        enhanced = luminance
    else:
        weighted = maximum * np.power(
            np.clip((probability - minimum) / (maximum - minimum), 0.0, 1.0),
            config.gamma_weight,
        )
        weighted /= max(weighted.sum(), 1e-12)
        cumulative = np.cumsum(weighted)
        gamma = 1.0 - cumulative
        normalized_levels = np.arange(256, dtype=np.float64) / 255.0
        lookup = np.power(normalized_levels, gamma, where=normalized_levels > 0)
        lookup[0] = 0.0
        enhanced = lookup[levels]
    return _replace_lab_luminance(rgb, enhanced)


class DeterministicPreprocessing:
    """PIL transform selecting one frozen input representation."""

    def __init__(
        self,
        method: str = "raw",
        config: PreprocessingConfig = DEFAULT_CONFIG,
    ) -> None:
        if method not in PREPROCESSING_METHODS:
            raise ValueError(
                f"Unknown preprocessing method {method!r}; "
                f"choose from {PREPROCESSING_METHODS}"
            )
        self.method = method
        self.config = config

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if self.method == "raw":
            return image.copy()
        if self.method == "clahe":
            return luminance_clahe(image, self.config)
        if self.method == "retinex":
            return multiscale_retinex(image, self.config)
        return adaptive_gamma_correction(image, self.config)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(method={self.method!r})"
