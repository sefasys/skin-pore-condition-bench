import random

import numpy as np
import pytest
import torch
from PIL import Image

from pore_assessment.metrics import CLASS_NAMES
from pore_assessment.preprocessing import (
    PREPROCESSING_METHODS,
    DeterministicPreprocessing,
)
from pore_assessment.train import (
    MODEL_ARCHITECTURES,
    RandomRightAngleRotation,
    build_model,
    classifier_parameter_ids,
)


def test_right_angle_rotation_preserves_pixels_and_dimensions() -> None:
    image = Image.new("RGB", (12, 12), (10, 20, 30))
    image.putpixel((0, 0), (255, 0, 0))
    original_pixels = sorted(image.getdata())
    transform = RandomRightAngleRotation()

    random.seed(123)
    for _ in range(12):
        transformed = transform(image)
        assert transformed.size == image.size
        assert sorted(transformed.getdata()) == original_pixels


def _nonuniform_image() -> Image.Image:
    y, x = np.mgrid[0:64, 0:64]
    red = 60 + x * 2
    green = 45 + y * 2
    blue = 35 + ((x + y) % 24)
    array = np.stack((red, green, blue), axis=-1).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


@pytest.mark.parametrize("method", PREPROCESSING_METHODS)
def test_preprocessing_is_deterministic_rgb_and_bounded(method: str) -> None:
    image = _nonuniform_image()
    transform = DeterministicPreprocessing(method)

    first = transform(image)
    second = transform(image)

    assert first.mode == "RGB"
    assert first.size == image.size
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert np.asarray(first).dtype == np.uint8


def test_raw_preprocessing_preserves_pixels() -> None:
    image = _nonuniform_image()
    transformed = DeterministicPreprocessing("raw")(image)
    assert transformed is not image
    assert np.array_equal(np.asarray(transformed), np.asarray(image))


@pytest.mark.parametrize("method", ("clahe", "retinex", "adaptive_gamma"))
def test_enhancement_changes_nonuniform_image(method: str) -> None:
    image = _nonuniform_image()
    transformed = DeterministicPreprocessing(method)(image)
    assert not np.array_equal(np.asarray(transformed), np.asarray(image))


def test_unknown_preprocessing_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown preprocessing"):
        DeterministicPreprocessing("not-a-method")


@pytest.mark.parametrize("architecture", MODEL_ARCHITECTURES)
def test_supported_architecture_has_five_class_output(architecture: str) -> None:
    model = build_model(architecture=architecture, pretrained=False)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, len(CLASS_NAMES))


@pytest.mark.parametrize("architecture", MODEL_ARCHITECTURES)
def test_classifier_parameters_are_nonempty_subset(architecture: str) -> None:
    model = build_model(architecture=architecture, pretrained=False)
    all_ids = {id(parameter) for parameter in model.parameters()}
    head_ids = classifier_parameter_ids(model, architecture)
    assert head_ids
    assert head_ids < all_ids
