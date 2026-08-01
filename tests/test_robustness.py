import numpy as np
from PIL import Image

from blood_experiment.robustness import CORRUPTION_NAMES, build_corruption


def test_severity_zero_is_identity() -> None:
    image = Image.new("RGB", (16, 16), color=(120, 80, 40))

    result = build_corruption("contrast", severity=0, seed=1)(image)

    assert np.array_equal(np.asarray(result), np.asarray(image))


def test_noise_corruption_is_repeatable_for_same_seed() -> None:
    image = Image.new("RGB", (16, 16), color=(120, 80, 40))
    corruption = build_corruption("gaussian_noise", severity=2, seed=42)

    assert np.array_equal(np.asarray(corruption(image)), np.asarray(corruption(image)))


def test_noise_corruption_varies_deterministically_between_images() -> None:
    corruption = build_corruption("gaussian_noise", severity=2, seed=42)
    first = Image.new("RGB", (16, 16), color=(120, 80, 40))
    second = Image.new("RGB", (16, 16), color=(121, 80, 40))

    first_noise = np.asarray(corruption(first), dtype=np.int16) - np.asarray(first, dtype=np.int16)
    second_noise = np.asarray(corruption(second), dtype=np.int16) - np.asarray(second, dtype=np.int16)
    assert not np.array_equal(first_noise, second_noise)


def test_registry_covers_stain_and_acquisition_perturbations() -> None:
    assert {"brightness", "white_balance", "gaussian_noise", "jpeg"} <= set(CORRUPTION_NAMES)
