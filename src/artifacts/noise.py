# gaussian noise
# rician-like noise (MRI-ish)

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import numpy as np

NoiseType = Literal["gaussian", "rician"]


@dataclass(frozen=True) #frozen=True - nie można zmienić po stworzeniu

class NoiseParams:
    noise_type: NoiseType
    sigma: float
    clip: bool

def _validate_image(img: np.ndarray) -> np.ndarray:
    """Ensure img is 2D float array."""
    if not isinstance(img, np.ndarray):
        raise TypeError("img must be numpy.ndarray")
    if img.ndim != 2:
        raise ValueError(f"img must be 2D (H,W). Got shape={img.shape}")
    if img.dtype not in (np.float32, np.float64): # konwertujemy jeśli trzeba
        img = img.astype(np.float32, copy=False)

    return img

    

#clip przycina wartości do zadanego zakresu
def _maybe_clip(img: np.ndarray, clip: bool) -> np.ndarray:
    return np.clip(img, 0.0, 1.0) if clip else img



def add_gaussian_noise(
        img: np.ndarray,
        sigma: float, #intensywnosc szumu
        *,
        clip: bool = True,
        rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, NoiseParams]:
    """
    Add additive Gaussian noise to a 2D image.

    Model: y = x + n, where n ~ N(0, sigma^2)

    Parameters
    ----------
    img : np.ndarray
        2D image (H, W), expected to be float and typically normalized to [0, 1].
    sigma : float
        Standard deviation of the noise (in the same scale as img).
        Example: sigma=0.05 means fairly visible noise if img is in [0,1].
    clip : bool
        If True, clip output to [0, 1] after adding noise.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    noisy_img : np.ndarray
        Image with added Gaussian noise.
    params : NoiseParams
        Parameters describing the applied noise.
    """
    img = _validate_image(img)

    if sigma < 0 or sigma > 0.3:
        raise ValueError("sigma must be in range [0, 0.3] for normalized images")

    
    rng = rng or np.random.default_rng()

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=img.shape 
    ).astype(img.dtype, copy=False)

    noisy_img = img + noise
    noisy_img = _maybe_clip(noisy_img, clip)

    params = NoiseParams(
        noise_type="gaussian", 
        sigma=float(sigma), 
        clip=bool(clip)
        )
    
    return noisy_img, params


def sample_sigma_for_quality(
    quality: str,
    rng: np.random.Generator,
) -> float:
    """
    Sample sigma based on a desired quality class.
    Assumes images are normalized to [0, 1].
    """
    if quality == "ok":
        return float(rng.uniform(0.0, 0.01))
    if quality == "borderline":
        return float(rng.uniform(0.02, 0.05))
    if quality == "reject":
        return float(rng.uniform(0.06, 0.12))
    raise ValueError("quality must be one of: ok, borderline, reject")


def apply_random_gaussian_noise(
    img: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply Gaussian noise with a randomly chosen quality class.
    Returns (noisy_img, metadata_dict).
    """
    rng = rng or np.random.default_rng()

    # 1) losujemy klasę jakości (możesz zmienić proporcje)
    quality = rng.choice(["ok", "borderline", "reject"], p=[0.5, 0.3, 0.2])

    # 2) losujemy sigma zależnie od klasy
    sigma = sample_sigma_for_quality(quality, rng)

    # 3) aplikujemy noise
    noisy_img, params = add_gaussian_noise(img, sigma, rng=rng, clip=True)

    # 4) metadane do labels.csv
    meta = {
        "artifact_type": "noise",
        "noise_type": params.noise_type,
        "sigma": params.sigma,
        "quality": quality,
    }
    return noisy_img, meta