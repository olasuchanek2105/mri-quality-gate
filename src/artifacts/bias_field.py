# intensity gradient

from __future__ import annotations

from typing import Optional, Tuple, Dict, Literal
import numpy as np
from scipy.ndimage import gaussian_filter

Quality = Literal["ok", "borderline", "reject"]

#amplituda nie sigma
def sample_amp_for_quality(quality: Quality, rng: np.random.Generator) -> float:
    """
    amp controls how strong the multiplicative field is.
    Field will be roughly in [1-amp, 1+amp] after normalization.
    """
    if quality == "ok":
        return float(rng.uniform(0.05, 0.10))
    if quality == "borderline":
        return float(rng.uniform(0.15, 0.25))
    if quality == "reject":
        return float(rng.uniform(0.30, 0.50))
    raise ValueError("quality must be one of: ok, borderline, reject")

def make_bias_field(shape: tuple[int, int], rng: np.random.Generator, *, smooth_sigma: float, amp: float) -> np.ndarray:
    base = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    smooth = gaussian_filter(base, sigma=smooth_sigma)

    # normalize to mean=1
    smooth = smooth - float(np.mean(smooth))
    smooth = smooth / (float(np.std(smooth)) + 1e-8)

    field = 1.0 + amp * smooth

    # optional: clamp extreme values a bit (stability)
    field = np.clip(field, 1.0 - 2.5*amp, 1.0 + 2.5*amp)
    return field.astype(np.float32)


def sample_smooth_sigma(shape: tuple[int, int], rng: np.random.Generator) -> float:
    h, w = shape
    base = 0.15 * min(h, w)
    # trochę losowości
    return float(rng.uniform(0.8 * base, 1.2 * base))


def apply_random_bias_field(
    img: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Multiplicative low-frequency intensity inhomogeneity (bias field).
    img: 2D float array in [0,1]
    Returns (out_img, meta_dict)
    """
    rng = rng or np.random.default_rng()

    quality: Quality = rng.choice(["ok", "borderline", "reject"], p=[0.5, 0.3, 0.2])
    amp = sample_amp_for_quality(quality, rng)

    smooth_sigma = sample_smooth_sigma(img.shape, rng)
    field = make_bias_field(img.shape, rng, smooth_sigma=smooth_sigma, amp=amp)

    out = img * field
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    meta = {
        "artifact_type": "bias_field",
        "amp": float(amp),
        "smooth_sigma": float(smooth_sigma),
        "quality": quality,
    }
    return out, meta