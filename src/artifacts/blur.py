# gaussian blur
# motion blur

from __future__ import annotations
from typing import Optional, Tuple, Dict, Literal
import numpy as np
from scipy.ndimage import gaussian_filter

Quality = Literal["ok", "borderline", "reject"]

def sample_sigma_for_quality(quality: Quality, rng: np.random.Generator) -> float:
    if quality == "ok":
        return float(rng.uniform(0.2, 0.6))
    if quality == "borderline":
        return float(rng.uniform(0.8, 1.6))
    if quality == "reject":
        return float(rng.uniform(1.8, 3.2))
    raise ValueError("quality must be one of: ok, borderline, reject")



def apply_random_blur(
        img: np.ndarray,
        rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply blur artifact to img (2D, float in [0,1]).
    Returns (blurred_img, meta_dict).
    """
    rng = rng or np.random.default_rng()

    quality = rng.choice(["ok", "borderline", "reject"], p=[0.5, 0.3, 0.2])
    sigma = sample_sigma_for_quality(quality, rng)

    out = gaussian_filter(img, sigma = sigma)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    meta = {
        "artifact_type": "blur",
        "blur_type": "gaussian", 
        "sigma": float(sigma),
        "quality": quality
    }
    return out, meta


