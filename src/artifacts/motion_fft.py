from __future__ import annotations

from typing import Optional, Tuple, Dict, Literal
import numpy as np

Quality = Literal["ok", "borderline", "reject"]


def sample_motion_params_for_quality(
    quality: Quality,
    rng: np.random.Generator,
    shape: tuple[int, int],
) -> Tuple[int, int, float]:
    """
    Zwraca (shift_px, every_n_lines, mix).
    - shift_px: o ile pikseli przesuwamy obraz referencyjny (źródło "drugiej pozycji")
    - every_n_lines: co ile linii k-space podmieniamy (mniejsze -> mocniejszy artefakt)
    - mix: ile tej podmiany faktycznie stosujemy (0..1), 1 = pełna podmiana
    """
    h, w = shape
    max_shift = max(4, int(0.20 * min(h, w)))  # do 20% krótszego boku

    if quality == "ok":
        shift = int(rng.integers(2, max(4, max_shift // 4 + 1)))
        every = int(rng.choice([12, 16, 20]))
        mix = float(rng.uniform(0.3, 0.6))
        return shift, every, mix

    if quality == "borderline":
        shift = int(rng.integers(6, max_shift // 3))
        every = int(rng.choice([10, 12, 14]))
        mix = float(rng.uniform(0.4, 0.65))
        return shift, every, mix

    if quality == "reject":
        shift = int(rng.integers(max(8, max_shift // 2), max_shift + 1))
        every = int(rng.choice([2, 3, 4, 5]))
        mix = float(rng.uniform(0.85, 1.0))
        return shift, every, mix

    raise ValueError("quality must be one of: ok, borderline, reject")


def apply_random_motion_fft(
    img: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    axis: int = 0,
) -> Tuple[np.ndarray, Dict]:
    """
    Pseudo motion MRI w k-space:
    - liczymy FFT obrazu (k-space)
    - liczymy FFT obrazu przesuniętego (symulacja ruchu)
    - co N-tą linię w k-space podmieniamy danymi z przesuniętego
    """
    rng = rng or np.random.default_rng()

    quality: Quality = rng.choice(["ok", "borderline", "reject"], p=[0.5, 0.3, 0.2])
    shift_px, every_n, mix = sample_motion_params_for_quality(quality, rng, img.shape)

    # "druga pozycja" pacjenta
    shifted = np.roll(img, shift=shift_px, axis=axis)

    # k-space
    k = np.fft.fft2(img)
    k_shift = np.fft.fft2(shifted)

    k_motion = k.copy()

    if axis == 0:
        # podmieniamy wybrane wiersze
        idx = np.arange(k_motion.shape[0])
        mask = (idx % every_n) == 0
        # mieszanie: zamiast brutalnej podmiany robimy blend
        k_motion[mask, :] = (1.0 - mix) * k_motion[mask, :] + mix * k_shift[mask, :]
    else:
        # podmieniamy wybrane kolumny
        idx = np.arange(k_motion.shape[1])
        mask = (idx % every_n) == 0
        k_motion[:, mask] = (1.0 - mix) * k_motion[:, mask] + mix * k_shift[:, mask]

    out = np.fft.ifft2(k_motion).real
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    meta = {
        "artifact_type": "motion_fft",
        "quality": quality,
        "shift_px": int(shift_px),
        "every_n_lines": int(every_n),
        "mix": float(mix),
        "axis": int(axis),
    }
    return out, meta