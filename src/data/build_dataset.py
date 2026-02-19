from typing import Optional, Tuple, Dict, List
import numpy as np
from pathlib import Path

from src.artifacts.noise import add_gaussian_noise, NoiseParams
from src.dicom.io import load_dicom_as_float01
from src.dicom.io import load_analyze_slice_float01


def collect_analyze_img_paths(data_root: str | Path) -> List[Path]:
    data_root = Path(data_root)
    imgs: List[Path] = []
    for disc in sorted(data_root.glob("disc*")):
        for subj in sorted(disc.glob("OAS1_*_MR1")):
            raw_dir = subj / "RAW"
            if raw_dir.exists():
                imgs.extend(sorted(raw_dir.rglob("*.img")))
    return imgs

def make_one_example(dcm_path: Path, rng=None):
    rng = rng or np.random.default_rng()
    # img = load_dicom_as_float01(dcm_path)
    img = load_analyze_slice_float01(dcm_path)  # tu dcm_path to teraz .img path


    noisy_img, meta = apply_random_gaussian_noise(img, rng=rng)
    return img, noisy_img, meta


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





#tymczasowo

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    paths = collect_analyze_img_paths("data")
    print(f"Found {len(paths)} ANALYZE .img files")


    rng = np.random.default_rng(42)

    for i in range(3):
        dcm_path = paths[i]

        img, noisy, meta = make_one_example(dcm_path, rng=rng)

        print(meta)

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(img, cmap="gray")
        axs[0].set_title("clean")
        axs[1].imshow(noisy, cmap="gray")
        axs[1].set_title("noisy")

        plt.show()
