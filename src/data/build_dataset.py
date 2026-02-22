from typing import Optional, Tuple, Dict, List
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from src.artifacts.noise import add_gaussian_noise, NoiseParams
from src.dicom.io import load_dicom_as_float01
from src.dicom.io import load_analyze_slice_float01
from src.artifacts.blur import apply_random_blur
from src.artifacts.noise import apply_random_gaussian_noise
from src.artifacts.bias_field import apply_random_bias_field

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
    img = load_analyze_slice_float01(dcm_path)  
    
    artifact = "bias_field"  # albo "noise"

    if artifact == "noise":
        noisy_img, meta = apply_random_gaussian_noise(img, rng=rng)
    elif artifact == "blur":
        noisy_img, meta = apply_random_blur(img, rng=rng)
    elif artifact =="bias_field": 
        noisy_img, meta = apply_random_bias_field(img, rng=rng)

    return img, noisy_img, meta




#tymczasowo sprawdzenie

if __name__ == "__main__":
    

    paths = collect_analyze_img_paths("data")
    print(f"Found {len(paths)} ANALYZE .img files")


    rng = np.random.default_rng(42)

for i in range(3):
    dcm_path = paths[i]

    img, noisy, meta = make_one_example(dcm_path, rng=rng)

    print(meta)

    diff = np.abs(noisy - img)

    fig, axs = plt.subplots(1, 3, figsize=(12,4))

    axs[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("clean")

    axs[1].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("noisy")

    axs[2].imshow(diff, cmap="gray")
    axs[2].set_title("abs diff")

    for ax in axs:
        ax.axis("off")

    # plt.show()
    ratio = noisy / (img + 1e-8)
    plt.imshow(ratio, cmap="gray")
    plt.title("bias field")
    plt.colorbar()
    plt.show()
