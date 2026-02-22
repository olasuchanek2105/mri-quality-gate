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
from src.artifacts.motion_fft import apply_random_motion_fft

def collect_analyze_img_paths(data_root: str | Path) -> List[Path]:
    data_root = Path(data_root)
    imgs: List[Path] = []
    for disc in sorted(data_root.glob("disc*")):
        for subj in sorted(disc.glob("OAS1_*_MR1")):
            raw_dir = subj / "RAW"
            if raw_dir.exists():
                imgs.extend(sorted(raw_dir.rglob("*.img")))
    return imgs

# def make_one_example(dcm_path: Path, rng=None):
#     rng = rng or np.random.default_rng()
#     img = load_analyze_slice_float01(dcm_path)  
    
#     artifact = "motion_fft"  # albo "noise"

#     if artifact == "noise":
#         noisy_img, meta = apply_random_gaussian_noise(img, rng=rng)
#     elif artifact == "blur":
#         noisy_img, meta = apply_random_blur(img, rng=rng)
#     elif artifact =="bias_field": 
#         noisy_img, meta = apply_random_bias_field(img, rng=rng)
#     elif artifact == "motion_fft":
#         noisy_img, meta = apply_random_motion_fft(img, rng=rng, axis=0)

#     return img, noisy_img, meta

def make_one_example(dcm_path: Path, rng=None):
    rng = rng or np.random.default_rng()

    img = load_analyze_slice_float01(dcm_path)
    x = img.copy()

    # ---- 1. ile artefaktów? ----
    n_artifacts = rng.choice([1, 2, 3], p=[0.75, 0.20, 0.05])

    # ---- 2. jakie artefakty? ----
    artifact_types = ["noise", "blur", "bias_field", "motion_fft"]
    artifact_probs = [0.35, 0.20, 0.25, 0.20]

    chosen = rng.choice(
        artifact_types,
        size=n_artifacts,
        replace=False,
        p=artifact_probs,
    )

    metas = []

    # ---- 3. aplikacja w realistycznej kolejności ----
    order = ["bias_field", "motion_fft", "blur", "noise"]

    for art in order:
        if art not in chosen:
            continue

        if art == "noise":
            x, meta = apply_random_gaussian_noise(x, rng=rng)

        elif art == "blur":
            x, meta = apply_random_blur(x, rng=rng)

        elif art == "bias_field":
            x, meta = apply_random_bias_field(x, rng=rng)

        elif art == "motion_fft":
            x, meta = apply_random_motion_fft(x, rng=rng, axis=0)

        metas.append(meta)

    # ---- 4. final quality = najgorsza ----
    quality_order = {"ok": 0, "borderline": 1, "reject": 2}
    worst_quality = max(metas, key=lambda m: quality_order[m["quality"]])["quality"]

    final_meta = {
        "artifact_chain": "+".join([m["artifact_type"] for m in metas]),
        "quality": worst_quality,
        "n_artifacts": int(n_artifacts),
    }

    return img, x, final_meta


#tymczasowo sprawdzenie

if __name__ == "__main__":
    

    paths = collect_analyze_img_paths("data")
    print(f"Found {len(paths)} ANALYZE .img files")


    rng = np.random.default_rng()

for i in range(10):
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

    plt.show()
    ratio = noisy / (img + 1e-8)
    plt.imshow(ratio, cmap="gray")
    plt.title("bias field")
    plt.colorbar()
    plt.show()

    # shifted = np.roll(img, shift=10, axis=0)
    # ghost = 0.7 * img + 0.3 * shifted
    # plt.imshow(ghost, cmap="gray", vmin=0, vmax=1)
    # plt.title("shifted")
    # plt.show()

    

