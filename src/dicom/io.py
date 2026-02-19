import numpy as np
import pydicom
from pathlib import Path


from pydicom.pixel_data_handlers.util import apply_voi_lut

def load_dicom_as_float01(path: str | Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    img = ds.pixel_array

    # windowing / VOI LUT (jeśli istnieje)
    try:
        img = apply_voi_lut(img, ds)
    except Exception:
        pass

    # float
    img = img.astype(np.float32)

    # robust min-max (percentyle, lepsze niż czyste min/max)
    lo, hi = np.percentile(img, (1, 99))
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6)

    return img



from pathlib import Path
import numpy as np
import nibabel as nib

def load_analyze_slice_float01(path_img: str | Path) -> np.ndarray:
    """
    Load ANALYZE 7.5 .img/.hdr volume and return a single 2D slice normalized to [0,1].
    """
    nii = nib.load(str(path_img))         # nib sam znajdzie odpowiadający .hdr
    vol = nii.get_fdata().astype(np.float32)  # (H, W, D) zwykle

    # OASIS czasem ma shape (H, W, D, 1) – usuń ostatni wymiar
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]

    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={vol.shape}")

    # wybór slice: środkowy (na start)
    z = vol.shape[2] // 2
    img2d = vol[:, :, z]

    # robust normalization do [0,1]
    lo, hi = np.percentile(img2d, (1, 99))
    img2d = np.clip(img2d, lo, hi)
    img2d = (img2d - lo) / (hi - lo + 1e-6)

    return img2d
