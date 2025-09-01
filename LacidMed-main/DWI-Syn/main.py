# This code synthesizes diffusion images (built for 1.5T, should also work for 3T; NOT for 0.35T).
# Verify inputs each time you call a function. Ensure the `save` function is correctly configured.

import warnings
warnings.filterwarnings("ignore")  # avoid division-by-zero warnings, etc.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nibabel as nib
import pydicom
import re

from lmfit import Parameters, Minimizer

# --- PROJECT IMPORTS ---
from sorted_files import sort
from read_dcm import leer_dcm
from ROI import ROI
from b_values import b_values
from create_maps import create_maps
from generate_DWI import generate_maps
from plot import plot_DWI
from correlation import calculate_pearson
from save import save
from graph import biexponential_fit_and_plot

# Base directory of this script (use Path; no hard-coded absolute paths)
BASE_DIR = Path(__file__).resolve().parent


# =========================
# Helpers: Histogram Shifting (in-place)
# =========================
def auto_detect_min_real(image: np.ndarray, min_count: int = 20) -> int:
    """
    Detect the lowest pixel value with significant presence in the histogram.
    """
    # Use bins from 0..max inclusive
    hist, _ = np.histogram(image, bins=np.arange(0, image.max() + 2))
    for i, count in enumerate(hist):
        if count >= min_count:
            return int(i)
    return 0  # fallback


def shift_histogram(image: np.ndarray, min_real_value: int) -> np.ndarray:
    shifted = image.astype(np.int32) - int(min_real_value)
    shifted = np.clip(shifted, 0, None)
    return shifted.astype(np.uint16)


def process_all_dicoms_inplace(dicom_dir: Path, min_count: int = 20) -> None:
    """
    Shift histogram in-place for all .dcm in dicom_dir. No new folder is created.
    """
    if not dicom_dir.exists():
        raise FileNotFoundError(f"Directory not found: {dicom_dir}")

    dcm_paths = sorted([p for p in dicom_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"])
    if not dcm_paths:
        print(f"No DICOM files found in: {dicom_dir}")
        return

    for path in dcm_paths:
        ds = pydicom.dcmread(str(path))
        original = ds.pixel_array
        min_real = auto_detect_min_real(original, min_count=min_count)
        shifted = shift_histogram(original, min_real)
        new_max = int(shifted.max())

        # Replace pixel data & basic bit depth tags
        ds.PixelData = shifted.tobytes()
        ds.BitsStored = 16
        ds.HighBit = 15

        # Update Windowing for visualization
        ds.WindowCenter = int(new_max / 2)
        ds.WindowWidth = int(new_max)

        # Save back to the same file (in-place)
        ds.save_as(str(path))
        print(f"Shifted (min={min_real}): {path.name}")


# =========================
# Helpers: DICOM Viewing + NIfTI export
# =========================
def _natural_key(name: str):
    """Natural sort key: splits digits and text to sort 1,2,10 correctly."""
    parts = re.split(r"(\d+)", name)
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def apply_dicom_windowing(pixel_array: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """Apply DICOM windowing using metadata."""
    # Rescale if present
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    # Handle WindowCenter/Width possibly being sequences
    def _first_val(v):
        try:
            return float(v[0])
        except Exception:
            return float(v)

    if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
        center = _first_val(ds.WindowCenter)
        width = _first_val(ds.WindowWidth)
        vmin = center - width / 2
        vmax = center + width / 2
    else:
        vmin, vmax = np.percentile(pixel_array, [0.5, 99.5])

    scaled = np.clip(pixel_array, vmin, vmax)
    scaled = (scaled - vmin) / (vmax - vmin + 1e-12)
    return scaled


def view_dicoms_with_slider_and_save_nifti(dicom_dir: Path, nifti_out: Path) -> None:
    """
    Interactive viewer with a slice slider; after the window is closed, exports a NIfTI volume.
    """
    dcm_files = sorted(
        [p for p in dicom_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"],
        key=lambda p: _natural_key(p.name),
    )
    if not dcm_files:
        print(f"No .dcm files to display in: {dicom_dir}")
        return

    dicoms = [pydicom.dcmread(str(p)) for p in dcm_files]
    file_names = [p.name for p in dcm_files]

    scaled_images = [apply_dicom_windowing(ds.pixel_array, ds) for ds in dicoms]

    # Plot with slider
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    plt.axis('off')
    current_slice = ax.imshow(scaled_images[0], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"Slice 1: {file_names[0]} (Windowed)")

    slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
    slice_slider = Slider(slider_ax, 'Slice', 1, len(scaled_images), valinit=1, valstep=1)

    def update(val):
        idx = int(slice_slider.val) - 1
        current_slice.set_data(scaled_images[idx])
        ax.set_title(f"Slice {idx + 1}: {file_names[idx]} (Windowed)")
        fig.canvas.draw_idle()

    slice_slider.on_changed(update)

    def on_key(event):
        current_val = slice_slider.val
        if event.key == 'right':
            slice_slider.set_val(min(current_val + 1, len(scaled_images)))
        elif event.key == 'left':
            slice_slider.set_val(max(current_val - 1, 1))

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()  # Close window to proceed.

    # Rotate each image 90° clockwise and stack into a 3D volume (H x W x Slices)
    rotated_images = [np.rot90(img, k=1) for img in scaled_images]
    volume_3d = np.stack(rotated_images, axis=-1)

    # Identity affine (customize if spatial info is needed)
    affine = np.eye(4)
    nifti_out.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume_3d.astype(np.float32), affine), str(nifti_out))
    print(f"NIfTI file saved at: {nifti_out}")


# =========================
# MAIN
# =========================
def main():
    # 1) INPUT DICOMs (Original, registered stack)
    input_path = BASE_DIR / "Ground_truth" / "patient1" / "Original"
    directory_files = sort(input_path)

    # 2) Read DICOMs
    vauxx, vaux = leer_dcm(directory_files)

    # 3) ROI selection & b-values
    num_b = len(vaux)
    filas, columnas, fil_min, fil_max, col_min, col_max = ROI(vaux, vauxx, num_b)
    orden = b_values(vauxx, directory_files, num_b)

    # 4) Bi-exponential model for LMFIT
    def mob(params, x, data=None):
        S_o = params['S_o']
        f = params['f']
        D_m = params['D']
        D_star = params['D_star']
        model = np.nan_to_num(
            np.log(S_o * (f * np.exp(-x * D_star) + (1 - f) * np.exp(-x * D_m))),
            nan=0, posinf=0, neginf=0
        )
        return model if data is None else (model - data)

    pars = Parameters()
    pars.add('S_o', value=504)
    pars.add('D', value=1e-2, min=0, max=0.1)
    pars.add('D_star', value=1e-5, min=0, max=4)
    pars.add('f', value=0.01, min=0, max=1)

    # 5) Limits for b-values (adjust according to your setup!)
    limite_inferior = int(np.where(orden == 0.25)[0][0])    # corresponds to b = 0
    limite_superior = int(np.where(orden == 18)[0][0]) + 1  # corresponds to b max; +1 for indexing
    b_value = orden[limite_inferior:limite_superior]

    # 6) Create empty maps
    D_map = np.zeros((filas, columnas))
    S0_map = np.zeros((filas, columnas))
    f_map = np.zeros((filas, columnas))
    D_star_map = np.zeros((filas, columnas))

    b_maps, new_b_maps, b_final, new_b_final, truth_maps, truth_final, truth_complete = create_maps(
        filas, columnas, vaux
    )

    # 7) Fit per-pixel inside ROI
    Senal = []
    for i in range(filas):
        for j in range(columnas):
            if (fil_min <= i <= fil_max) and (col_min <= j <= col_max):
                if vaux[limite_inferior][i, j] != 0:
                    for k in range(limite_superior - limite_inferior):
                        Senal.append(vaux[limite_inferior + k][i, j])
                    Senal_log = np.array(Senal)
                    Senal_log = np.nan_to_num(np.log(Senal_log), nan=0, posinf=0, neginf=0)

                    fitter = Minimizer(mob, pars, fcn_args=(b_value, Senal_log))
                    result = fitter.minimize()

                    # Copy ROI region into truth_maps
                    for idx in range(min(8, len(truth_maps))):
                        truth_maps[idx][i, j] = truth_complete[idx][i, j]

                    r_squared = 1 - (result.residual.var() / np.var(Senal_log))
                    if 0 < r_squared <= 1:
                        S0_map[i, j] = result.params['S_o'].value
                        f_map[i, j] = result.params['f'].value
                        D_star_map[i, j] = result.params['D_star'].value
                        D_map[i, j] = result.params['D'].value
                    else:
                        S0_map[i, j] = vaux[0][i, j]
                    Senal = []  # reset buffer

    # 8) Generate synthetic maps (original b's and new b's)
    b_final, new_b_final, truth_final = generate_maps(
        b_maps, new_b_maps, b_final, new_b_final,
        truth_maps, truth_final, truth_complete,
        S0_map, f_map, D_star_map, D_map, 2048
    )

    # 9) Optional analysis/visualization in-plane
    # plot_DWI(vaux1, vaux2, truth_final, fil_min, fil_max, col_min, col_max)  # if desired later

    # 10) Inspect one random fitted pixel curve
    r_f = int(np.random.randint(fil_max - fil_min))
    r_c = int(np.random.randint(col_max - col_min))
    biexponential_fit_and_plot(
        r_f, r_c, limite_superior, limite_inferior, fil_min, col_min, vaux, b_value, mob, pars
    )

    # 11) Pearson correlation between original & synthetic images (prints results)
    calculate_pearson(b_final, truth_final, fil_min, fil_max, col_min, col_max)

    # 12) Save synthetic DICOMs (your `save` should write to: Ground_truth/patient1/Synthetic)
    save(b_final, new_b_final)


    # 13) Histogram shift IN-PLACE over Synthetic folder (no "Shifted" folder)
    synthetic_dir = BASE_DIR / "Ground_truth" / "patient1" / "Synthetic"
    process_all_dicoms_inplace(synthetic_dir, min_count=20)

    # 14) View DICOMs with slider and export a NIfTI volume
    nifti_out = BASE_DIR / "Ground_truth" / "patient1" / "DWI_Syn_sequence.nii.gz"
    view_dicoms_with_slider_and_save_nifti(synthetic_dir, nifti_out)


if __name__ == '__main__':
    main()
    author__ = 'Pablo Irusta <irusta.pablo.b@gmail.com>'
    __version__ = "2.0.0"
