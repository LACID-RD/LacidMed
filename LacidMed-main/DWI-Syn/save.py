#THIS CODE, used to save generated images, requires you to adjust:
# - the directories,
# - the generated b-values,
# - and the output filenames.
#Images to read must be in "actual", and generated images will be written in "generated".

from pathlib import Path
import numpy as np
import pydicom

# Base directory of this script (no hard-coded absolute paths)
BASE_DIR = Path(__file__).resolve().parent

def adjust_contrast(pixel_array, min_val=0, max_val=None):
    """
    Linearly stretch the contrast of the pixel_array to fit within [min_val, max_val],
    preserving the actual dynamic range for correct display in DICOM viewers.
    """
    if max_val is None:
        raise ValueError("You must provide max_val from the original image.")
    
    input_min = pixel_array.min()
    input_max = pixel_array.max()

    if input_max == input_min:
        # Avoid divide-by-zero
        return np.full_like(pixel_array, fill_value=min_val, dtype=np.uint16)

    # Scale to the desired range while preserving actual range
    scaled = (pixel_array - input_min) / (input_max - input_min)
    adjusted = scaled * (max_val - min_val) + min_val

    return np.clip(adjusted, min_val, max_val).astype(np.uint16)

def save(pixel_arrays, new_b_final, window_center=2048, window_width=4096):
    """
    Save synthetic and interpolated DWI maps in DICOM format using metadata 
    from reference DICOM files. Automatically adjusts contrast to match original images.
    """

    # Paths (relative to this script)
    input_path = BASE_DIR / "Ground_truth" / "patient1" / "Original"
    output_path = BASE_DIR / "Ground_truth" / "patient1" / "Synthetic"

    output_path.mkdir(parents=True, exist_ok=True)

    # Original b-values and file structure
    b_values_list = [25, 50, 100, 200, 500, 1000, 1500, 1800]
    input_files = [input_path / f"ground_truth_{b}_registered.dcm" for b in b_values_list]
    output_files = [output_path / f"{b}_generated.dcm" for b in b_values_list]

    # Save synthetic maps for original b-values
    for i, pixel_array in enumerate(pixel_arrays):
        ds = pydicom.dcmread(str(input_files[i]))
        original_pixel_array = ds.pixel_array
        max_val = int(original_pixel_array.max())

        pixel_array = adjust_contrast(pixel_array, max_val=max_val)
        ds.PixelData = np.array(pixel_array, dtype=np.uint16).tobytes()
        ds.WindowCenter = window_center
        ds.WindowWidth = window_width
        ds.SeriesDescription = f"Synthetic DWI b={b_values_list[i]}"
        ds.save_as(str(output_files[i]))

    # Save synthetic maps for new interpolated b-values
    new_b_values = [75, 125, 175, 350, 800, 1200, 1600]
    ref_ds = pydicom.dcmread(str(input_path / "ground_truth_25_registered.dcm"))
    ref_max_val = int(ref_ds.pixel_array.max())
    new_output_files = [output_path / f"{b}_generated.dcm" for b in new_b_values]

    for i, pixel_array in enumerate(new_b_final):
        pixel_array = adjust_contrast(pixel_array, max_val=ref_max_val)
        ds = ref_ds.copy()
        ds.PixelData = np.array(pixel_array, dtype=np.uint16).tobytes()
        ds.WindowCenter = window_center
        ds.WindowWidth = window_width
        ds.SeriesDescription = f"Interpolated DWI b={new_b_values[i]}"
        ds.save_as(str(new_output_files[i]))

    print(f"Saved {len(pixel_arrays) + len(new_b_final)} synthetic DICOM files to: {output_path}")
