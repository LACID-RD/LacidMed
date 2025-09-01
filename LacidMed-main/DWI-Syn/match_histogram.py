#CAMBIAR EL DIRECTORIO

import os
import numpy as np
import pydicom
from pathlib import Path


def auto_detect_min_real(image, min_count=20):
    """
    Detect the lowest pixel value with significant presence in the histogram.
    """
    hist, bin_edges = np.histogram(image, bins=np.arange(0, image.max() + 2))
    for i, count in enumerate(hist):
        if count >= min_count:
            return i
    return 0  # fallback

def shift_histogram(image, min_real_value):
    shifted = image.astype(np.int32) - min_real_value
    shifted = np.clip(shifted, 0, None)
    return shifted.astype(np.uint16)

def process_all_dicoms(input_dir, output_dir, min_count=20):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".dcm"):
            path = os.path.join(input_dir, filename)
            ds = pydicom.dcmread(path)

            original = ds.pixel_array
            min_real = auto_detect_min_real(original, min_count=min_count)
            shifted = shift_histogram(original, min_real)
            new_max = shifted.max()

            # Replace pixel data
            ds.PixelData = shifted.tobytes()
            ds.BitsStored = 16
            ds.HighBit = 15

            # Update Window Center and Width for visualization
            ds.WindowCenter = int(new_max / 2)
            ds.WindowWidth = int(new_max)

            # Save new file
            out_path = os.path.join(output_dir, filename)
            ds.save_as(out_path)
            print(f"Processed: {filename} (min shifted: {min_real})")

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent
input_dir = BASE_DIR / "Ground_truth" / "patient1" / "Synthetic"
output_dir = BASE_DIR / "Ground_truth" / "patient1" / "Shifted"


process_all_dicoms(input_dir, output_dir, min_count=20)
