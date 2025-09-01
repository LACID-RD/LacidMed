import os
import re
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nibabel as nib
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent
dicom_dir = BASE_DIR / 'Ground_truth' / "patient1" / "Shifted" 

def numericalSort(value):
    """
    Helper function to sort filenames in natural numeric order.
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def apply_dicom_windowing(pixel_array, dicom_file):
    """Apply DICOM windowing using metadata."""
    if hasattr(dicom_file, 'RescaleSlope') and hasattr(dicom_file, 'RescaleIntercept'):
        pixel_array = pixel_array * float(dicom_file.RescaleSlope) + float(dicom_file.RescaleIntercept)
    if hasattr(dicom_file, 'WindowCenter') and hasattr(dicom_file, 'WindowWidth'):
        center = float(dicom_file.WindowCenter)
        width = float(dicom_file.WindowWidth)
        vmin = center - width / 2
        vmax = center + width / 2
    else:
        vmin, vmax = np.percentile(pixel_array, [0.5, 99.5])
    scaled = np.clip(pixel_array, vmin, vmax)
    scaled = (scaled - vmin) / (vmax - vmin)
    return scaled

# Get sorted list of DICOM file paths using natural numeric order
file_paths = sorted(
    [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.lower().endswith('.dcm')],
    key=lambda x: numericalSort(os.path.basename(x))
)

# Load DICOM files and filenames
dicom_files = [pydicom.dcmread(fp) for fp in file_paths]
file_names = [os.path.basename(fp) for fp in file_paths]

# Apply windowing to all slices
scaled_images = [apply_dicom_windowing(ds.pixel_array, ds) for ds in dicom_files]

# Create plot with slider
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
plt.axis('off')
current_slice = ax.imshow(scaled_images[0], cmap='gray', vmin=0, vmax=1)
ax.set_title(f"Slice 1: {file_names[0]} (Windowed)")

# Slider configuration
slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
slice_slider = Slider(slider_ax, 'Slice', 1, len(scaled_images), valinit=1, valstep=1)

def update(val):
    slice_index = int(slice_slider.val) - 1
    current_slice.set_data(scaled_images[slice_index])
    ax.set_title(f"Slice {slice_index + 1}: {file_names[slice_index]} (Windowed)")
    fig.canvas.draw_idle()

slice_slider.on_changed(update)

# Keyboard event handling for slider control
def on_key(event):
    current_val = slice_slider.val
    if event.key == 'right':  # Move slider forward
        new_val = min(current_val + 1, len(scaled_images))
        slice_slider.set_val(new_val)
    elif event.key == 'left':  # Move slider backward
        new_val = max(current_val - 1, 1)
        slice_slider.set_val(new_val)

fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()
# Rotate each image 90 degrees to the right (clockwise)
rotated_images = [np.rot90(img, k=1) for img in scaled_images]

# Stack into a 3D volume (H x W x Slices)
volume_3d = np.stack(rotated_images, axis=-1)

# Create an identity affine (or customize if spatial info is needed)
affine = np.eye(4)

# Create and save the NIfTI file
nifti_img = nib.Nifti1Image(volume_3d, affine)
output_nifti_path = BASE_DIR / 'Ground_truth' / "patient1" / 'DWI_Syn_sequence.nii.gz'
nib.save(nifti_img, output_nifti_path)

print(f"NIfTI file saved at: {output_nifti_path}")
