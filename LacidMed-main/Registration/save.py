#Code written to save the images in the folder "Registered". Instead of creating new dicom files, it takes an already
#existing one and replaces the original pixel array.

import os
import numpy as np
import pydicom
from pathlib import Path

#Function created due to difference between the contrast of the original image and the registrated one.
def adjust_contrast(pixel_array, min_val=0, max_val=None):
    pixel_array = pixel_array - pixel_array.min()
    if pixel_array.max() != 0:  # Avoid division by zero
        pixel_array = pixel_array / pixel_array.max() * (max_val - min_val)
    return np.clip(pixel_array + min_val, min_val, max_val).astype(np.uint16)


def save(reg_img):
    BASE_DIR = Path(__file__).resolve().parent  
    input_path = BASE_DIR / 'images'
    output_path = BASE_DIR / 'registered'
    
    #b-values (manually selected)
    b_values_list = [0, 25, 50, 100, 150, 200] 
    input_files = [os.path.join(input_path, f"b_{b}.dcm") for b in b_values_list]
    output_files = [os.path.join(output_path, f"b_{b}_registered.dcm") for b in b_values_list]

    for i, pixel_array in enumerate(reg_img):
        # Read the input DICOM to get the corresponding max value
        ds = pydicom.dcmread(input_files[i])
        original_pixel_array = ds.pixel_array
        max_val = int(original_pixel_array.max())  # Dynamically get max value

        # Adjust contrast using the original image's max
        pixel_array = adjust_contrast(pixel_array, max_val=max_val)

        # Replace and save
        ds.PixelData = np.array(pixel_array, dtype=np.uint16).tobytes()
        ds.save_as(output_files[i])
    
    print("All files saved correctly.")
