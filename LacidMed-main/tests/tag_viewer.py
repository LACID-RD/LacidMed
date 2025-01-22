import os
import pydicom
from pydicom.misc import is_dicom

# Path to the folder containing DICOM files
dicom_folder = "C:/Users/santi/Desktop/Volumetria/Slaibe_Yamila/23-TFE/tag_check"

# Loop through each file in the folder
for item in os.listdir(dicom_folder):
    item_path = os.path.join(dicom_folder, item)
    
    # Check if the item is a file and not a directory
    if os.path.isfile(item_path):
        # Check if the file is a valid DICOM file
        if is_dicom(item_path):
            # Read the DICOM file
            dicom_data = pydicom.dcmread(item_path)

            # Print the DICOM tags
            print(f"Tags for {item}:")
            print(dicom_data)
            print("\n")
        else:
            print(f"{item} is not a DICOM file.")
    else:
        print(f"{item} is a directory, skipping.")
