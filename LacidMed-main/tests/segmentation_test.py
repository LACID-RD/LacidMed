import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
from skimage import data  
from PIL import Image
from scipy.ndimage import label

# lacid_med directory append.
import sys
sys.path.append("C:/Users/santi/Desktop/Fuesmen/code/LacidMed")
from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.handler.writer import SingleFileWriter
from lacid_med.src.handler.writer import MultipleFileWriter
from lacid_med.src.handler.writer import Deanonymizer
from lacid_med.src.processing.filters import Filters
from lacid_med.src.visualization.plotter import DicomPlotter
from lacid_med.src.visualization.histograms import HistogramGenerator
from lacid_med.src.processing.fitting import Fitting
from lacid_med.src.processing.operations import Operations
from lacid_med.src.processing.segmentation import Segmentation

# dicom_path = "C:/Users/santi/Desktop/Fuesmen/imagenes/Segmented/MR.1.2.840.113619.2.363.10499743.3637646.25026.1684513445.974.70.dcm"
# dicom_image = pydicom.dcmread(dicom_path)
# dicom_array = dicom_image.pixel_array
# filtrator_1 = Filters(sequence_directory=dicom_path)
# pre_filt = filtrator_1.apply_minimum_filter(img_arr=dicom_array, kernel_size=6)
# segmenter_1 = Segmentation(image_array=dicom_array)
# segmentation_array = segmenter_1.region_growing(
#             seed_point=[20, 20], 
#             multiplier=5,
#             number_of_iterations=100, 
#             invert_mask=True
#             )
# filtered_array = np.multiply(segmentation_array, dicom_array)

# segmenter_2 = Segmentation(image_array=segmentation_array) 
# segmentation_array_2 = segmenter_2.region_growing(
#             seed_point=[140, 140], 
#             multiplier=3,
#             number_of_iterations=1, 
#             invert_mask=False
#             )

# fig, axs = plt.subplots(1, 4, figsize=(12, 6))
# axs[0].imshow(dicom_array, cmap='gray')
# axs[0].set_title('Original Image')
# axs[0].axis('off')
# axs[1].imshow(pre_filt, cmap='gray')
# axs[1].set_title('Pre filter')
# axs[1].axis('off')
# axs[2].imshow(segmentation_array_2, cmap='gray')
# axs[2].set_title('Mask')
# axs[2].axis('off')
# axs[3].imshow(filtered_array, cmap='gray')
# axs[3].set_title('Filtered image')
# axs[3].axis('off')
# plt.show()

# directory_path_1 = "C:/Users/santi/Desktop/Fuesmen/imagenes/Fuesmen/imagenes/ZTE_H01_descomp"
# loader_1 = DicomLoaderMRI(directory_path=directory_path_1) 
# vol_array_1 = loader_1.volumetric_array 
# segmenter_1 = Segmentation(volumetric_array=vol_array_1)
# segmented_vol_1 = segmenter_1.region_growing(
#             seed_point=[10, 10, 10], 
#             multiplier=5, 
#             number_of_iterations=10, 
#             invert_mask=True
#             )
# filtered_vol_1 = np.multiply(segmented_vol_1, vol_array_1)

# sitk.WriteImage(sitk.GetImageFromArray(filtered_vol_1),"C:/Users/santi/Desktop/Fuesmen/imagenes/segmented_vol.nrrd")
# writer = MultipleFileWriter(loader_1.sorted_files)
# writer.write(filtered_vol_1, 'C:/Users/santi/Desktop/Fuesmen/imagenes/Segmented')

# directory_path = 'C:/Users/santi/Desktop/Fuesmen/imagenes/Segmented'
# loader_2 = DicomLoaderMRI(directory_path=directory_path) 
# vol_array_2 = loader_2.volumetric_array 
# segmenter_2 = Segmentation(volumetric_array=vol_array_2)
# largest_component = segmenter_2.background_remover_volumetric()
# sitk.WriteImage(sitk.GetImageFromArray(largest_component),"C:/Users/santi/Desktop/Fuesmen/imagenes/largest_component.nrrd")
# writer = MultipleFileWriter(loader_2.sorted_files)
# writer.write(largest_component, 'C:/Users/santi/Desktop/Fuesmen/imagenes/largest_component')
# segmenter_3 = Segmentation(volumetric_array=largest_component)
# canny_volume = segmenter_3.canny_segmentation()

directory_path = 'C:/Users/santi/Desktop/Fuesmen/imagenes/largest_component'
loader_3 = DicomLoaderMRI(directory_path=directory_path) 
vol_array_3 = loader_3.volumetric_array
segmenter_3 = Segmentation(volumetric_array=vol_array_3)
region_growing = segmenter_3.region_growing(seed_point=[218, 86, 70], multiplier=1.5, number_of_iterations=5, invert_mask=None)




print("The volume shape is: ", vol_array_3.shape)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(vol_array_3[:, :, 80], cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(region_growing[:, :, 70], cmap='gray')
axs[1].set_title('Region growing test')
axs[1].axis('off')

plt.show()


if not os.path.exists('C:/Users/santi/Desktop/Fuesmen/imagenes/region_growing'):
    os.makedirs('C:/Users/santi/Desktop/Fuesmen/imagenes/region_growing', exist_ok=True)
writer = MultipleFileWriter(loader_3.sorted_files)
writer.write(region_growing, 'C:/Users/santi/Desktop/Fuesmen/imagenes/region_growing')


