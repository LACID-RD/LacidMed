import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import ants

import nibabel as nib
import os
import cv2
import time
from nilearn import image
from skimage import data  
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import label
from skimage import data, img_as_float
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.draw import circle_perimeter


# lacid_med directory append.
import sys
sys.path.append("C:/Users/santi/Desktop/Fuesmen/code/LacidMed")
from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.handler.writer import SingleFileWriter, MultipleFileWriter, Deanonymizer
from lacid_med.src.processing.filters import Filters
from lacid_med.src.visualization.plotter import DicomPlotter
from lacid_med.src.visualization.histograms import HistogramGenerator
from lacid_med.src.processing.fitting import Fitting
from lacid_med.src.processing.operations import Operations
from lacid_med.src.processing.segmentation import Segmentation

def registration(fixed_path: str, moving_path: str, output_path: str = None):
    print("Registering ", moving_path, " to ", fixed_path)
    fixed_image = ants.image_read(fixed_path)
    moving_image = ants.image_read(moving_path)
    transform = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyN') 
    registered_image = transform['warpedmovout']
    index = moving_path.rfind("/")
    if index != -1 and output_path is None:
        output_path = moving_path[:index]  
        output_path = os.path.join(output_path, 'registered')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, 'registered.nii')
    ants.image_write(registered_image, output_path)
    print("Registered at ", output_path)

# def brain_segmentation(input_path: str, output_path: str = None):
#     image = ants.image_read(input_path)
#     brain_mask = compute_brain_mask(image) 
#     brain_extracted_image = image * brain_mask
#     index = input_path.rfind("/")
#     if index != -1 and output_path is None:
#         output_path = moving_path[:index]  
#         output_path = os.path.join(output_path, 'brain_segmentation')
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     output_path = os.path.join(output_path, 'brain_segmentation.nii')
#     ants.image_write(brain_extracted_image, output_path)



def main():
    # load the fixed and moving images
    fixed_path_1 = 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/23-CUBE/nii/23-CUBE.nii'
    moving_path_2 = 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/22-TFE/nii/22-TFE.nii'
    moving_path_3 = 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/21-TFE/nii/21-TFE.nii'
    moving_path_4 = 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/20-TFE/nii/20-TFE.nii'
    moving_path_5 = 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/19-CUBE/nii/19-CUBE.nii'
    
    image_1 = ants.image_read(fixed_path_1)
    image_2 = ants.image_read(moving_path_2)
    image_3 = ants.image_read(moving_path_3)
    image_4 = ants.image_read(moving_path_4)
    image_5 = ants.image_read(moving_path_5)
    
    # registration(fixed_path_1, moving_path_2, 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/22-TFE/nii/registered')
    # registration(fixed_path_1, moving_path_3, 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/21-TFE/nii/registered')
    # registration(fixed_path_1, moving_path_4, 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/20-TFE/nii/registered')
    # registration(fixed_path_1, moving_path_5, 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/19-CUBE/nii/registered')


    # load registered images. 
    registerd_path_1 = 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/23-CUBE/nii/23-CUBE.nii'
    registerd_path_2 = 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/22-TFE/nii/registered/registered.nii'
    registerd_path_3 = 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/21-TFE/nii/registered/registered.nii'
    registerd_path_4 = 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/20-TFE/nii/registered/registered.nii'
    registerd_path_5 = 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/19-CUBE/nii/registered/registered.nii'
    
    registered_1 = ants.image_read(registerd_path_1)
    registered_2 = ants.image_read(registerd_path_2)
    registered_3 = ants.image_read(registerd_path_3)
    registered_4 = ants.image_read(registerd_path_4)
    registered_5 = ants.image_read(registerd_path_5)
    
    # load brain mask.
    mask_path_1 = 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/23-CUBE/nii/brain_mask/23-CUBE-mask-brain-label.nii'
    mask_1 = ants.image_read(mask_path_1)
    # segment the remaining registered volumes using a single brain mask.
    brain_1 = np.multiply(registered_1, mask_1)
    brain_2 = np.multiply(registered_2, mask_1)
    brain_3 = np.multiply(registered_3, mask_1)
    brain_4 = np.multiply(registered_4, mask_1)
    brain_5 = np.multiply(registered_5, mask_1)
    
    ants.image_write(brain_1, 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/23-CUBE/nii/brain/23-CUBE-brain.nii')
    ants.image_write(brain_2, 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/22-TFE/nii/brain/22-TFE-brain.nii')
    ants.image_write(brain_3, 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/21-TFE/nii/brain/21-TFE-brain.nii')
    ants.image_write(brain_4, 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/20-TFE/nii/brain/20-TFE-brain.nii')
    ants.image_write(brain_5, 'C:/Users/santi/Desktop/Volumetria/Marquez_Gisela/19-CUBE/nii/brain/19-CUBE-brain.nii')
    
    brain_1_array = brain_1.numpy()
    brain_2_array = brain_2.numpy()
    brain_3_array = brain_3.numpy()
    brain_4_array = brain_4.numpy()
    brain_5_array = brain_5.numpy()
    
    scaler_1 = Operations(volumetric_array_1=brain_1_array)
    scaler_2 = Operations(volumetric_array_1=brain_2_array)
    scaler_3 = Operations(volumetric_array_1=brain_3_array)
    scaler_4 = Operations(volumetric_array_1=brain_4_array)
    scaler_5 = Operations(volumetric_array_1=brain_5_array)
    
    rescaled_brain_1 = scaler_1.scale_matrix_to_value()
    rescaled_brain_2 = scaler_2.scale_matrix_to_value()
    rescaled_brain_3 = scaler_3.scale_matrix_to_value()
    rescaled_brain_4 = scaler_4.scale_matrix_to_value()
    rescaled_brain_5 = scaler_5.scale_matrix_to_value()
    
    histogram_generator_1 = HistogramGenerator(array_3D=rescaled_brain_1)
    histogram_generator_2 = HistogramGenerator(array_3D=rescaled_brain_2)
    histogram_generator_3 = HistogramGenerator(array_3D=rescaled_brain_3)
    histogram_generator_4 = HistogramGenerator(array_3D=rescaled_brain_4)
    histogram_generator_5 = HistogramGenerator(array_3D=rescaled_brain_5)
    
    hist_1, bins_1 = histogram_generator_1.create_histogram_of_3d_array(offset=1, show=False)
    hist_2, bins_2 = histogram_generator_2.create_histogram_of_3d_array(offset=1, show=False)
    hist_3, bins_3 = histogram_generator_3.create_histogram_of_3d_array(offset=1, show=False)
    hist_4, bins_4 = histogram_generator_4.create_histogram_of_3d_array(offset=1, show=False)
    hist_5, bins_5 = histogram_generator_5.create_histogram_of_3d_array(offset=1, show=False)
    
    histogram_generator_7 = HistogramGenerator(array_3D=brain_1_array)
    histogram_generator_8 = HistogramGenerator(array_3D=brain_2_array)
    histogram_generator_9 = HistogramGenerator(array_3D=brain_3_array)
    histogram_generator_10 = HistogramGenerator(array_3D=brain_4_array)
    histogram_generator_11 = HistogramGenerator(array_3D=brain_5_array)
    
    hist_7, bins_7 = histogram_generator_7.create_histogram_of_3d_array(offset=1, show=False)
    hist_8, bins_8 = histogram_generator_8.create_histogram_of_3d_array(offset=1, show=False)
    hist_9, bins_9 = histogram_generator_9.create_histogram_of_3d_array(offset=1, show=False)
    hist_10, bins_10 = histogram_generator_10.create_histogram_of_3d_array(offset=1, show=False)
    hist_11, bins_11 = histogram_generator_11.create_histogram_of_3d_array(offset=1, show=False)
    
    
    gradient_hist_1 = np.gradient(hist_1)
    gradient_hist_2 = np.gradient(hist_2)
    gradient_hist_3 = np.gradient(hist_3)
    gradient_hist_4 = np.gradient(hist_4)
    gradient_hist_5 = np.gradient(hist_5)
    
    gradient_hist_7 = np.gradient(hist_7)
    gradient_hist_8 = np.gradient(hist_8)
    gradient_hist_9 = np.gradient(hist_9)
    gradient_hist_10 = np.gradient(hist_10)
    gradient_hist_11 = np.gradient(hist_11)
    

    hist_1_normalized = (hist_1 - np.min(hist_1)) / (np.max(hist_1) - np.min(hist_1))
    hist_2_normalized = (hist_2 - np.min(hist_2)) / (np.max(hist_2) - np.min(hist_2))
    hist_3_normalized = (hist_3 - np.min(hist_3)) / (np.max(hist_3) - np.min(hist_3))
    hist_4_normalized = (hist_4 - np.min(hist_4)) / (np.max(hist_4) - np.min(hist_4))
    hist_5_normalized = (hist_5 - np.min(hist_5)) / (np.max(hist_5) - np.min(hist_5))
    
    hist_7_normalized = (hist_7 - np.min(hist_7)) / (np.max(hist_7) - np.min(hist_7))
    hist_8_normalized = (hist_8 - np.min(hist_8)) / (np.max(hist_8) - np.min(hist_8))
    hist_9_normalized = (hist_9 - np.min(hist_9)) / (np.max(hist_9) - np.min(hist_9))
    hist_10_normalized = (hist_10 - np.min(hist_10)) / (np.max(hist_10) - np.min(hist_10))
    hist_11_normalized = (hist_11 - np.min(hist_11)) / (np.max(hist_11) - np.min(hist_11))
    

    global_max = np.max([max(hist_1), max(hist_2), max(hist_3), max(hist_4), max(hist_5)])
    hist_1_global_normalized = hist_1 / global_max
    hist_2_global_normalized = hist_2 / global_max
    hist_3_global_normalized = hist_3 / global_max
    hist_4_global_normalized = hist_4 / global_max
    hist_5_global_normalized = hist_5 / global_max


    # Define the window size (number of points to average over)
    window_size = 5
    window = np.ones(window_size) / window_size

    # Apply the moving average
    hist_1_global_normalized_smoothed = np.convolve(hist_1_global_normalized, window, mode='same')
    hist_2_global_normalized_smoothed = np.convolve(hist_2_global_normalized, window, mode='same')
    hist_3_global_normalized_smoothed = np.convolve(hist_3_global_normalized, window, mode='same')
    hist_4_global_normalized_smoothed = np.convolve(hist_4_global_normalized, window, mode='same')
    hist_5_global_normalized_smoothed = np.convolve(hist_5_global_normalized, window, mode='same')

    hist_1_global_normalized_smoothed_peaks, _ = find_peaks(hist_1_global_normalized_smoothed)
    hist_2_global_normalized_smoothed_peaks, _ = find_peaks(hist_2_global_normalized_smoothed)
    hist_3_global_normalized_smoothed_peaks, _ = find_peaks(hist_3_global_normalized_smoothed)
    hist_4_global_normalized_smoothed_peaks, _ = find_peaks(hist_4_global_normalized_smoothed)
    hist_5_global_normalized_smoothed_peaks, _ = find_peaks(hist_5_global_normalized_smoothed)

    hist_1_global_normalized_first_gaussian = np.zeros_like(hist_1_global_normalized)
    hist_1_global_normalized_first_gaussian[0:hist_1_global_normalized_smoothed_peaks[0]] = hist_1_global_normalized[0:hist_1_global_normalized_smoothed_peaks[0]] 
    hist_1_reversed_first_gaussian = np.flip(hist_1_global_normalized[0:hist_1_global_normalized_smoothed_peaks[0]])
    hist_1_global_normalized_first_gaussian[hist_1_global_normalized_smoothed_peaks[0]:hist_1_global_normalized_smoothed_peaks[0] * 2] = hist_1_reversed_first_gaussian

    hist_2_global_normalized_first_gaussian = np.zeros_like(hist_2_global_normalized)
    hist_2_global_normalized_first_gaussian[0:hist_2_global_normalized_smoothed_peaks[0]] = hist_2_global_normalized[0:hist_2_global_normalized_smoothed_peaks[0]] 
    hist_2_reversed_first_gaussian = np.flip(hist_2_global_normalized[0:hist_2_global_normalized_smoothed_peaks[0]])
    hist_2_global_normalized_first_gaussian[hist_2_global_normalized_smoothed_peaks[0]:hist_2_global_normalized_smoothed_peaks[0] * 2] = hist_2_reversed_first_gaussian

    hist_3_global_normalized_first_gaussian = np.zeros_like(hist_3_global_normalized)
    hist_3_global_normalized_first_gaussian[0:hist_3_global_normalized_smoothed_peaks[0]] = hist_3_global_normalized[0:hist_3_global_normalized_smoothed_peaks[0]] 
    hist_3_reversed_first_gaussian = np.flip(hist_3_global_normalized[0:hist_3_global_normalized_smoothed_peaks[0]])
    hist_3_global_normalized_first_gaussian[hist_3_global_normalized_smoothed_peaks[0]:hist_3_global_normalized_smoothed_peaks[0] * 2] = hist_3_reversed_first_gaussian

    hist_4_global_normalized_first_gaussian = np.zeros_like(hist_4_global_normalized)
    hist_4_global_normalized_first_gaussian[0:hist_4_global_normalized_smoothed_peaks[0]] = hist_4_global_normalized[0:hist_4_global_normalized_smoothed_peaks[0]]
    hist_4_reversed_first_gaussian = np.flip(hist_4_global_normalized[0:hist_4_global_normalized_smoothed_peaks[0]])
    hist_4_global_normalized_first_gaussian[hist_4_global_normalized_smoothed_peaks[0]:hist_4_global_normalized_smoothed_peaks[0] * 2] = hist_4_reversed_first_gaussian

    hist_5_global_normalized_first_gaussian = np.zeros_like(hist_5_global_normalized)
    hist_5_global_normalized_first_gaussian[0:hist_5_global_normalized_smoothed_peaks[0]] = hist_5_global_normalized[0:hist_5_global_normalized_smoothed_peaks[0]]
    hist_5_reversed_first_gaussian = np.flip(hist_5_global_normalized[0:hist_5_global_normalized_smoothed_peaks[0]])
    hist_5_global_normalized_first_gaussian[hist_5_global_normalized_smoothed_peaks[0]:hist_5_global_normalized_smoothed_peaks[0] * 2] = hist_5_reversed_first_gaussian


    hist_1_global_normalized_no_gaussian = hist_1_global_normalized - hist_1_global_normalized_first_gaussian
    hist_2_global_normalized_no_gaussian = hist_2_global_normalized - hist_2_global_normalized_first_gaussian
    hist_3_global_normalized_no_gaussian = hist_3_global_normalized - hist_3_global_normalized_first_gaussian
    hist_4_global_normalized_no_gaussian = hist_4_global_normalized - hist_4_global_normalized_first_gaussian
    hist_5_global_normalized_no_gaussian = hist_5_global_normalized - hist_5_global_normalized_first_gaussian

    global_max_no_scaling = np.max([max(hist_7), max(hist_8), max(hist_9), max(hist_10), max(hist_11)])
    hist_7_global_normalized = hist_7 / global_max_no_scaling
    hist_8_global_normalized = hist_8 / global_max_no_scaling
    hist_9_global_normalized = hist_9 / global_max_no_scaling
    hist_10_global_normalized = hist_10 / global_max_no_scaling
    hist_11_global_normalized = hist_11 / global_max_no_scaling

    # Define the window size (number of points to average over)
    window_size_no_scaling = 100
    window_no_scaling = np.ones(window_size_no_scaling) / window_size_no_scaling

    # Apply the moving average
    hist_7_global_normalized_smoothed = np.convolve(hist_7_global_normalized, window_no_scaling, mode='same')
    hist_8_global_normalized_smoothed = np.convolve(hist_8_global_normalized, window_no_scaling, mode='same')
    hist_9_global_normalized_smoothed = np.convolve(hist_9_global_normalized, window_no_scaling, mode='same')
    hist_10_global_normalized_smoothed = np.convolve(hist_10_global_normalized, window_no_scaling, mode='same')
    hist_11_global_normalized_smoothed = np.convolve(hist_11_global_normalized, window_no_scaling, mode='same')
    
    hist_7_global_normalized_smoothed_peaks, _ = find_peaks(hist_7_global_normalized_smoothed)
    hist_8_global_normalized_smoothed_peaks, _ = find_peaks(hist_8_global_normalized_smoothed)
    hist_9_global_normalized_smoothed_peaks, _ = find_peaks(hist_9_global_normalized_smoothed)
    hist_10_global_normalized_smoothed_peaks, _ = find_peaks(hist_10_global_normalized_smoothed)
    hist_11_global_normalized_smoothed_peaks, _ = find_peaks(hist_11_global_normalized_smoothed)

    hist_7_global_normalized_first_gaussian = np.zeros_like(hist_7_global_normalized) 
    hist_7_global_normalized_first_gaussian[0:hist_7_global_normalized_smoothed_peaks[0]] = hist_7_global_normalized[0:hist_7_global_normalized_smoothed_peaks[0]]
    hist_7_reversed_first_gaussian = np.flip(hist_7_global_normalized[0:hist_7_global_normalized_smoothed_peaks[0]])
    hist_7_global_normalized_first_gaussian[hist_7_global_normalized_smoothed_peaks[0]:hist_7_global_normalized_smoothed_peaks[0] * 2] = hist_7_reversed_first_gaussian

    hist_8_global_normalized_first_gaussian = np.zeros_like(hist_8_global_normalized) 
    hist_8_global_normalized_first_gaussian[0:hist_8_global_normalized_smoothed_peaks[0]] = hist_8_global_normalized[0:hist_8_global_normalized_smoothed_peaks[0]]
    hist_8_reversed_first_gaussian = np.flip(hist_8_global_normalized[0:hist_8_global_normalized_smoothed_peaks[0]])
    hist_8_global_normalized_first_gaussian[hist_8_global_normalized_smoothed_peaks[0]:hist_8_global_normalized_smoothed_peaks[0] * 2] = hist_8_reversed_first_gaussian

    hist_9_global_normalized_first_gaussian = np.zeros_like(hist_9_global_normalized)
    hist_9_global_normalized_first_gaussian[0:hist_9_global_normalized_smoothed_peaks[0]] = hist_9_global_normalized[0:hist_9_global_normalized_smoothed_peaks[0]]
    hist_9_reversed_first_gaussian = np.flip(hist_9_global_normalized[0:hist_9_global_normalized_smoothed_peaks[0]])
    hist_9_global_normalized_first_gaussian[hist_9_global_normalized_smoothed_peaks[0]:hist_9_global_normalized_smoothed_peaks[0] * 2] = hist_9_reversed_first_gaussian

    hist_10_global_normalized_first_gaussian = np.zeros_like(hist_10_global_normalized)
    hist_10_global_normalized_first_gaussian[0:hist_10_global_normalized_smoothed_peaks[0]] = hist_10_global_normalized[0:hist_10_global_normalized_smoothed_peaks[0]]
    hist_10_reversed_first_gaussian = np.flip(hist_10_global_normalized[0:hist_10_global_normalized_smoothed_peaks[0]])
    hist_10_global_normalized_first_gaussian[hist_10_global_normalized_smoothed_peaks[0]:hist_10_global_normalized_smoothed_peaks[0] * 2] = hist_10_reversed_first_gaussian

    hist_11_global_normalized_first_gaussian = np.zeros_like(hist_11_global_normalized)
    hist_11_global_normalized_first_gaussian[0:hist_11_global_normalized_smoothed_peaks[0]] = hist_11_global_normalized[0:hist_11_global_normalized_smoothed_peaks[0]]
    hist_11_reversed_first_gaussian = np.flip(hist_11_global_normalized[0:hist_11_global_normalized_smoothed_peaks[0]])
    hist_11_global_normalized_first_gaussian[hist_11_global_normalized_smoothed_peaks[0]:hist_11_global_normalized_smoothed_peaks[0] * 2] = hist_11_reversed_first_gaussian


    hist_1_global_normalized_no_gaussian = hist_1_global_normalized - hist_1_global_normalized_first_gaussian
    hist_2_global_normalized_no_gaussian = hist_2_global_normalized - hist_2_global_normalized_first_gaussian
    hist_3_global_normalized_no_gaussian = hist_3_global_normalized - hist_3_global_normalized_first_gaussian
    hist_4_global_normalized_no_gaussian = hist_4_global_normalized - hist_4_global_normalized_first_gaussian
    hist_5_global_normalized_no_gaussian = hist_5_global_normalized - hist_5_global_normalized_first_gaussian
    
    hist_7_global_normalized_no_gaussian = hist_7_global_normalized - hist_7_global_normalized_first_gaussian
    hist_8_global_normalized_no_gaussian = hist_8_global_normalized - hist_8_global_normalized_first_gaussian
    hist_9_global_normalized_no_gaussian = hist_9_global_normalized - hist_9_global_normalized_first_gaussian
    hist_10_global_normalized_no_gaussian = hist_10_global_normalized - hist_10_global_normalized_first_gaussian
    hist_11_global_normalized_no_gaussian = hist_11_global_normalized - hist_11_global_normalized_first_gaussian


    gradient_hist_1_normalized = (gradient_hist_1 - np.min(gradient_hist_1)) / (np.max(gradient_hist_1) - np.min(gradient_hist_1))
    gradient_hist_2_normalized = (gradient_hist_2 - np.min(gradient_hist_2)) / (np.max(gradient_hist_2) - np.min(gradient_hist_2))
    gradient_hist_3_normalized = (gradient_hist_3 - np.min(gradient_hist_3)) / (np.max(gradient_hist_3) - np.min(gradient_hist_3))
    gradient_hist_4_normalized = (gradient_hist_4 - np.min(gradient_hist_4)) / (np.max(gradient_hist_4) - np.min(gradient_hist_4))
    gradient_hist_5_normalized = (gradient_hist_5 - np.min(gradient_hist_5)) / (np.max(gradient_hist_5) - np.min(gradient_hist_5))
    
    gradient_hist_7_normalized = (gradient_hist_7 - np.min(gradient_hist_7)) / (np.max(gradient_hist_7) - np.min(gradient_hist_7))
    gradient_hist_8_normalized = (gradient_hist_8 - np.min(gradient_hist_8)) / (np.max(gradient_hist_8) - np.min(gradient_hist_8))
    gradient_hist_9_normalized = (gradient_hist_9 - np.min(gradient_hist_9)) / (np.max(gradient_hist_9) - np.min(gradient_hist_9))
    gradient_hist_10_normalized = (gradient_hist_10 - np.min(gradient_hist_10)) / (np.max(gradient_hist_10) - np.min(gradient_hist_10))
    gradient_hist_11_normalized = (gradient_hist_11 - np.min(gradient_hist_11)) / (np.max(gradient_hist_11) - np.min(gradient_hist_11))
    

    # Plot the original array and its gradient
    plt.plot(bins_1, hist_1_global_normalized, label='Hist 1 scaled normalized')
    plt.plot(bins_1, hist_1_global_normalized_smoothed, label='Hist 1 scaled normalized smoothed')
    plt.plot(bins_1[hist_1_global_normalized_smoothed_peaks[0]], hist_1_global_normalized_smoothed[hist_1_global_normalized_smoothed_peaks[0]], 'ro')
    plt.plot(bins_1, hist_1_global_normalized_first_gaussian, label='Hist 1 first gaussian')
    plt.plot(bins_2, hist_2_global_normalized, label='Hist 2 scaled normalized')
    plt.plot(bins_2, hist_2_global_normalized_smoothed, label='Hist 2 scaled normalized smoothed')
    plt.plot(bins_2[hist_2_global_normalized_smoothed_peaks[0]], hist_2_global_normalized_smoothed[hist_2_global_normalized_smoothed_peaks[0]], 'ro')
    plt.plot(bins_2, hist_2_global_normalized_first_gaussian, label='Hist 2 first gaussian')
    plt.plot(bins_3, hist_3_global_normalized, label='Hist 3 scaled normalized')
    plt.plot(bins_3, hist_3_global_normalized_smoothed, label='Hist 3 scaled normalized smoothed')
    plt.plot(bins_3[hist_3_global_normalized_smoothed_peaks[0]], hist_3_global_normalized_smoothed[hist_3_global_normalized_smoothed_peaks[0]], 'ro')
    plt.plot(bins_3, hist_3_global_normalized_first_gaussian, label='Hist 3 first gaussian')
    plt.plot(bins_4, hist_4_global_normalized, label='Hist 4 scaled normalized')
    plt.plot(bins_4, hist_4_global_normalized_smoothed, label='Hist 4 scaled normalized smoothed')
    plt.plot(bins_4[hist_4_global_normalized_smoothed_peaks[0]], hist_4_global_normalized_smoothed[hist_4_global_normalized_smoothed_peaks[0]], 'ro')
    plt.plot(bins_4, hist_4_global_normalized_first_gaussian, label='Hist 4 first gaussian')
    plt.plot(bins_5, hist_5_global_normalized, label='Hist 5 scaled normalized')
    plt.plot(bins_5, hist_5_global_normalized_smoothed, label='Hist 5 scaled normalized smoothed')
    plt.plot(bins_5[hist_5_global_normalized_smoothed_peaks[0]], hist_5_global_normalized_smoothed[hist_5_global_normalized_smoothed_peaks[0]], 'ro')
    plt.plot(bins_5, hist_5_global_normalized_first_gaussian, label='Hist 5 first gaussian')
    
    
    # Add labels and legend
    plt.xlabel('Intensity')
    plt.ylabel('Normalized Values')
    plt.title('Global normalized and scaled Histograms')
    plt.legend()

    # Show the plot
    plt.show()

    # Plot the original array and its gradient
    plt.plot(bins_1, hist_1_global_normalized_no_gaussian, label='Hist 1, no first gaussian')
    plt.plot(bins_2, hist_2_global_normalized_no_gaussian, label='Hist 2, no first gaussian')
    plt.plot(bins_3, hist_3_global_normalized_no_gaussian, label='Hist 3, no first gaussian')
    plt.plot(bins_4, hist_4_global_normalized_no_gaussian, label='Hist 4, no first gaussian')
    plt.plot(bins_5, hist_5_global_normalized_no_gaussian, label='Hist 5, no first gaussian')
    
    
    # Add labels and legend
    plt.xlabel('Intensity')
    plt.ylabel('Normalized Values')
    plt.title('Histograms, no first gaussian')
    plt.legend()

    # Show the plot
    plt.show()
    

    # Find unique values
    unique_values = np.unique(brain_5_array)
    # Print unique values
    print("Unique values in the matrix:", unique_values)
    print("lenght unique values in the matrix: ", len(unique_values))
    print("max value in the matrix: ", np.max(unique_values))

    print("brain_1 is: ", type(brain_1))
    print("brain_1_array is: ", type(brain_1_array))
    print("image_1 shape: ", image_1.shape)
    print("registered_1 shape: ", registered_1.shape)
    print("mask_1 shape: ", mask_1.shape)
    print("brain_1 shape: ", brain_1.shape)

    volume_1 = np.sum(hist_1_global_normalized_no_gaussian)
    volume_2 = np.sum(hist_2_global_normalized_no_gaussian)
    volume_3 = np.sum(hist_3_global_normalized_no_gaussian)
    volume_4 = np.sum(hist_4_global_normalized_no_gaussian)
    volume_5 = np.sum(hist_5_global_normalized_no_gaussian)
    
    print("volume_1 is: ", volume_1)
    print("volume_2 is: ", volume_2)
    print("volume_3 is: ", volume_3)  
    print("volume_4 is: ", volume_4)
    print("volume_5 is: ", volume_5)

    full_volume_1 = np.sum(hist_7)
    full_volume_2 = np.sum(hist_8)
    full_volume_3 = np.sum(hist_9)
    full_volume_4 = np.sum(hist_10)
    full_volume_5 = np.sum(hist_11)

    print("Full volume_1 is: ", full_volume_1)
    print("Full volume_2 is: ", full_volume_2)
    print("Full volume_3 is: ", full_volume_3)
    print("Full volume_4 is: ", full_volume_4)
    print("Full volume_5 is: ", full_volume_5)

    print("Full volume 1 without gaussian: ", np.sum(hist_7_global_normalized_no_gaussian))
    print("Full volume 2 without gaussian: ", np.sum(hist_8_global_normalized_no_gaussian))
    print("Full volume 3 without gaussian: ", np.sum(hist_9_global_normalized_no_gaussian))
    print("Full volume 4 without gaussian: ", np.sum(hist_10_global_normalized_no_gaussian))
    print("Full volume 5 without gaussian: ", np.sum(hist_11_global_normalized_no_gaussian))



if __name__ == "__main__":
    main()