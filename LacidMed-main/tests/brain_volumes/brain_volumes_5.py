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
    # Cargar los path de los volumenes en nii (los que estan guardados sueltos en la carpeta nii). 
    # Fixed path deberia ser el ultimo volumen, del cual generamos la mascara.
    fixed_path_1 = 'C:/Users/santi/Desktop/Volumetria/Federici_Natalia/2023-TFE-218/nii/2023-TFE-218.nii'
    moving_path_2 = 'C:/Users/santi/Desktop/Volumetria/Federici_Natalia/2019-TFE-218/nii/2019-TFE-218.nii'
    

    # Se cargan los volumenes enteros.
    image_1 = ants.image_read(fixed_path_1)
    image_2 = ants.image_read(moving_path_2)

    
    # Se realiza la registacion usando la funcion registration. No es actualmente una clase. 
    # El output_path es donde se va a guardar la imagen registrada. Se tiene que guardar en la carpeta registered dentro de la carpeta nii.
    # registration(fixed_path_1, moving_path_2, output_path= "C:/Users/santi/Desktop/Volumetria/Federici_Natalia/2019-TFE-218/nii/registered")
    

    # Cargar los paths de las imagenes registradas.
    registerd_path_1 = 'C:/Users/santi/Desktop/Volumetria/Federici_Natalia/2023-TFE-218/nii/2023-TFE-218.nii'
    registerd_path_2 = 'C:/Users/santi/Desktop/Volumetria/Federici_Natalia/2019-TFE-218/nii/registered/registered.nii'
    

    # Se cargan las imagenes registradas. 
    # registered_1 es el volumen mas reciente, al cual se registran el resto de volumenes.
    registered_1 = ants.image_read(registerd_path_1)
    registered_2 = ants.image_read(registerd_path_2)
    

    # Cargar el path de la mascara. 
    # Cargar la mascara.
    mask_path_1 = 'C:/Users/santi/Desktop/Volumetria/Federici_Natalia/2023-TFE-218/nii/brain_mask/2023-TFE-218-mask-brain-label.nii'
    mask_1 = ants.image_read(mask_path_1)


    # Se multiplica la mascara a los volumenes registrados para extraer el cerebro.
    brain_1 = np.multiply(registered_1, mask_1)
    brain_2 = np.multiply(registered_2, mask_1)
    

    # Se guardan los cerebros en .nii en la carpeta de brain.
    ants.image_write(brain_1, 'C:/Users/santi/Desktop/Volumetria/Federici_Natalia/2023-TFE-218/nii/brain/brain-2023-TFE-218.nii')
    ants.image_write(brain_2, 'C:/Users/santi/Desktop/Volumetria/Federici_Natalia/2019-TFE-218/nii/brain/brain-2019-TFE-218.nii')
    

    # Convertir los volumenes a arrays de numpy para poder usar operations.
    brain_1_array = brain_1.numpy()
    brain_2_array = brain_2.numpy()
    

    # Se escalan los volumenes a 255 con la clase operations.
    scaler_1 = Operations(volumetric_array_1=brain_1_array)
    scaler_2 = Operations(volumetric_array_1=brain_2_array)
    rescaled_brain_1 = scaler_1.scale_matrix_to_value()
    rescaled_brain_2 = scaler_2.scale_matrix_to_value()
    
    
    # Se crean los histogramas de los cerebros escalados.
    histogram_generator_1 = HistogramGenerator(array_3D=rescaled_brain_1)
    histogram_generator_2 = HistogramGenerator(array_3D=rescaled_brain_2)
    hist_1, bins_1 = histogram_generator_1.create_histogram_of_3d_array(offset=1, show=False)
    hist_2, bins_2 = histogram_generator_2.create_histogram_of_3d_array(offset=1, show=False)
    

    # Se crean los histogramas de los cerebros sin escalar.
    histogram_generator_7 = HistogramGenerator(array_3D=brain_1_array)
    histogram_generator_8 = HistogramGenerator(array_3D=brain_2_array)
    hist_7, bins_7 = histogram_generator_7.create_histogram_of_3d_array(offset=1, show=False)
    hist_8, bins_8 = histogram_generator_8.create_histogram_of_3d_array(offset=1, show=False)



    hist_7_no_zeroes = np.copy(hist_7)
    bins_7_no_zeroes = np.copy(bins_7)
    bins_7_no_zeroes = bins_7_no_zeroes[hist_7_no_zeroes != 0]
    hist_7_no_zeroes = hist_7_no_zeroes[hist_7_no_zeroes != 0]
    hist_7 = hist_7_no_zeroes
    bins_7 = bins_7_no_zeroes

    hist_8_no_zeroes = np.copy(hist_8)
    bins_8_no_zeroes = np.copy(bins_8)
    bins_8_no_zeroes = bins_8_no_zeroes[hist_8_no_zeroes != 0]
    hist_8_no_zeroes = hist_8_no_zeroes[hist_8_no_zeroes != 0]
    hist_8 = hist_8_no_zeroes
    bins_8 = bins_8_no_zeroes


    volume_hist_7_no_zeroes = np.sum(hist_7_no_zeroes)
    volume_hist_8_no_zeroes = np.sum(hist_8_no_zeroes)


    # Se normalizan los histogramas escalados con el maximo global.
    global_max = np.max([max(hist_1), max(hist_2)])
    hist_1_global_normalized = hist_1 / global_max
    hist_2_global_normalized = hist_2 / global_max


    # Se normalizan los histogramas sin escalar con el maximo global.
    global_max = np.max([max(hist_7), max(hist_8)])
    hist_7_global_normalized = hist_7 / global_max
    hist_8_global_normalized = hist_8 / global_max
    

    # Se define el tamano de la ventana con la que se suaviza el histograma escalado. 
    # El histograma escalado tiene una escala de 0 a 255, por lo que una ventana de 5 muestras sera suficiente.
    window_size = 5
    window = np.ones(window_size) / window_size
    

    # Se suavizan los histogramas escalados.
    # Se buscan los picos en los histogramas escalados suavizados.
    # Se extrae la primera gaussiana del histograma escalado suavizado. 
    # Se elimina la primera gaussiana del histograma escalado suavizado.
    hist_1_global_normalized_smoothed = np.convolve(hist_1_global_normalized, window, mode='same')
    hist_1_global_normalized_smoothed_peaks, _ = find_peaks(hist_1_global_normalized_smoothed)
    hist_1_global_normalized_first_gaussian = np.zeros_like(hist_1_global_normalized)
    hist_1_global_normalized_first_gaussian[0:hist_1_global_normalized_smoothed_peaks[0]] = hist_1_global_normalized[0:hist_1_global_normalized_smoothed_peaks[0]] 
    hist_1_reversed_first_gaussian = np.flip(hist_1_global_normalized[0:hist_1_global_normalized_smoothed_peaks[0]])
    hist_1_global_normalized_first_gaussian[hist_1_global_normalized_smoothed_peaks[0]:hist_1_global_normalized_smoothed_peaks[0] * 2] = hist_1_reversed_first_gaussian
    hist_1_global_normalized_no_gaussian = hist_1_global_normalized - hist_1_global_normalized_first_gaussian

    
    hist_2_global_normalized_smoothed = np.convolve(hist_2_global_normalized, window, mode='same')
    hist_2_global_normalized_smoothed_peaks, _ = find_peaks(hist_2_global_normalized_smoothed)
    hist_2_global_normalized_first_gaussian = np.zeros_like(hist_2_global_normalized)
    hist_2_global_normalized_first_gaussian[0:hist_2_global_normalized_smoothed_peaks[0]] = hist_2_global_normalized[0:hist_2_global_normalized_smoothed_peaks[0]] 
    hist_2_reversed_first_gaussian = np.flip(hist_2_global_normalized[0:hist_2_global_normalized_smoothed_peaks[0]])
    hist_2_global_normalized_first_gaussian[hist_2_global_normalized_smoothed_peaks[0]:hist_2_global_normalized_smoothed_peaks[0] * 2] = hist_2_reversed_first_gaussian
    hist_2_global_normalized_no_gaussian = hist_2_global_normalized - hist_2_global_normalized_first_gaussian


    # Se suavizan los histogramas escalados.
    # Se buscan los picos en los histogramas suavizados.
    # Se extrae la primera gaussiana del histograma suavizado. 
    # Se elimina la primera gaussiana del histograma suavizado.
    hist_7_global_normalized_smoothed = np.convolve(hist_7_global_normalized, window, mode='same')
    hist_7_global_normalized_smoothed_peaks, _ = find_peaks(hist_7_global_normalized_smoothed)
    hist_7_global_normalized_first_gaussian = np.zeros_like(hist_7_global_normalized)
    hist_7_global_normalized_first_gaussian[0:hist_7_global_normalized_smoothed_peaks[0]] = hist_7_global_normalized[0:hist_7_global_normalized_smoothed_peaks[0]] 
    hist_7_reversed_first_gaussian = np.flip(hist_7_global_normalized[0:hist_7_global_normalized_smoothed_peaks[0]])
    hist_7_global_normalized_first_gaussian[hist_7_global_normalized_smoothed_peaks[0]:hist_7_global_normalized_smoothed_peaks[0] * 2] = hist_7_reversed_first_gaussian
    hist_7_global_normalized_no_gaussian = hist_7_global_normalized - hist_7_global_normalized_first_gaussian

    
    hist_8_global_normalized_smoothed = np.convolve(hist_8_global_normalized, window, mode='same')
    hist_8_global_normalized_smoothed_peaks, _ = find_peaks(hist_8_global_normalized_smoothed)
    hist_8_global_normalized_first_gaussian = np.zeros_like(hist_8_global_normalized)
    hist_8_global_normalized_first_gaussian[0:hist_8_global_normalized_smoothed_peaks[0]] = hist_8_global_normalized[0:hist_8_global_normalized_smoothed_peaks[0]] 
    hist_8_reversed_first_gaussian = np.flip(hist_8_global_normalized[0:hist_8_global_normalized_smoothed_peaks[0]])
    hist_8_global_normalized_first_gaussian[hist_8_global_normalized_smoothed_peaks[0]:hist_8_global_normalized_smoothed_peaks[0] * 2] = hist_8_reversed_first_gaussian
    hist_8_global_normalized_no_gaussian = hist_8_global_normalized - hist_8_global_normalized_first_gaussian
    
    
    plt.plot(bins_1, hist_1_global_normalized, label='Hist 1 scaled normalized')
    plt.plot(bins_1, hist_1_global_normalized_smoothed, label='Hist 1 scaled normalized smoothed')
    plt.plot(bins_1[hist_1_global_normalized_smoothed_peaks[0]], hist_1_global_normalized_smoothed[hist_1_global_normalized_smoothed_peaks[0]], 'ro')
    plt.plot(bins_1, hist_1_global_normalized_first_gaussian, label='Hist 1 first gaussian')
    plt.plot(bins_2, hist_2_global_normalized, label='Hist 2 scaled normalized')
    plt.plot(bins_2, hist_2_global_normalized_smoothed, label='Hist 2 scaled normalized smoothed')
    plt.plot(bins_2[hist_2_global_normalized_smoothed_peaks[0]], hist_2_global_normalized_smoothed[hist_2_global_normalized_smoothed_peaks[0]], 'ro')
    plt.plot(bins_2, hist_2_global_normalized_first_gaussian, label='Hist 2 first gaussian')
    plt.xlabel('Intensity')
    plt.ylabel('Normalized Values')
    plt.title('Global normalized and scaled Histograms')
    plt.legend()
    plt.show()


    plt.plot(bins_7, hist_7_global_normalized, label='Hist 7 normalized')
    plt.plot(bins_7, hist_7_global_normalized_smoothed, label='Hist 7 normalized smoothed')
    plt.plot(bins_7[hist_7_global_normalized_smoothed_peaks[0]], hist_7_global_normalized_smoothed[hist_7_global_normalized_smoothed_peaks[0]], 'ro')
    plt.plot(bins_7, hist_7_global_normalized_first_gaussian, label='Hist 7 first gaussian')
    plt.plot(bins_8, hist_8_global_normalized, label='Hist 8 normalized')
    plt.plot(bins_8, hist_8_global_normalized_smoothed, label='Hist 8 normalized smoothed')
    plt.plot(bins_8[hist_8_global_normalized_smoothed_peaks[0]], hist_8_global_normalized_smoothed[hist_8_global_normalized_smoothed_peaks[0]], 'ro')
    plt.plot(bins_8, hist_8_global_normalized_first_gaussian, label='Hist 8 first gaussian')
    plt.xlabel('Intensity')
    plt.ylabel('Normalized Values')
    plt.title('Global normalized Histograms')
    plt.legend()
    plt.show()


    plt.plot(bins_1, hist_1_global_normalized_no_gaussian, label='Hist 1, no first gaussian')
    plt.plot(bins_2, hist_2_global_normalized_no_gaussian, label='Hist 2, no first gaussian')    
    # Add labels and legend
    plt.xlabel('Intensity')
    plt.ylabel('Normalized Values')
    plt.title('Histograms, no first gaussian')
    plt.legend()
    # Show the plot
    plt.show()


    plt.plot(bins_7_no_zeroes, hist_7_no_zeroes, label='Hist 7, no zeroes')
    # Add labels and legend
    plt.title('Histogram, no zeroes')
    plt.legend()
    # Show the plot
    plt.show()


    plt.plot(bins_8_no_zeroes, hist_8_no_zeroes, label='Hist 8, no zeroes')
    # Add labels and legend
    plt.title('Histogram, no zeroes')
    plt.legend()
    # Show the plot
    plt.show()



    
    
    
if __name__ == "__main__":
    main()
