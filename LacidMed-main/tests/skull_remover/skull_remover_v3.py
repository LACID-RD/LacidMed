import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage import data  
from PIL import Image
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


def log_scale(image, scale_factor=1):
    """
    Escala no lineal usando logaritmo para resaltar bordes de mayor intensidad.
    
    Parámetros:
    - image: numpy array con la imagen de bordes.
    - scale_factor: Factor de escala para ajustar la intensidad (default=1).

    Retorna:
    - Imagen escalada con la función logarítmica.
    """
    # Añadir una constante pequeña para evitar log(0)
    image = np.float32(image)
    scaled_image = scale_factor * np.log1p(image)
    
    # Normalizar el resultado a [0, 255]
    scaled_image = cv2.normalize(scaled_image, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(scaled_image)


def power_scale(image, gamma=2.0):
    """
    Escala no lineal usando una función de potencia para resaltar bordes de mayor intensidad.
    
    Parámetros:
    - image: numpy array con la imagen de bordes.
    - gamma: Exponente de la función de potencia (default=2.0).

    Retorna:
    - Imagen escalada con la función de potencia.
    """
    image = np.float32(image)
    scaled_image = np.power(image, gamma)
    
    # Normalizar el resultado a [0, 255]
    scaled_image = cv2.normalize(scaled_image, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(scaled_image)


# Set directory path
directory_path_1 = "C:/Users/santi/Desktop/Fuesmen/imagenes/CASTRO NELIDA FLASH/S0001952/SE020097"

# Load MRI data
loader_1 = DicomLoaderMRI(directory_path=directory_path_1) 
vol_array_1 = loader_1.volumetric_array 
scaler_1 = Operations(volumetric_array_1=vol_array_1)
vol_array_1 = scaler_1.scale_matrix_to_value()

# Background removal
segmenter_1 = Segmentation(volumetric_array=vol_array_1)
bg_removed = segmenter_1.background_remover_volumetric(background_seed_point=[10,10,10], background_multiplier=3, background_number_of_iterations=10)
# Apply Sobel filter
filter_1 = Filters(sequence_directory=directory_path_1)
sobel_filtered = filter_1.sobel_image_filter(img_arr=bg_removed)
scaler_2 = Operations(volumetric_array_1=sobel_filtered)
sobel_filtered = scaler_2.scale_matrix_to_value()
# Apply thresholding
segmenter_2 = Segmentation(volumetric_array=sobel_filtered)
sobel_tresholded = segmenter_2.threshold_segmentation(lower_threshold=20)
scaler_3 = Operations(volumetric_array_1=sobel_tresholded)
sobel_tresholded = scaler_3.scale_matrix_to_value()
# Apply non-linear sclaling
sobel_rescaled_power_1 = power_scale(sobel_tresholded, gamma=2)
sobel_rescaled_log_1 = log_scale(sobel_rescaled_power_1, scale_factor=20)
sobel_rescaled_log_pow_1 = log_scale(sobel_rescaled_log_1, scale_factor=20)
sobel_rescaled_power_2 = power_scale(sobel_rescaled_power_1, gamma=2)
sobel_rescaled_log_2 = log_scale(sobel_rescaled_power_2, scale_factor=20)
sobel_rescaled_log_pow_2 = log_scale(sobel_rescaled_log_2, scale_factor=20)
sobel_rescaled_power_3 = power_scale(sobel_rescaled_power_2, gamma=2)
sobel_rescaled_log_3 = log_scale(sobel_rescaled_power_3, scale_factor=20)
sobel_rescaled_log_pow_3 = log_scale(sobel_rescaled_log_3, scale_factor=20)



# Visualización
fig, axs = plt.subplots(3, 3, figsize=(12, 8))  # Cambiar a 2 filas y 3 columnas

# Primera fila de gráficos
axs[0, 0].imshow(sobel_rescaled_power_1[:, :, 20], cmap='gray')
axs[0, 0].set_title('Imagen 1')
axs[0, 0].axis('off')

axs[0, 1].imshow(sobel_rescaled_power_2[:, :, 20], cmap='gray')
axs[0, 1].set_title('Imagen 2')
axs[0, 1].axis('off')

axs[0, 2].imshow(sobel_rescaled_power_3[:, :, 20], cmap='gray')
axs[0, 2].set_title('Imagen 3')
axs[0, 2].axis('off')

axs[0, 2].imshow(sobel_rescaled_power_3[:, :, 20], cmap='gray')
axs[0, 2].set_title('Imagen 3')
axs[0, 2].axis('off')

# Segunda fila de gráficos
axs[1, 0].imshow(sobel_rescaled_log_1[:, :, 20], cmap='gray')  
axs[1, 0].set_title('Imagen 4')
axs[1, 0].axis('off')

axs[1, 1].imshow(sobel_rescaled_log_2[:, :, 20], cmap='gray')
axs[1, 1].set_title('Imagen 5')
axs[1, 1].axis('off')

axs[1, 2].imshow(sobel_rescaled_log_3[:, :, 20], cmap='gray')
axs[1, 2].set_title('Imagen 6')
axs[1, 2].axis('off')

# Tercera fila de gráficos
axs[2, 0].imshow(sobel_rescaled_log_pow_1[:, :, 20], cmap='gray')  
axs[2, 0].set_title('Imagen 7')
axs[2, 0].axis('off')

axs[2, 1].imshow(sobel_rescaled_log_pow_2[:, :, 20], cmap='gray')
axs[2, 1].set_title('Imagen 8')
axs[2, 1].axis('off')

axs[2, 2].imshow(sobel_rescaled_log_pow_3[:, :, 20], cmap='gray')
axs[2, 2].set_title('Imagen 9')
axs[2, 2].axis('off')


plt.show()


# Save images (optional)
# sitk.WriteImage(sitk.GetImageFromArray(sobel_tresholded), "C:/Users/santi/Desktop/Fuesmen/imagenes/skull_remover/sobel_tresholded/sobel_tresholded.nrrd")
# writer = MultipleFileWriter(loader_1.sorted_files)
# writer.write(tresholded, 'C:/Users/santi/Desktop/Fuesmen/imagenes/skull_remover/tresholded')
