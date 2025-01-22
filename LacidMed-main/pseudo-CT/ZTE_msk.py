import sys
sys.path.append("c:/codigos_tesis/Repositorio/LacidMed") #le indica al script de python donde se encuentra el paquete de clases que se utilizarán

import pydicom
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import SimpleITK as sitk 
import cython
#Se importan clases específicas de módulos dentro del paquete lacid_med
from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.processing.filters import Filters
from lacid_med.src.visualization.plotter import DicomPlotter
from lacid_med.src.visualization.histograms import HistogramGenerator
from lacid_med.src.processing.fitting import Fitting
from lacid_med.src.processing.operations import Operations


def main():
    OFFSET=3
    sequence_dir = "c:/Users/Gabriel/OneDrive - alumno.um.edu.ar/FUESMEN/ZTE_H01"

    loader = DicomLoaderMRI(directory_path=sequence_dir) #se instancia la clase Loader
    org_vol = loader.volumetric_array #raw data organizada
    
    #N4 filter
    filter = Filters(sequence_directory=sequence_dir)
    n4_files = filter.N4_bias_correction_filter(max_iterations=[50,50,50], convergence_threshold=0.001,mask_image=None)
    sitk.WriteImage(sitk.GetImageFromArray(n4_files), "c:/codigos_tesis/n4_corrected.nrrd")
    
    #Variable de imagenes filtradas 
    image = sitk.ReadImage("c:/codigos_tesis/n4_corrected.nrrd") #imagenes filtradas
    vol_filt= sitk.GetArrayFromImage(image) #vol_fit es una matriz 3D de imagenes filtradas
    
    vol_filt_stack = np.dstack(vol_filt) #dstack apila los arrays
    sitk.WriteImage(sitk.GetImageFromArray(vol_filt_stack),"c:/codigos_tesis/no_norm_filt.nrrd")

    # normalizado.    
    escalador_1 = Operations(volumetric_array_1=org_vol)
    escalador_2 = Operations(volumetric_array_1=vol_filt)

    norm_vol_raw = escalador_1.scale_matrix_to_value(value=255)
    norm_vol_filt = escalador_2.scale_matrix_to_value(value=255)


    sitk.WriteImage(sitk.GetImageFromArray(norm_vol_filt),"c:/codigos_tesis/filtrada_norm.nrrd")

    #Histogram from image raw normalized
    histogram3D = HistogramGenerator(array_3D=norm_vol_raw)
    hist_raw, bins = histogram3D.create_mean_histogram_of_3d_array(offset=OFFSET, show=True, plot_title="Histogram of raw image", xlabel="Pixel value", ylabel="Frequency")

    #Histogram from image norm filtered
    histogram3D = HistogramGenerator(array_3D=norm_vol_filt)
    hist_filt, bins = histogram3D.create_mean_histogram_of_3d_array(offset=OFFSET, show=True, plot_title="Histogram of filtered image normalized", xlabel="Pixel value", ylabel="Frequency")
    
    #Fiteo first gaussian
    fit_1= Fitting(x=bins[0:70], y=hist_filt[0:70])
    gaussian_fit_1= fit_1.fit_gaussian_to_histogram(initial_guess=[12, 3000, 0.7])
   
    #Fiteo second gaussian
    fit_2= Fitting(x=bins[70:210], y=hist_filt[70:210])
    gaussian_fit_2= fit_2.fit_gaussian_to_histogram( initial_guess=[125, 300, 0.25])

    #Ploteo de histogramas y fiteos
    obj_gaussian= Fitting()
    elevation = hist_filt[50]
    plt.figure(figsize=(8, 6))
    plt.plot(bins, hist_filt, label='Histogram', color='blue')
    plt.plot(bins[0:70], obj_gaussian.gaussian(bins[0:70], *gaussian_fit_1)+elevation, label='First Gaussian Fit', color='red')
    plt.plot(bins[70:250], obj_gaussian.gaussian(bins[70:250], *gaussian_fit_2), label='Second Gaussian Fit', color='green',)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title ('ZTE Fit to Histogram')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f'first gaussian={np.round(gaussian_fit_1, 3)}\nsecond gaussian={np.round(gaussian_fit_2, 3)}') 

    suma_gauss= Fitting.suma_gaussianas(x=bins, gaussianas=[gaussian_fit_1, gaussian_fit_2], elevation=elevation)

    #Ploteo de la suma de gaussians
    y_step=100
    plt.figure(figsize=(8, 6))
    plt.plot(bins, suma_gauss, label='Sum of Gaussians', color='purple')
    plt.yticks(np.arange(0, np.max(suma_gauss)+y_step,y_step))
    plt.grid()
    plt.show()
    """
    #Threshold segmentation
    segmentator= Operations(volumetric_array_1 = vol_filt)
    seg_vol= segmentator.threshold_segmentation(lower_threshold=100, upper_threshold=125)
    sitk.WriteImage(sitk.GetImageFromArray(seg_vol),"c:\codigos_tesis\seg_vol.nrrd")
    """
    #Realzado de bordes
    folder_n4_filtered="c:/codigos_tesis/n4_corrected.nrrd"
    filter_border = Filters(sequence_directory=folder_n4_filtered)
    sitk_image=sitk.ReadImage(folder_n4_filtered)
    image_array=sitk.GetArrayFromImage(sitk_image)
    vol_sobel= filter_border.sobel_image_filter(img_arr=image_array)
    sitk.WriteImage(sitk.GetImageFromArray(vol_sobel),"c:/codigos_tesis/sobel_tresholded.nrrd")
    
if __name__ == "__main__":
    main()