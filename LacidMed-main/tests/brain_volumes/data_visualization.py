import sys
sys.path.append("c:/codigos_tesis/Repositorio/LacidMed") #le indica al script de python donde se encuentra el paquete de clases que se utilizarán

import pydicom
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.optimize import curve_fit

#import cython
#Se importan clases específicas de módulos dentro del paquete lacid_med
from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.processing.filters import Filters
from lacid_med.src.visualization.plotter import DicomPlotter
from lacid_med.src.visualization.histograms import HistogramGenerator
from lacid_med.src.processing.fitting import Fitting
from lacid_med.src.processing.operations import Operations
from lacid_med.src.processing.segmentation import Segmentation

def main():
    OFFSET=40 #es el mejor offset encontrado para eliminar pico de ruido
    sequence_dir = "c:/codigos_volumetria/ingenia_20_21_22/20_brain_ingenia"
    loader = DicomLoaderMRI(directory_path=sequence_dir) #se instancia la clase Loader
    org_vol = loader.volumetric_array #raw data organizada
    print(org_vol.shape)
    
    #N4 filter
    filter = Filters(sequence_directory=sequence_dir)
    n4_files = filter.N4_bias_correction_filter(max_iterations=[50,50,50], convergence_threshold=0.001,mask_image=None)
    sitk.WriteImage(sitk.GetImageFromArray(n4_files), "C:/codigos_volumetria/n4_corrected.nrrd")
    
    #Variable de imagenes filtradas 
    image = sitk.ReadImage("C:/codigos_volumetria/n4_corrected.nrrd") #imagenes filtradas
    vol_filt= sitk.GetArrayFromImage(image) #vol_fit es una matriz 3D de imagenes filtradas
    
    vol_filt_stack = np.dstack(vol_filt) #dstack apila los arrays
    sitk.WriteImage(sitk.GetImageFromArray(vol_filt_stack),"C:/codigos_volumetria/no_norm_filt.nrrd")

    # normalizado    
    escalador_1 = Operations(volumetric_array_1=org_vol)   
    escalador_2 = Operations(volumetric_array_1=vol_filt_stack)  

    norm_vol_raw = escalador_1.scale_matrix_to_value(value=255)
    norm_vol_filt = escalador_2.scale_matrix_to_value(value=255)

    sitk.WriteImage(sitk.GetImageFromArray(vol_filt),"C:/codigos_volumetria/filtrada_norm.nrrd")

    #Histogram from raw image
    histogram3D = HistogramGenerator(array_3D=org_vol)
    hist_org, bins = histogram3D.create_mean_histogram_of_3d_array(offset=OFFSET, show=True, plot_title="Histogram of raw image ", xlabel="Pixel value", ylabel="Frequency")

    #Histogram from image raw normalized
    histogram3D = HistogramGenerator(array_3D=norm_vol_raw)
    hist_raw, bins = histogram3D.create_mean_histogram_of_3d_array(offset=OFFSET, show=True, plot_title="Histogram of raw image normalized", xlabel="Pixel value", ylabel="Frequency")
    #print(bins)

    #Histogram from image norm filtered
    histogram3D = HistogramGenerator(array_3D=norm_vol_filt)
    hist_filt, bins = histogram3D.create_mean_histogram_of_3d_array(offset=OFFSET, show=True, plot_title="Histogram of filtered image normalized", xlabel="Pixel value", ylabel="Frequency")
    print(hist_filt)

    #Fiteo 1er gaussian
    fit_1= Fitting(x=bins[45:63], y=hist_filt[45:63])
    gaussian_fit_1= fit_1.fit_gaussian_to_histogram(initial_guess=[100, 120, 11])

    #Fiteo 2da gaussian
    fit_2= Fitting(x=bins[65:105], y=hist_filt[65:105])
    gaussian_fit_2= fit_2.fit_gaussian_to_histogram(initial_guess=[125, 100, 0.25])
   
    #Fiteo 3er gaussian
    fit_3= Fitting(x=bins[3:25], y=hist_filt[3:25])
    gaussian_fit_3= fit_3.fit_gaussian_to_histogram( initial_guess=[55, 40, 0.15])

    #Ploteo de histogramas y fiteos
    obj_gaussian= Fitting()
    elevation = 40
    #elevation = hist_filt[50]
    plt.figure(figsize=(8, 6))
    plt.plot(bins, hist_filt, label='Histogram', color='black')
    plt.plot(bins[30:85], obj_gaussian.gaussian(bins[30:85], *gaussian_fit_1)+elevation, label='First Gaussian Fit', color='red')
    plt.plot(bins[40:105], obj_gaussian.gaussian(bins[40:105], *gaussian_fit_2), label='Second Gaussian Fit', color='blue')
    plt.plot(bins[1:50], obj_gaussian.gaussian(bins[1:50], *gaussian_fit_3), label='Third Gaussian Fit', color='orange')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title ('Volume Fit to Histogram')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f'first gaussian={np.round(gaussian_fit_1, 3)}\nsecond gaussian={np.round(gaussian_fit_2, 3)}\nthird gaussian={np.round(gaussian_fit_3, 3)}')
    #print(f'first gaussian={np.round(gaussian_fit_1, 3)}\nsecond gaussian={np.round(gaussian_fit_2, 3)}')
    suma_gauss= Fitting.suma_gaussianas(x=bins, gaussianas=[gaussian_fit_1, gaussian_fit_2, gaussian_fit_3], elevation=elevation)
    
    #Ploteo de la suma de gaussians
    y_step=100
    plt.figure(figsize=(8, 6))
    plt.plot(bins, suma_gauss, label='Sum of Gaussians', color='purple')
    plt.yticks(np.arange(0, np.max(suma_gauss)+y_step,y_step))
    plt.grid()
    plt.show()
    
    #Threshold segmentation
    segmentator= Segmentation(volumetric_array = vol_filt)
    seg_vol= segmentator.threshold_segmentation(lower_threshold=100, upper_threshold=125)
    sitk.WriteImage(sitk.GetImageFromArray(seg_vol),"C:/codigos_volumetria/seg_vol.nrrd")
if __name__ == "__main__":
    main()

# Definir una función gaussiana
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# Definir la suma de tres gaussianas
def triple_gaussian(x, amp1, mu1, sigma1, amp2, mu2, sigma2, amp3, mu3, sigma3):
    return (gaussian(x, amp1, mu1, sigma1) +
            gaussian(x, amp2, mu2, sigma2) +
            gaussian(x, amp3, mu3, sigma3))

# Generar datos de ejemplo (puedes reemplazar esta parte con tus propios datos)
np.random.seed(0)
data1 = (0, 1, 500)
data2 = np.random.normal(5, 0.5, 300)
data3 = np.random.normal(10, 1.5, 200)
data = np.concatenate([data1, data2, data3])

# Crear el histograma
hist_filt, bins = np.histogram(data, bins=50, density=True)
x_hist = (bins[1:] + bins[:-1]) / 2  # Puntos medios de los bins

# Estimación inicial de los parámetros de las gaussianas
initial_guesses = [0.1, 0, 1, 0.1, 5, 0.5, 0.1, 10, 1.5]

# Ajustar el histograma con la suma de tres gaussianas
params, covariance = curve_fit(triple_gaussian, x_hist, hist_filt, p0=initial_guesses)

# Graficar el histograma y el ajuste
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.6, color='g', label='Histograma de datos')

# Graficar la función ajustada (suma de tres gaussianas)
x_fit = np.linspace(min(x_hist), max(x_hist), 1000)
y_fit = triple_gaussian(x_fit, *params)
plt.plot(x_fit, hist_filt, 'r-', label='Ajuste con tres gaussianas')

# Graficar cada una de las gaussianas por separado
y_gauss1 = gaussian(x_fit, params[0], params[1], params[2])
y_gauss2 = gaussian(x_fit, params[3], params[4], params[5])
y_gauss3 = gaussian(x_fit, params[6], params[7], params[8])

plt.plot(x_fit, y_gauss1, 'b--', label='Gaussiana 1')
plt.plot(x_fit, y_gauss2, 'y--', label='Gaussiana 2')
plt.plot(x_fit, y_gauss3, 'm--', label='Gaussiana 3')

# Etiquetas y leyenda
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.title('Ajuste de un histograma con tres gaussianas')
plt.legend()

plt.show()

# Imprimir los parámetros ajustados
print("Parámetros ajustados para las tres gaussianas:")
print(f"Gaussiana 1: Amplitud = {params[0]}, Media = {params[1]}, Sigma = {params[2]}")
print(f"Gaussiana 2: Amplitud = {params[3]}, Media = {params[4]}, Sigma = {params[5]}")
print(f"Gaussiana 3: Amplitud = {params[6]}, Media = {params[7]}, Sigma = {params[8]}")