import sys
sys.path.append("c:/codigos_tesis/Repositorio/LacidMed")

import pydicom
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.optimize import curve_fit

from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.processing.filters import Filters
from lacid_med.src.visualization.histograms import HistogramGenerator
from lacid_med.src.processing.operations import Operations

# Definir la función de tres gaussianas
def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu) ** 2 / (2*sigma ** 2))

def triple_gaussian(x, amp1, mu1, sigma1, amp2, mu2, sigma2, amp3, mu3, sigma3, amp4, mu4, sigma4, amp5, mu5, sigma5):
    return (gaussian(x, amp1, mu1, sigma1) +
            gaussian(x, amp2, mu2, sigma2) +
            gaussian(x, amp3, mu3, sigma3) +
            gaussian(x, amp4, mu4, sigma4) +
            gaussian(x, amp5, mu5, sigma5))

def main():
    OFFSET=37
    sequence_dir = "c:/codigos_volumetria/2023-TFE-218"

    loader = DicomLoaderMRI(directory_path=sequence_dir)
    org_vol = loader.volumetric_array
    print(org_vol.shape)

    # Poner path de volumen previamente filtrado por N4. NO SE APLICA EL FILTRO EN ESTE SCRIPT 
    image = sitk.ReadImage("C:/codigos_volumetria/n4_corrected.nrrd")
    vol_filt = sitk.GetArrayFromImage(image)
    
    vol_filt_stack = np.dstack(vol_filt)
    sitk.WriteImage(sitk.GetImageFromArray(vol_filt_stack),"C:/codigos_volumetria/no_norm_filt.nrrd")

    # Normalización
    #escalador_1 = Operations(volumetric_array_1=org_vol)   
    escalador_2 = Operations(volumetric_array_1=vol_filt_stack)

    #norm_vol_raw = escalador_1.scale_matrix_to_value(value=255)
    norm_vol_filt_stack = escalador_2.scale_matrix_to_value(value=255)

    sitk.WriteImage(sitk.GetImageFromArray(norm_vol_filt_stack),"C:/codigos_volumetria/filtrada_norm.nrrd")

    # Histograma de la imagen normalizada filtrada
    histogram3D = HistogramGenerator(array_3D=norm_vol_filt_stack)
    hist_filt, bins = histogram3D.create_mean_histogram_of_3d_array(offset=OFFSET, show=True, plot_title="Histogram of filtered image normalized", xlabel="Pixel value", ylabel="Frequency")

    # Ajuste del histograma con tres gaussianas
    initial_guesses = [12, 50, 5, 10, 70, 10, 15, 110, 10, 30, 150, 0.5, 40, 220, 17]  # Valores iniciales para las 3 gaussianas
    params, covariance = curve_fit(triple_gaussian, bins, hist_filt, p0=initial_guesses)

    # Parámetros ajustados
    amp1, mu1, sigma1, amp2, mu2, sigma2, amp3, mu3, sigma3, amp4, mu4, sigma4, amp5, mu5, sigma5 = params
    print(f"Primera gaussiana: Amplitud = {amp1}, Media = {mu1}, Sigma = {sigma1}")
    print(f"Segunda gaussiana: Amplitud = {amp2}, Media = {mu2}, Sigma = {sigma2}")
    print(f"Tercera gaussiana: Amplitud = {amp3}, Media = {mu3}, Sigma = {sigma3}")
    print(f"Cuarta gaussiana: Amplitud = {amp4}, Media = {mu4}, Sigma = {sigma4}")
    print(f"Quinta gaussiana: Amplitud = {amp5}, Media = {mu5}, Sigma = {sigma5}")

    # Ploteo del histograma con el ajuste de tres gaussianas
    x_fit = np.linspace(min(bins), max(bins), 1000)
    y_fit = triple_gaussian(x_fit, *params)
    plt.figure(figsize=(8, 6))
    plt.plot(bins, hist_filt, label='Histogram', color='black')
    plt.plot(x_fit, y_fit, label='Ajuste con tres gaussianas', color='red')
    
    # Graficar las gaussianas individuales
    plt.plot(x_fit, gaussian(x_fit, amp1, mu1, sigma1), 'b--', label='Gaussiana 1')
    plt.plot(x_fit, gaussian(x_fit, amp2, mu2, sigma2), 'y--', label='Gaussiana 2')
    plt.plot(x_fit, gaussian(x_fit, amp3, mu3, sigma3), 'm--', label='Gaussiana 3')
    plt.plot(x_fit, gaussian(x_fit, amp4, mu4, sigma4), 'g--', label='Gaussiana 4')
    plt.plot(x_fit, gaussian(x_fit, amp5, mu5, sigma5), 'r--', label='Gaussiana 5')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Ajuste del histograma con gaussianas')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Graficar el histograma con el ajuste de tres gaussianas
    plt.figure(figsize=(8, 6))
    plt.plot(bins, hist_filt, label='Histogram', color='black')
    plt.plot(x_fit, y_fit, label='Ajuste con tres gaussianas', color='red')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Ajuste del histograma con gaussianas')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()