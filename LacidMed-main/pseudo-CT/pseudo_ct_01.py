
#codigo optimizado con chat GPT

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
sys.path.append("c:/codigos_tesis/Repositorio/LacidMed")

from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.processing.filters import Filters
from lacid_med.src.visualization.histograms import HistogramGenerator
from lacid_med.src.processing.operations import Operations
import SimpleITK as sitk

# Definimos la función Gaussiana
def gaussian(x, mean, amplitude, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def main():
    OFFSET = 3
    sequence_dir = "c:/Users/Gabriel/OneDrive - alumno.um.edu.ar/FUESMEN/ZTE_H01"

    # Carga de datos y filtrado
    loader = DicomLoaderMRI(directory_path=sequence_dir)
    org_vol = loader.volumetric_array

    filter = Filters(sequence_directory=sequence_dir)
    n4_files = filter.N4_bias_correction_filter(max_iterations=[50, 50, 50], convergence_threshold=0.001, mask_image=None)
    sitk.WriteImage(sitk.GetImageFromArray(n4_files), "c:/codigos_tesis/n4_corrected.nrrd")

    # Normalización
    vol_filt = sitk.GetArrayFromImage(sitk.ReadImage("c:/codigos_tesis/n4_corrected.nrrd"))
    escalador = Operations(volumetric_array_1=vol_filt)
    norm_vol_filt = escalador.scale_matrix_to_value(value=255)

    # Generación de histograma
    histogram3D = HistogramGenerator(array_3D=norm_vol_filt)
    hist_filt, bins = histogram3D.create_mean_histogram_of_3d_array(offset=OFFSET, show=True, plot_title="Filtered Histogram", xlabel="Pixel value", ylabel="Frequency")
    bins_centers = (bins[:-1] + bins[1:]) / 2

    # Ajuste de la primera Gaussiana
    initial_guess_1 = [12, 3000, 0.7]
    popt_1, _ = curve_fit(gaussian, bins_centers[0:70], hist_filt[0:70], p0=initial_guess_1)

    # Ajuste de la segunda Gaussiana
    initial_guess_2 = [125, 300, 0.25]
    popt_2, _ = curve_fit(gaussian, bins_centers[70:210], hist_filt[70:210], p0=initial_guess_2)

    # Elevación para la visualización
    elevation = hist_filt[50]

    # Visualización del histograma y los ajustes
    plt.figure(figsize=(8, 6))
    plt.plot(bins_centers, hist_filt, label='Histogram', color='blue')
    plt.plot(bins_centers[0:70], gaussian(bins_centers[0:70], *popt_1) + elevation,
             label='First Gaussian Fit', color='red')
    plt.plot(bins_centers[70:210], gaussian(bins_centers[70:210], *popt_2),
             label='Second Gaussian Fit', color='green')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram with Gaussian Fits')
    plt.grid(True)
    plt.legend()
    plt.show()

    print(f'First Gaussian parameters: {np.round(popt_1, 3)}')
    print(f'Second Gaussian parameters: {np.round(popt_2, 3)}')

    # Suma de Gaussianas
    suma_gauss = gaussian(bins_centers, *popt_1) + gaussian(bins_centers, *popt_2) + elevation

    # Ploteo de la suma de Gaussianas
    plt.figure(figsize=(8, 6))
    plt.plot(bins_centers, suma_gauss, label='Sum of Gaussians', color='purple')
    plt.grid()
    plt.title('Sum of Gaussian Fits')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
