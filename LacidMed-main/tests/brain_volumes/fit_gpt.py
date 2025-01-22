import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import ants
import nibabel as nib
import os
import cv2
import time
from lmfit import Model
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

import numpy as np
import matplotlib.pyplot as plt
import ants
from lmfit import Model

# Function with three Gaussian components and an offset
def three_gaussians(x, amp1, cen1, sigma1, amp2, cen2, sigma2, amp3, cen3, sigma3, offset):
    gauss1 = (amp1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-((x - cen1) ** 2) / (2 * sigma1 ** 2))
    gauss2 = (amp2 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-((x - cen2) ** 2) / (2 * sigma2 ** 2))
    gauss3 = (amp3 / (np.sqrt(2 * np.pi) * sigma3)) * np.exp(-((x - cen3) ** 2) / (2 * sigma3 ** 2))
    return gauss1 + gauss2 + gauss3 + offset, gauss1, gauss2, gauss3

def main():
    # Generate synthetic data (replace this with your actual data)
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(loc=3, scale=0.5, size=400),   # First Gaussian
        np.random.normal(loc=6, scale=0.8, size=300),   # Second Gaussian
        np.random.normal(loc=8, scale=0.3, size=300)    # Third Gaussian
    ])

    image_1_path = 'C:/Users/santi/Desktop/Volumetria/Slaibe_Yamila/21-TFE/nii/brain/21-TFE.nii'
    image_1 = ants.image_read(image_1_path)
    image_1_array = image_1.numpy()
    histogram_generator_1 = HistogramGenerator(array_3D=image_1_array)
    
    # Create histogram
    bin_counts, bin_edges = histogram_generator_1.create_histogram_of_3d_array(offset=1, show=True)

    # Midpoints of bins (for fitting purposes)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fix mismatch: Trim bin_counts to match bin_centers
    bin_counts = bin_counts[:len(bin_centers)]  # Trim bin_counts to match the length of bin_centers

    print(f'Length of bin_counts: {len(bin_counts)}')
    print(f'Length of bin_centers: {len(bin_centers)}')

    # Define the model for three Gaussians and an offset
    def model_func(x, amp1, cen1, sigma1, amp2, cen2, sigma2, amp3, cen3, sigma3, offset):
        total, gauss1, gauss2, gauss3 = three_gaussians(x, amp1, cen1, sigma1, amp2, cen2, sigma2, amp3, cen3, sigma3, offset)
        return total

    model = Model(model_func)

    # Provide initial guesses for parameters (optional but helps convergence)
    params = model.make_params(
        amp1=50, cen1=1000, sigma1=0.5,
        amp2=400, cen2=6000, sigma2=0.8,
        amp3=300, cen3=10000, sigma3=0.3,
        offset=0
    )

    # Fit the model to the histogram data
    result = model.fit(bin_counts, params, x=bin_centers)

    # Extract the best-fit parameters
    best_values = result.best_values
    _, gauss1, gauss2, gauss3 = three_gaussians(bin_centers, **best_values)

    # Plot the histogram, fit, and individual Gaussian components
    plt.bar(bin_centers, bin_counts, width=bin_edges[1] - bin_edges[0], label='Histogram data', alpha=0.6)
    plt.plot(bin_centers, result.best_fit, label='Fit: Three Gaussians + Offset', color='red', lw=2)
    
    # Plot individual Gaussians
    plt.plot(bin_centers, gauss1, label='Gaussian 1', linestyle='--', color='blue')
    plt.plot(bin_centers, gauss2, label='Gaussian 2', linestyle='--', color='green')
    plt.plot(bin_centers, gauss3, label='Gaussian 3', linestyle='--', color='orange')
    
    # Show the fitting results
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram and Three Gaussians + Offset Fit')
    plt.show()

    # Print fitting results
    print(result.fit_report())

if __name__ == "__main__":
    main()
