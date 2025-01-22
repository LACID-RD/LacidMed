# Author: Rodrigo N. Alcalá M. rodrigoalcala@protonmail.com

"""
In this file you will find a general example of the filters workflow.
Note: This example will not necesarily compile, and it should be used as an
example

run on CLI in the project dir as python -m examples.processing.general_filters_ex
to prevent relative import issues
"""


import sys

sys.path.append("/home/ralcala/Documents/FUESMEN/LacidMed/lacid_med")

import pydicom
import matplotlib.pyplot as plt
import numpy as np

from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.processing.filters import Filters

from lacid_med.src.visualization.plotter import DicomPlotter


def main():
    img_path = "data/20135603/20211018/rmac_prostata_multiparametrica/ax_t2_frfse_prost/MR.1.2.840.113619.2.363.10499743.3637646.19278.1634553977.815.16.dcm"
    img_arr = np.array(pydicom.dcmread(img_path).pixel_array)
    plotter = DicomPlotter()
    #plotter.plot_single_file(img_arr)

    filter = Filters()

    #normalized_img = filter.normalize_image_filter(img_arr)
    #plotter.plot_single_file(normalized_img)

    #gaussian_img = filter.gaussian_image_filter(img_arr, sigma=3.0)
    #plotter.plot_single_file(gaussian_img)

    #median_img = filter.median_image_filter(img_arr)
    #plotter.plot_single_file(median_img)

    binary_img = filter.binary_threshold_image_filter(
        img_arr,
        lower_threshold=1200,
        upper_threshold=1300,
        inside_value=100,
        outside_value=0,
    )
    #plotter.plot_single_file(binary_img)

    sobel_img = filter.sobel_image_filter(img_arr)
    #plotter.plot_single_file(sobel_img)
    
    laplacian_img = filter.laplacian_image_filter(img_arr)
    #plotter.plot_single_file(laplacian_img)
    
    fourier_img = filter.fourier_transform_filter(img_arr)
    #plotter.plot_single_file(fourier_img)
    
    inv_fourier_img = filter.inverse_fourier_trasform_filter(fourier_img)
    plotter.plot_single_file(inv_fourier_img)
    
    return


if __name__ == "__main__":
    main()