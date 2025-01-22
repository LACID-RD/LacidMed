# Author: Rodrigo N. Alcalá M. rodrigo.alcala@skiff.com

"""
In this file you will find a general example of the visualization workflow.
Note: This example will not necesarily compile, and it should be used as an
example

run on CLI in the project dir as python -m examples.visualization.visual_ex
to prevent relative import issues
"""


import sys

sys.path.append("/home/ralcala/Documents/FUESMEN/LacidMed/lacid_med")

import pydicom
import matplotlib.pyplot as plt
import numpy as np

from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.visualization.plotter import DicomPlotter
from lacid_med.src.visualization.histograms import HistogramGenerator


def main():

    sequence_dir = "/home/ralcala/Documents/FUESMEN/LacidMed/data/20135603/20211018/rmac_prostata_multiparametrica/ax_t2_frfse_prost"

    loader = DicomLoaderMRI(directory_path=sequence_dir)
    sorted_files = loader.sorted_files

    plotter = DicomPlotter(dicom_files=sorted_files)
    # plotter.plot_single_file(img_num_input=0)
    # plotter.plot_all_files()

    dcm_files = loader.dcm_files
    vol = loader.volumetric_array
    print(np.shape(vol))

    histogram3D = HistogramGenerator(array_3D=vol)
    hist, bins = histogram3D.create_histogram_of_3d_array(offset=1, show=True)

    return 0


if __name__ == "__main__":
    main()
