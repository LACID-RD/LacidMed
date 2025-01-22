import sys
sys.path.append("C:/Users/santi/Desktop/Fuesmen/code/LacidMed")

import pydicom
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.processing.filters import Filters
from lacid_med.src.visualization.plotter import DicomPlotter
from lacid_med.src.visualization.histograms import HistogramGenerator
from lacid_med.src.processing.fitting import Fitting
from lacid_med.src.processing.operations import Operations


def main():
    OFFSET=3
    sequence_dir = "C:/Users/santi/Desktop/prueba_input"
    loader = DicomLoaderMRI(directory_path=sequence_dir) #se instancia la clase Loader
    org_vol = loader.volumetric_array #raw data organizada
    
    #N4 filter
    filter = Filters(sequence_directory=sequence_dir)
    n4_files = filter.N4_bias_correction_filter(max_iterations=[400,400,400], convergence_threshold=0.001,mask_image=None)
    sitk.WriteImage(sitk.GetImageFromArray(n4_files), "C:/Users/santi/Desktop/prueba/prueba_n4.nrrd")
    
if __name__ == "__main__":
    main()