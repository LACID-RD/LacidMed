import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import ants
import nibabel as nib
import os
import os
from nilearn import image
from skimage import data  
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import label
from skimage import data, img_as_float
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.draw import circle_perimeter
from nipype.interfaces.dcm2nii import Dcm2niix

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

def convert_dcm2nii(input_path: str, output_path: str, compression: str = "y"):
    # a dcm2niix instalation is required to run this function. 
    # Remember to add the dcm2niix executable path as a system variable.
    converter = Dcm2niix()
    converter.inputs.source_dir = input_path  
    converter.inputs.output_dir = output_path  
    converter.inputs.compress = 'y' 
    converter.run()

def brains(input_directory):
    if not os.path.isdir(input_directory):
        raise ValueError("Invalid directory path")
    
    
def main(): 
    print("running")

if __name__ == "__main__":
    main()