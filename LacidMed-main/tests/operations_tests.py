import pydicom
import matplotlib.pyplot as plt
import numpy as np

# lacid_med directory append.
import sys
sys.path.append("C:/Users/santi/Desktop/Fuesmen/code/LacidMed")

from lacid_med.src.processing.operations import Operations
from lacid_med.src.visualization.plotter import DicomPlotter
from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.handler.writer import MultipleFileWriter

# np.set_printoptions(threshold=sys.maxsize)

def main():

    # test using two paths pointing to one image each.
    path1 = "C:/Users/santi/Desktop/Fuesmen/imagenes/ZTE_H01_descomp/MR.1.2.840.113619.2.363.10499743.3637646.25026.1684513445.974.90.dcm"
    image_1 = pydicom.dcmread(path1)
    image_1_array = image_1.pixel_array
    path2 = "C:/Users/santi/Desktop/Fuesmen/imagenes/correction_N4_original/1.2.840.113619.2.363.10499743.3637646.25026.1684513445.974.90.dcm"
    image_2 = pydicom.dcmread(path2)
    image_2_array = image_2.pixel_array
    paths = [path1, path2]
    
    operations_1 = Operations(image_1_array, image_2_array)
    diff = operations_1.image_difference(image_2_array, image_1_array, clipping=True)
    negative_image_1 = operations_1.negative_transform()

    
    plotter_1 = DicomPlotter(paths)
    plotter_1._show_image(image_1_array)
    plotter_1._show_image(image_2_array)
    plotter_1._show_image(diff)
    plotter_1._show_image(negative_image_1) 

    
    
    # test using two directories.
    directory_1 = "C:/Users/santi/Desktop/Fuesmen/imagenes/ZTE_H01_descomp"
    loader_1 = DicomLoaderMRI(directory_1)
    vol_array_1 = loader_1.generate_volumetric_array()

    directory_2 = "C:/Users/santi/Desktop/Fuesmen/imagenes/test_volume"
    loader_2 = DicomLoaderMRI(directory_2)
    vol_array_2 = loader_2.generate_volumetric_array()
    
    plotter_2 = DicomPlotter(directory_1)

    operations_2 = Operations(vol_array_2, vol_array_1)
    
    volume_diff = operations_2.volume_difference(image_number=89, clipping=True)
    
    plotter_2._show_image(volume_diff)
    


if __name__ == "__main__":
    main()