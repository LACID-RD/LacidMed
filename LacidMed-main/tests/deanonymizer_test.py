import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import re
# lacid_med directory append.
import sys
sys.path.append("C:/Users/santi/Desktop/Fuesmen/code/LacidMed")

from lacid_med.src.handler.writer import SingleFileWriter
from lacid_med.src.handler.writer import Deanonymizer
from lacid_med.src.handler.loader import DicomLoaderMRI

    
def main():
    # deanonymizer(old_directory = "C:/Users/santi/Desktop/Fuesmen/imagenes/ZTE_H01_descomp", new_directory = "C:/Users/santi/Desktop/Fuesmen/imagenes/correction_N4_original", output_directory = "C:/Users/santi/Desktop/Fuesmen/imagenes/N4_volume_2", output_name = "test_name")
    deanonymizer = Deanonymizer("C:/Users/santi/Desktop/Fuesmen/imagenes/ZTE_H01_descomp", "C:/Users/santi/Desktop/Fuesmen/imagenes/correction_N4_original", "C:/Users/santi/Desktop/Fuesmen/imagenes/test_volume")
    deanonymizer.deanonymizer(output_name="test_name")


if __name__ == "__main__":
    main()
    
    