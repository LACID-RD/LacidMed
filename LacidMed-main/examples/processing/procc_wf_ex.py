# Author: Rodrigo N. Alcalá M. rodrigo.alcala@skiff.com

"""
In this file you will find a general example of the filters workflow.
Note: This example will not necesarily compile, and it should be used as an
example

run on CLI in the project dir as python -m examples.processing.procc_wf_ex
to prevent relative import issues
"""


import sys

sys.path.append("/home/ralcala/Documents/FUESMEN/LacidMed/lacid_med")

import pydicom
import matplotlib.pyplot as plt
import numpy as np

from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.processing.filters import Filters

def main():
    
    sequence_dir = "/home/ralcala/Documents/FUESMEN/LacidMed/data/20135603/20211018/rmac_prostata_multiparametrica/ax_t2_frfse_prost"

    loader = DicomLoaderMRI(directory_path=sequence_dir)
    sorted_files = loader.sorted_files
    dcm_files = loader.dcm_files
    
    filter = Filters(sequence_directory=sequence_dir, dicom_files=dcm_files, output_dir="/output")
    n4_files = filter.N4_bias_correction_filter()
        
    print(n4_files)
    
    return

if __name__ == "__main__":
    main()