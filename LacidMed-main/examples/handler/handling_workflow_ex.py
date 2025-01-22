# Author: Rodrigo N. Alcalá M. rodrigo.alcala@skiff.com

"""
In this file you will find a general example of the handling workflow.
The idea is to organize, decompress, load and anonimize DICOM files
Note: This example will not necesarily compile, and it should be used as an
example

run on CLI as python -m examples.handler.handling_workflow_ex to prevent
relative import issues
"""
import sys
sys.path.append("/home/ralcala/Documents/FUESMEN/LacidMed/lacid_med")

from lacid_med.src.handler.organizer import DicomReorganizer
from lacid_med.src.handler.loader import DicomLoaderMRI
from lacid_med.src.handler.anonymizer import Anonymizer
from lacid_med.src.handler.writer import MultipleFileWriter


def main():
    
    org_path = "/home/ralcala/Documents/FUESMEN/LacidMed/data" # path to the original DICOM files
    
    reorganizer = DicomReorganizer(study_directory=org_path)
    reorganizer.reorganize()
    
    decompressed_files = "/home/ralcala/Documents/FUESMEN/LacidMed/data/20135603"
    
    loader = DicomLoaderMRI(directory_path=decompressed_files)
    sorted_files = loader.sorted_files
    
    anomyzer = Anonymizer(dicom_files=sorted_files, output_dir="/home/ralcala/Documents/FUESMEN/LacidMed/data/anom")
    # anomyzer.anonymize()
    
    return 0

if __name__ == "__main__":
    main()  