import os
import re

def numericalSort(value):
    """
    Helper function to sort the DICOM files in numerical order.
    Args:
        value (str): The name of the DICOM file.
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def sort(input_path):
    directory = input_path
    directory_files = []
    for file in sorted(os.listdir(directory), key=numericalSort):    
        path = os.path.join(directory, file)
        if file.endswith(".dcm"):                
            directory_files.append(path)
        else:
            print("This file is not a DICOM file: " + path)
            
    return directory_files