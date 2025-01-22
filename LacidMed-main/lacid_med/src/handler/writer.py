import multiprocessing
import os
import pydicom
import numpy as np
from typing import List
import pathlib
import re


class SingleFileWriter:
    def __init__(self, dicom_file_path: str) -> None:
        """
        Initializes the SingleFileWriter instance with the given DICOM file path.
        Args:
            dicom_file_path (str): The path of the DICOM file to be modified.
        Raises:
            ValueError: If the file path is invalid or the file does not exist.
        """
        try:
            self.dicom_file = pydicom.dcmread(dicom_file_path)
        except (FileNotFoundError, pydicom.errors.InvalidDicomError):
            raise ValueError("Invalid DICOM file or file does not exist.")

    def write(self, new_pixel_array: np.ndarray, output_path: str) -> None:
        """
        Write a new pixel array to a DICOM file at the given output path.
        Intended for keeping DICOM tags in processed images that are anonymized.
        Args:
            new_pixel_array (np.ndarray): The new pixel array to be written.
            output_path (str): The path of the DICOM file to be written.
        """
        if not isinstance(new_pixel_array, np.ndarray):
            raise TypeError("new_pixel_array must be a numpy ndarray")
        if new_pixel_array.ndim != 2:
            raise ValueError("new_pixel_array must have exactly two dimensions")
        ds = self.dicom_file
        ds.PixelData = new_pixel_array.tobytes()
        ds.Rows, ds.Columns = new_pixel_array.shape
        try:
            ds.save_as(output_path)
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error writing file: {str(e)}")

class MultipleFileWriter:
    def __init__(self, dicom_files: List[str]) -> None:
        """
        Initializes the MultipleFileWriter instance with the given list of DICOM file paths.
        Args:
            dicom_files (List[str]): A list of DICOM file paths to be modified.
        """
        self.dicom_files = dicom_files

    def write(self, volumetric_array: np.ndarray, output_path: str) -> None:
        """
        Writes a new pixel array to a DICOM file at the given output path for multiple files sequentially.
        Args:
            volumetric_array (np.ndarray): The new pixel array to be written.
            output_path (str): The path of the DICOM file to be written.
        """
        i = 0
        for dicom_file in self.dicom_files:
            try:
                ds = pydicom.dcmread(dicom_file)
                new_pixel_array = volumetric_array[:, :, i]
                i += 1
                if new_pixel_array.shape == (ds.Rows, ds.Columns):
                    ds.PixelData = new_pixel_array.tobytes()
                    ds.Rows, ds.Columns = new_pixel_array.shape
                    unique_output_path = (
                        output_path + "/" + os.path.basename(dicom_file)
                    )
                    ds.save_as(unique_output_path)
                else:
                    raise ValueError("Shape of new_pixel_array does not match original pixel data")
            except Exception as e:
                print(f"Error writing file {dicom_file}: {str(e)}")

class Deanonymizer:
    def __init__(
            self, 
            old_directory: str, 
            new_directory: str, 
            output_directory: str
            ) -> None:
        """
        Initializes the Deanonymizer instance with the given directory paths.
        Args:
            old_directory (str): The directory path for the original DICOM files to extract the tags from.
            new_directory (str): The directory path for the new DICOM files to write the extracted tags to.
            output_directory (str): The directory path for the output deanonymized DICOM files. If the directory does not exist, it will be created.
        Raises:
            ValueError: If the directory paths are invalid.
        """
        if not os.path.isdir(old_directory):
            raise ValueError("Invalid directory old_directory path")
        if not os.path.isdir(new_directory):
            raise ValueError("Invalid directory new_directory path")
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
            print("Output directory created in path.")
        self.old_directory = old_directory
        self.new_directory = new_directory
        self.output_directory = output_directory
        
    def numericalSort(self, value):
        """
        Helper function to sort the DICOM files in numerical order.
        Args:
            value (str): The name of the DICOM file.
        """
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def deanonymizer(self, output_name: str = None) -> None:
        """
        Write DICOM tags on anonymized images and write them to a new directory.
        Args:
            output_name (str, optional): The name of the output deanonymized DICOM file. Defaults to None.
        """
        old_files = []
        for file in sorted(os.listdir(self.old_directory), key=self.numericalSort):    
            old_path = os.path.join(self.old_directory, file)
            if file.endswith(".dcm"):                
                old_files.append(old_path)
            else:
                print("This file is not a DICOM file: " + old_path)
        new_files = []
        for file in sorted(os.listdir(self.new_directory), key=self.numericalSort):
            new_path = os.path.join(self.new_directory, file)
            if file.endswith(".dcm"):
                new_files.append(new_path)
            else:
                print("This file is not a DICOM file: " + new_path)
        if len(old_files) != len(new_files):
            raise ValueError("The number of files in the old and new directories must be the same")
        else:
            for i in range(0,  len(old_files)):    
                if output_name is not None:
                    output_path = self.output_directory + "/" + output_name + "_" + str(i + 1) + ".dcm"
                else:
                    output_path = self.output_directory + "/" + str(i + 1) + ".dcm"
                new_file = pydicom.dcmread(new_files[i])
                new_array = new_file.pixel_array
                writer = SingleFileWriter(old_files[i])
                writer.write(new_array, output_path)