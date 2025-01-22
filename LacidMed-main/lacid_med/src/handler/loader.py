import logging
import os
import pathlib
import numpy as np
from typing import List
import pydicom


class DicomLoaderMRI:
    """
    A class for loading and processing DICOM files from a given directory path.

    Attributes:
        directory_path (str): The directory path where the DICOM files are located.
        _sorted_files (List[str]): A private field to store the sorted file paths of the DICOM files.
        _volumetric_array (np.ndarray): A private field to store the volumetric array of the DICOM files.
    """

    def __init__(self, directory_path: str) -> None:
        """
        Initializes the DicomLoaderMRI instance with the given directory path.

        Args:
            directory_path (str): The directory path where the DICOM files are located.

        Raises:
            ValueError: If the directory path is invalid.
        """
        self.validate_directory(directory_path)
        self.directory_path = directory_path
        self._sorted_files = None
        self.volumetric_array = self.generate_volumetric_array()
        self.dcm_files = []

    def validate_directory(self, directory_path: str) -> None:
        """
        Validates the directory path to ensure it is a valid directory.

        Args:
            directory_path (str): The directory path to validate.

        Raises:
            ValueError: If the directory path is invalid.
        """
        if not os.path.isdir(directory_path):
            raise ValueError("Invalid directory path")

    def dicom_loader(self, sorted_files: List[str]) -> List[pydicom.Dataset]:
        """
        Load DICOM files from a given list of sorted file paths.

        Args:
            sorted_files (List[str]): A list of sorted file paths of DICOM files.

        Returns:
            List[pydicom.Dataset]: A list of DICOM files read from the sorted file paths.
        """
        dicom_files = [
            pydicom.dcmread(file) for file in sorted_files if os.path.isfile(file)
        ]
        return dicom_files

    @property
    def sorted_files(self) -> List[str]:
        """
        Retrieves the sorted file paths of the DICOM files.

        Returns:
            List[str]: A list of sorted file paths of the DICOM files.
        """
        if self._sorted_files is None:
            files = self._get_files()
            instance_numbers = {}
            for file in files:
                instance_number = pydicom.dcmread(
                    file, stop_before_pixels=True
                ).InstanceNumber
                instance_numbers[file] = instance_number
            files.sort(key=lambda file: instance_numbers[file])
            self._sorted_files = files
        return self._sorted_files

    def _get_files(self) -> List[str]:
        """
        This method returns a sorted list of the paths of each file in the directory

        Returns:
            List[str]: a sorted list of file paths
        """
        dicom_files = []
        try:
            for file in sorted(pathlib.Path(self.directory_path).rglob("*.dcm")):
                try:
                    dicom_files.append(str(file))
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"Error reading file: {file}. {str(e)}")
                except PermissionError as e:
                    raise PermissionError(f"Error reading file: {file}. {str(e)}")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Error accessing directory: {self.directory_path}. {str(e)}"
            )
        except PermissionError as e:
            raise PermissionError(
                f"Error accessing directory: {self.directory_path}. {str(e)}"
            )

        return dicom_files

    def generate_volumetric_array(self) -> np.ndarray:
        """
        Generate a 3D stacked array of the volume.

        Returns:
            np.ndarray: 3D numpy array representing the volume.
        """
        logging.info("Generating volumetric array...")
        dcm_matrices = [pydicom.dcmread(path).pixel_array for path in self.sorted_files]
        stacked_volume = np.dstack(dcm_matrices)
        return stacked_volume

    def load_files(self):
        """
        Load DICOM files from the sorted file paths.
        """
        self.dcm_files = self.dicom_loader(self.sorted_files)
