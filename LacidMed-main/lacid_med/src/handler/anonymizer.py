from pathlib import Path
import uuid
import os
import pydicom
from typing import List


class Anonymizer:
    def __init__(self, dicom_files: List[str], output_dir: str) -> None:
        """
        Initializes the Anonymizer object with the provided DICOM file paths and output directory.

        Args:
            dicom_files (List[str]): A list of DICOM file paths to be anonymized.
            output_dir (str): The directory where the anonymized DICOM files will be saved.

        Raises:
            ValueError: If the output directory does not exist or is not writable.
        """
        self.dicom_files = dicom_files
        self.output_dir = output_dir
        if not os.path.exists(output_dir) or not os.access(output_dir, os.W_OK):
            raise ValueError("Output directory does not exist or is not writable")

    def anonymize(self) -> None:
        """
        Calls the _perform_anonymization method to perform the anonymization process for each DICOM file in the input list.

        Raises:
            FileNotFoundError: If a DICOM file is not found.
            PermissionError: If there is a permission issue accessing the DICOM files.
        """
        try:
            self._perform_anonymization()
        except (FileNotFoundError, PermissionError) as e:
            raise e

    def _perform_anonymization(self) -> None:
        sequence_dirs = {}  # Dictionary to store mapping between imaging sequence and directory path

        # Check if the output directory exists and create it if it doesn't
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for dicom_file in self.dicom_files:
            file_path = Path(dicom_file)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            try:
                ds = pydicom.dcmread(file_path)
            except Exception as e:
                print(f"Error reading DICOM file: {file_path}")
                continue

            # Use a more reliable method to get the imaging sequence
            sequence = ds.get("SequenceName", "UnknownSequence")

            if sequence not in sequence_dirs:
                # Create a new directory for the imaging sequence
                #sequence_dir = self.output_dir / sequence
                sequence_dir = os.path.join(self.output_dir, sequence)
                os.makedirs(sequence_dir, exist_ok=True)
                sequence_dirs[sequence] = sequence_dir

            study_id = str(uuid.uuid4())
            ds.PatientID = study_id

            # Remove private tags and other sensitive information
            ds.remove_private_tags()
            ds.PatientName = ""
            ds.PatientBirthDate = ""
            ds.PatientSex = ""

            output_file = os.path.join(sequence_dirs[sequence], f"anonymized_{file_path.name}")
            ds.save_as(output_file)
