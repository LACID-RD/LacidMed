from pathlib import Path
import os
import pydicom


class DicomReorganizer:
    def __init__(self, study_directory: str) -> None:
        """
        Initialize the class instance.

        Args:
            study_directory (str): The directory path for the study.
        """
        if not os.path.isdir(study_directory):
            raise ValueError("Invalid study directory path.")

        self.study_directory = study_directory

    def _clean_text(


        self,
        string,
        forbidden_symbols=["*", ".", ",", '"', "\\", "/", "|", "[", "]", ":", ";", " "],
    ):
        """
        Clean and standardize text descriptions.

        Parameters:
        - string: a string representing the text to be cleaned.
        - forbidden_symbols: a list of symbols to be replaced with an underscore.

        Returns:
        - a string with cleaned and standardized text.

        Raises:
        - None
        """
        if string is None or not isinstance(string, str):
            return ""

        string = string.lower()  # convert the string to lowercase
        translation_table = str.maketrans(
            "".join(forbidden_symbols), "_" * len(forbidden_symbols)
        )
        return string.translate(translation_table)

    def reorganize(self):
        """
        Organizes DICOM files into a nested folder structure based on patient, study, and series information.
        Cleans and standardizes text descriptions and generates a new standardized file name for each DICOM file.

        Args:
            src (str): The source directory containing the DICOM files to be organized.

        Returns:
            None. The method organizes and saves the DICOM files into a nested folder structure based on patient, study, and series information.
        """
        dst = self.study_directory

        print("reading file list...")
        unsortedList = [dicom_path for dicom_path in Path(dst).rglob("*.dcm")]

        print(f"{len(unsortedList)} files found.")

        for dicom_loc in unsortedList:
            # read the file
            ds = pydicom.read_file(str(dicom_loc), force=True)

            # get patient, study, and series information
            patientID = self._clean_text(ds.get("PatientID", "NA"))
            studyDate = self._clean_text(ds.get("StudyDate", "NA"))
            studyDescription = self._clean_text(ds.get("StudyDescription", "NA"))
            seriesDescription = self._clean_text(ds.get("SeriesDescription", "NA"))

            # generate new, standardized file name
            modality = ds.Modality
            studyInstanceUID = ds.StudyInstanceUID
            seriesInstanceUID = ds.SeriesInstanceUID
            instanceNumber = str(ds.InstanceNumber)
            fileName = f"{modality}.{seriesInstanceUID}.{instanceNumber}.dcm"

            # uncompress files (using the gdcm package)

            if not ds.is_decompressed and hasattr(ds, "PixelData"):
                try:
                    ds.decompress()
                except Exception as e:
                    raise Exception(
                        f'an instance in file {patientID} - {studyDate} - {studyDescription} - {seriesDescription}" could not be decompressed. Exiting. Error: {str(e)}'
                    )

            # save files to a 4-tier nested folder structure
            patient_path = os.path.join(dst, patientID)
            study_path = os.path.join(patient_path, studyDate)
            study_desc_path = os.path.join(study_path, studyDescription)
            series_desc_path = os.path.join(study_desc_path, seriesDescription)

            os.makedirs(patient_path, exist_ok=True)
            os.makedirs(study_path, exist_ok=True)
            os.makedirs(study_desc_path, exist_ok=True)
            os.makedirs(series_desc_path, exist_ok=True)

            print(
                f"Saving out file: {patientID} - {studyDate} - {studyDescription} - {seriesDescription}."
            )

            ds.save_as(os.path.join(series_desc_path, fileName))
        print("done.")
