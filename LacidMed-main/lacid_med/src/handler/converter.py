import os
import pydicom
import numpy as np
from typing import List
import re 
from PIL import Image
import os
from skimage.color import rgb2gray
import SimpleITK as sitk
import imageio
import nrrd


class Converter:
    def __init__(
            self, 
            input_file: str = None,
            input_directory: str = None, 
            output_directory: str = None,
            ) -> None:
        """
        Initializes the Converter instance with the given directory paths.
        Args:
            input_file (str): The path to the DICOM file to be converted.
            input_directory (str): The directory path for the images files to be converted.
            output_directory (str): The directory path for the output converted images files.
        Raises:
            ValueError: If the input_file path is invalid.
            ValueError: If the directory paths are invalid.
            ValueError: If the input_file and input_directory are both None.
        """
        if not input_file and not input_directory:
            raise ValueError("Please provide either input_file or input_directory")
        if input_directory is not None:
            if not os.path.isdir(input_directory):
                raise ValueError("Invalid directory old_directory path")
        if input_file is not None:
            if not os.path.isfile(input_file):
                raise ValueError("Invalid file path")
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
            print("Output directory created in path.")
        self.input_file = input_file
        self.input_directory = input_directory
        self.output_directory = output_directory

    def numericalSort(self, value):
        """
        Helper function to sort the files in numerical order.
        Args:
            value (str): The name of the DICOM file.
        """
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def dicom_2_jpg(self, output_name_custom: str = None) -> None: 
        """
        Converts DICOM files in the input directory to JPEG images and saves them in the output directory.
        Args:
            output_name_custom (str, optional): The name of the output JPEG file. Defaults to None.
        """
        ds = pydicom.dcmread(self.input_file)
        pixel_array = ds.pixel_array
        pixel_array_normalized = (pixel_array / pixel_array.max()) * 255
        pixel_array_uint8 = pixel_array_normalized.astype('uint8')
        image = Image.fromarray(pixel_array_uint8)
        if output_name_custom is not None:
            output_name = output_name_custom + ".JPEG"
            output_path = os.path.join(self.output_directory, output_name)
            image.save(output_path, format='JPEG')
        else:
            output_path = os.path.join(self.output_directory, "Output_image.JPEG")
            image.save(self.output_directory, format='JPEG') 

    def dicom_directory_2_jpg(self, output_name_custom: str = None) -> None:
        """
        Converts DICOM files in the input directory to JPEG images and saves them in the output directory.
        Args:
            output_name_custom (str, optional): The name of the output JPEG file. Defaults to None.
        """
        i = 1
        for file in sorted(os.listdir(self.input_directory), key=self.numericalSort):
            if file.endswith(".dcm"):
                ds = pydicom.dcmread(os.path.join(self.input_directory, file))
                pixel_array = ds.pixel_array
                pixel_array_normalized = (pixel_array / pixel_array.max()) * 255
                pixel_array_uint8 = pixel_array_normalized.astype('uint8')
                image = Image.fromarray(pixel_array_uint8)
                if output_name_custom is not None:
                    output_name = output_name_custom + "_" + str(i) + ".JPEG"
                    output_path = os.path.join(self.output_directory, output_name)
                    image.save(output_path, format='JPEG')
                else:
                    output_name = str(i) + ".JPEG"
                    output_path = os.path.join(self.output_directory, output_name)
                    image.save(output_path, format='JPEG')
                i += 1
    
    def jpeg_2_nrrd(self, output_name_custom: str = None) -> None:
        """
        Convert a JPEG image to NRRD format.
        """
        image = imageio.imread(self.input_file)
        if output_name_custom is not None:     
            output_name = output_name_custom + ".nrrd"       
            output_path = os.path.join(self.output_directory, output_name_custom)
            nrrd.write(output_path, image)    
        else:
            output_path = os.path.join(self.output_directory, "Output_image.nrrd")
            nrrd.write(output_path, image)

    def jpeg_directory_2_nrrd(self, output_name_custom: str = None) -> None:
        """
        Convert a multiples JPEGs from a directory to a single NRRD file.
        Args:
            output_name_custom (str, optional): The name of the output NRRD file.
        """
        images = []
        for file in sorted(os.listdir(self.input_directory), key=self.numericalSort):
            if file.endswith(".JPEG") or file.endswith(".jpeg") or file.endswith(".jpg"):
                file_path = os.path.join(self.input_directory, file)
                image = imageio.imread(file_path)
                if image.ndim == 3:
                    image = rgb2gray(image)
                images.append(image)        
        image_stack = np.stack(images, axis=-1)
        print("The resulting nrrd file is a volume of dimensions:" + str(image_stack.shape)) 
        if output_name_custom is not None:     
            output_name = output_name_custom + ".nrrd"       
            output_path = os.path.join(self.output_directory, output_name)
            nrrd.write(output_path, image_stack)    
        else:
            output_path = os.path.join(self.output_directory, "Output_volume.nrrd")
            nrrd.write(output_path, image_stack)
