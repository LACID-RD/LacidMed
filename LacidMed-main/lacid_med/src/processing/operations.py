import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom


class Operations:
    def __init__(
        self, 
        volumetric_array_1: np.ndarray = None,
        volumetric_array_2: np.ndarray = None, 
    ):
        """
        Initializes operations object.
        Args:
            volumetric_array_1 (np.ndarray): first volumetric array.
            volumetric_array_2 (np.ndarray): second volumetric array.
        Raises:
            TypeError: If volumetric_array_1 or volumetric_array_2 is not a numpy array.
        """
        if not isinstance(volumetric_array_1, np.ndarray) and volumetric_array_1 is not None:
            raise TypeError("volumetric_array_1 must be a numpy array")
        
        if not isinstance(volumetric_array_2, np.ndarray) and volumetric_array_2 is not None:
            raise TypeError("volumetric_array_2 must be a numpy array")
        
        self.volumetric_array_1 = volumetric_array_1
        self.volumetric_array_2 = volumetric_array_2
    

    def image_difference(self, pixel_array_1: np.ndarray, pixel_array_2: np.ndarray, clipping: bool = False):  
        """
        Substract two images and return the difference in pixel values. If clipping is set to True, negative values are set to 0.
        Args:
            pixel_array_1 (np.ndarray): pixel array of the first image.
            pixel_array_2 (np.ndarray): pixel array of the second image.
            clipping (bool, optional): Whether to use clipping to set negative values to 0.

        Returns:
            The difference in pixel values between the two images.
        """
        if pixel_array_1.shape != pixel_array_2.shape:
            raise ValueError("pixel arrays must have the same shape")
        else:
            pixel_array_diff = pixel_array_1 - pixel_array_2
            if clipping:
                pixel_array_diff[pixel_array_diff < 0] = 0
            else:
                pixel_array_diff = np.abs(pixel_array_diff)
        return pixel_array_diff
    
    def volume_difference(self, image_number: int = None, clipping: bool = False):
        """ 
        Substract the second volumetric array from the first volumetric array and return the difference in pixel values, slice per slice. 
        If clipping is set to True, negative values are set to 0.
        Args:
            image_number (int, optional): The number of the image to use in case a single image is needed.
            clipping (bool, optional): Whether to use clipping to set negative values to 0. Defaults to False.

        Returns:
            The difference in pixel values between the two images.
        """
        if self.volumetric_array_1.shape != self.volumetric_array_2.shape:
            raise ValueError("volumetric arrays must have the same shape")
        else:
            if image_number is None:
                vol_array_diff = self.volumetric_array_1 - self.volumetric_array_2
                if clipping:
                    vol_array_diff[vol_array_diff < 0] = 0
                else:
                    vol_array_diff = np.abs(vol_array_diff)
                return vol_array_diff
            elif image_number not in range(0, self.volumetric_array_1.shape[2]):
                raise ValueError("image_number must be in range 0 to " + str(self.volumetric_array_1.shape[2]))
            elif image_number is not None:
                pixel_array_1 = self.volumetric_array_1[:, :,image_number]
                pixel_array_2 = self.volumetric_array_2[:, :,image_number]
                pixel_array_diff = self.image_difference(pixel_array_1, pixel_array_2, clipping)
            return pixel_array_diff
        

    def negative_transform(self, pixel_array: np.array = None):
        """
        Generate a negative transformation of an image or volume. If no image is given, volumetric_array_1 is used.
        Args:
            pixel_array (np.array, optional): The image or volume array to transform. Defaults to volumetric_array_1.
        Returns:
            np.array: The transformed image or volume array.
        """
        if pixel_array is None:
            pixel_array = self.volumetric_array_1
        top_pixel = np.max(pixel_array)
        pixel_array_negative = np.multiply(pixel_array, -1)
        pixel_array_negative = np.add(pixel_array_negative, top_pixel)
        return pixel_array_negative

        
    def hounsfield_transform(self, C_h: int, E_h: int, E_u: int, MRI_array: np.ndarray = None, CT_array: np.ndarray = None):
        """
        Generate a Hounsfield transformation of an MRI image using a CT image.
        Args:
            C_h (int): The soft tissue - hard tissue interface in the CT image, given in housnfield units.
            E_h (int): The hard tissue - air interface in the MRI image.
            E_u (int): The hard tissue - soft tissue interface in the MRI image.
            MRI_array (np.ndarray, optional): The MRI image array. Defaults to volumetric_array_1.
            CT_array (np.ndarray, optional): The CT image array. Defaults to volumetric_array_2.
        Returns:
            np.array: The Hounsfield transformed MRI image.
        """
        if MRI_array is None:
            MRI_array = self.volumetric_array_1
        if CT_array is None:
            CT_array = self.volumetric_array_2
        C_u = np.max(CT_array)
        hounsfield_array = MRI_array.copy()
        hounsfield_array = np.add(np.multiply(((C_u - C_h) / (E_u - E_h)), np.subtract(hounsfield_array, E_h)), C_h)
        return hounsfield_array
    
    def scale_matrix_to_value(self, value: int = 255):
        """
        Scale a matrix to a specific value.
        Args:
            value (int, optional): The value to scale the matrix to. Defaults to 255.
        Returns:
            np.array: The scaled matrix.
        """
        # Step 1: Normalize the matrix to the range 0 to 1
        min_val = np.min(self.volumetric_array_1)
        max_val = np.max(self.volumetric_array_1)
        if min_val > 0:
            min_val = 0
        normalized_matrix = (self.volumetric_array_1.copy() - min_val) / (max_val - min_val)
        scaled_matrix = normalized_matrix * value
        # Step 3: Convert the scaled matrix to uint8
        scaled_matrix_uint8 = scaled_matrix.astype(np.uint8)
        rounded_matrix_uint8 = np.round(scaled_matrix_uint8)
        return rounded_matrix_uint8
        
        
