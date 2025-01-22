import numpy as np
import os
import pydicom
from typing import List
import SimpleITK as sitk
import scipy.fft
from scipy.ndimage import minimum_filter

class Filters:
    def __init__(
        self,
        sequence_directory: str = None,
        #output_dir: str = None,
    ):

        self.sequence_directory = sequence_directory
        #self.output_dir = output_dir

        self.resampler = sitk.ResampleImageFilter()
        

    def resample_image(
        self,
        itk_image,
        new_spacing: List[float] = [1.0, 1.0, 1.0],
        is_label: bool = False,
        interpolator: int = sitk.sitkBSpline,
    ) -> np.ndarray:
        """
        Resample the input image to a new spacing.

        Args:
            itk_image: The input image to be resampled.
            new_spacing: The new spacing to be applied.
            is_label: Whether the image is a label image or not.
            interpolator: The interpolator to be used for resampling.

        Returns:
            The resampled image.
        """
        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()

        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
        ]
        self.resampler.SetOutputSpacing(new_spacing)
        self.resampler.SetSize(new_size)
        self.resampler.SetOutputDirection(itk_image.GetDirection())
        self.resampler.SetOutputOrigin(itk_image.GetOrigin())
        self.resampler.SetTransform(sitk.Transform())
        self.resampler.SetDefaultPixelValue(itk_image.GetPixelIDValue())
        self.resampler.SetInterpolator(interpolator)

        return sitk.GetArrayFromImage(self.resampler.Execute(itk_image))

    def N4_bias_correction_filter(
        self,
        mask_image=None,
        convergence_threshold: float = 0.001,
        max_iterations: List[int] = [50, 50, 50],
    ) -> sitk.Image:
        """
        Apply N4 bias correction to the input image.

        Args:
            mask_image: The mask image to be used for bias correction.
            convergence_threshold: The convergence threshold for the bias correction algorithm.
            max_iterations: The maximum number of iterations for the bias correction algorithm.

        Returns:
            The bias-corrected image.
        """
        image = sitk.ImageSeriesReader()
        image.SetFileNames(image.GetGDCMSeriesFileNames(self.sequence_directory))
        image = image.Execute()
        image = sitk.Cast(image, sitk.sitkFloat32)
        image = self.resample_image(image)
        image = (
            sitk.GetImageFromArray(image)
            if not isinstance(image, sitk.Image)
            else image
        )

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(max_iterations)
        corrector.SetConvergenceThreshold(convergence_threshold)

        if mask_image is not None:
            mask = sitk.ReadImage(mask_image)
            n4_filtered_image = corrector.Execute(image, mask)
        else:
            mask = sitk.OtsuThreshold(image, 0, 1, 200)
            n4_filtered_image = corrector.Execute(image, mask)

        log_bias_field = corrector.GetLogBiasFieldAsImage(image)
        corrected_image_full_res = n4_filtered_image / sitk.Exp(log_bias_field)

        corrected_array = sitk.GetArrayFromImage(corrected_image_full_res)

        return corrected_array

    def normalize_image_filter(self, image_arr: np.ndarray) -> np.ndarray:
        """
        Normalize the input image.

        Args:
            image_arr: The input image as a numpy array.

        Returns:
            The normalized image as a SimpleITK image.
        """
        if not isinstance(image_arr, np.ndarray):
            raise ValueError("Invalid input: image_arr must be a numpy array.")
        image = sitk.Cast(sitk.GetImageFromArray(image_arr), sitk.sitkFloat32)
        image = sitk.RescaleIntensity(image, 0, 255)
        return sitk.GetArrayFromImage(image)

    def gaussian_image_filter(self, img_arr: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply a Gaussian filter to the input image.

        Args:
            img_arr: The input image as a numpy array.

        Returns:
            The filtered image as a SimpleITK image.
        """
        if not isinstance(img_arr, np.ndarray):
            raise ValueError("Invalid input: img_arr must be a numpy array.")
        image = sitk.Cast(sitk.GetImageFromArray(img_arr), sitk.sitkFloat32)
        image = sitk.SmoothingRecursiveGaussian(image, sigma=sigma)
        return sitk.GetArrayFromImage(image)

    def median_image_filter(self, img_arr: np.ndarray) -> np.ndarray:
        """
        Apply median filter to the input image.

        Args:
            img_arr: The input image as a numpy array. It should have shape (height, width) and data type uint8.

        Returns:
            The filtered image as a SimpleITK image.
        """
        if not isinstance(img_arr, np.ndarray):
            raise ValueError("Invalid input: img_arr must be a numpy array.")
        if img_arr.size == 0:
            raise ValueError("Invalid input: img_arr is empty.")
        try:
            image = sitk.Cast(sitk.GetImageFromArray(img_arr), sitk.sitkFloat32)
            image = sitk.MedianImageFilter().Execute(image)
            return sitk.GetArrayFromImage(image)
        except Exception as e:
            raise ValueError("Error during median image filtering: {}".format(str(e)))

    def binary_threshold_image_filter(
        self,
        img_arr: np.ndarray,
        lower_threshold: int = 0,
        upper_threshold: int = 1,
        inside_value: int = 1,
        outside_value: int = 0,
    ) -> np.ndarray:
        """
        Apply binary thresholding to the input image array.

        Args:
            img_arr (np.ndarray): The input image array.
            lower_threshold (int): The lower threshold value (default is 0).
            upper_threshold (int): The upper threshold value (default is 1).
            inside_value (int): The value to assign to pixels within the threshold range (default is 1).
            outside_value (int): The value to assign to pixels outside the threshold range (default is 0).

        Returns:
            The thresholded image.
        """
        if not isinstance(img_arr, np.ndarray):
            raise ValueError("Invalid input: img_arr must be a numpy array.")
        if lower_threshold > upper_threshold:
            raise ValueError(
                "Invalid threshold values: lower_threshold must be less than or equal to upper_threshold."
            )
        try:
            image = sitk.GetImageFromArray(img_arr)
            image = sitk.Cast(image, sitk.sitkFloat32)
            thresholder = sitk.BinaryThresholdImageFilter()
            thresholder.SetLowerThreshold(lower_threshold)
            thresholder.SetUpperThreshold(upper_threshold)
            thresholder.SetInsideValue(inside_value)
            thresholder.SetOutsideValue(outside_value)
            image = thresholder.Execute(image)
            image = sitk.Cast(sitk.GetImageFromArray(img_arr), sitk.sitkFloat32)
            return sitk.GetArrayFromImage(image)
        except Exception as e:
            raise ValueError("Error during binary thresholding: {}".format(str(e)))

    def sobel_image_filter(self, img_arr: np.ndarray):
        """
        Apply Sobel edge detection filter to the input image array.

        Args:
            img_arr (np.ndarray): The input image array.

        Returns:
            np.ndarray: The filtered image array.
        """
        if not isinstance(img_arr, np.ndarray):
            raise ValueError("Invalid input: img_arr must be a numpy array.")

        if img_arr.size == 0:
            raise ValueError("Invalid input: img_arr is empty.")

        if len(img_arr.shape) != 2 and len(img_arr.shape) != 3:
            raise ValueError("Invalid input: img_arr must be a 2D or 3D numpy array.")

        try:
            image = sitk.Cast(sitk.GetImageFromArray(img_arr), sitk.sitkFloat32)
            image = sitk.SobelEdgeDetectionImageFilter().Execute(image)
            return sitk.GetArrayFromImage(image)
        except Exception as e:
            raise ValueError("Error during Sobel image filtering: {}".format(str(e)))

    def laplacian_image_filter(self, img_arr: np.ndarray):
        """
        Apply Laplacian image filter to the input image array.

        Args:
            img_arr (np.ndarray): Input image array.

        Returns:
            np.ndarray: Filtered image array.
        """
        if not isinstance(img_arr, np.ndarray):
            raise ValueError("Invalid input: img_arr must be a numpy array.")

        if img_arr.size == 0:
            raise ValueError("Invalid input: img_arr is empty.")

        if len(img_arr.shape) != 2 and len(img_arr.shape) != 3:
            raise ValueError("Invalid input: img_arr must be a 2D or 3D numpy array.")

        try:
            image = sitk.Cast(sitk.GetImageFromArray(img_arr), sitk.sitkFloat32)
            image = sitk.LaplacianImageFilter().Execute(image)
            return sitk.GetArrayFromImage(image)
        except Exception as e:
            raise ValueError(
                "Error during Laplacian image filtering: {}".format(str(e))
            )

    def fourier_transform_filter(self, img_arr: np.ndarray):
        """
        Apply Fourier transform to the input image array.

        Args:
            img_arr (np.ndarray): The input image array.

        Returns:
            np.ndarray: The filtered image array.
        """
        if not isinstance(img_arr, np.ndarray):
            raise ValueError("Invalid input: img_arr must be a numpy array.")

        if img_arr.size == 0:
            raise ValueError("Invalid input: img_arr is empty.")

        try:
            kspace = scipy.fft.fft2(img_arr)
            return np.log(np.abs(scipy.fft.fftshift(kspace)))
        except Exception as e:
            raise ValueError("Error during Fourier transform: {}".format(str(e)))
        
    def inverse_fourier_trasform_filter(self, kspace_arr: np.ndarray):
        magnitude_space = np.abs(np.fft.ifft2(kspace_arr))
        return self.normalize_image_filter(magnitude_space)
    
    def apply_minimum_filter(self, img_arr: np.ndarray, kernel_size: int = 3):
        """
        Applies a minimum filter to the input image.
        Args:
            img_arr (np.ndarray): The input image.
            kernel_size (int) optional: The size of the kernel. Default is 3.
        Returns:
            np.ndarray: The filtered image.
        """
        filtered_image = minimum_filter(img_arr, size=kernel_size)
        return filtered_image