import unittest

import numpy as np
import pydicom
import SimpleITK as sitk

from lacid_med.src.processing.filters import Filters


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.sequence_directory = "data/34847A52E3914AE134134F25EA808996/1.2.840.113619.2.5.181823172100.700.1634600743.180"
        self.dicom_files = [pydicom.Dataset(), pydicom.Dataset()]
        self.output_dir = "data"
        self.filters = Filters(self.sequence_directory, self.dicom_files, self.output_dir)

    def test_resample_image(self):
        itk_image = sitk.Image(10, 10, 10)
        resampled_image = self.filters.resample_image(itk_image)
        self.assertIsInstance(resampled_image, sitk.Image)

    def test_N4_bias_correction_filter(self):
        image = sitk.Image(10, 10, 10)
        n4_filtered_image = self.filters.N4_bias_correction_filter(image)
        self.assertIsInstance(n4_filtered_image, sitk.Image)

    def test_normalize_image_filter(self):
        image_arr = np.random.rand(10, 10, 10).astype(np.uint32)
        normalized_image = self.filters.normalize_image_filter(image_arr)
        self.assertIsInstance(normalized_image, np.ndarray)

    def test_gaussian_image_filter(self):
        img_arr = np.random.rand(10, 10).astype(np.uint32)
        gaussian_filtered_image = self.filters.gaussian_image_filter(img_arr)
        self.assertIsInstance(gaussian_filtered_image, np.ndarray)

    def test_median_image_filter(self):
        img_arr = np.random.rand(10, 10).astype(np.uint32)
        median_filtered_image = self.filters.median_image_filter(img_arr)
        self.assertIsInstance(median_filtered_image, np.ndarray)

    def test_binary_threshold(self):
        img_arr = np.random.rand(10, 10).astype(np.uint32)
        binary_thresholded_image = self.filters.binary_threshold(img_arr)
        self.assertIsInstance(binary_thresholded_image, sitk.Image)

    def test_sobel_image_filter(self):
        img_arr = np.random.rand(10, 10).astype(np.uint32)
        sobel_filtered_image = self.filters.sobel_image_filter(img_arr)
        self.assertIsInstance(sobel_filtered_image, np.ndarray)

    def test_laplacian_image_filter(self):
        img_arr = np.random.rand(10, 10).astype(np.uint32)
        laplacian_filtered_image = self.filters.laplacian_image_filter(img_arr)
        self.assertIsInstance(laplacian_filtered_image, np.ndarray)

    def test_fourier_transform_filter(self):
        img_arr = np.random.rand(10, 10).astype(np.uint32)
        fourier_transformed_image = self.filters.fourier_transform_filter(img_arr)
        self.assertIsInstance(fourier_transformed_image, np.ndarray)


if __name__ == "__main__":
    unittest.main()