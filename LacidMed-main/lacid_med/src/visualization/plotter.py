import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
from typing import List

class DicomPlotter:
    def __init__(self, dicom_files: List[str]) -> None:
        self.dicom_files = None

    def _show_image(self, arr: np.ndarray, show: bool = True):
        """
        Plot a 2D or 3D array using SimpleITK.

        Args:
            arr (np.ndarray): A 2D or 3D array.

        Returns:
            matplotlib.image.AxesImage: The plot object representing the image.
        """
        plt_images = []
        itk_image = sitk.GetImageFromArray(arr)
        plt_images.append(plt.imshow(sitk.GetArrayViewFromImage(itk_image), cmap="gray"))
        print(plt_images)
        if show:
            plt.show()
        return plt_images

    def plot_all_files(self):
        
        def load_and_plot_images():
            for dicom in self.dicom_files:
                ds = pydicom.dcmread(dicom)
                yield ds.pixel_array

        if self.dicom_files is None:
            return
        else:
            for pixel_array in load_and_plot_images():
                self._show_image(pixel_array)


    def plot_single_file(self, img_arr:np.ndarray):
        return self._show_image(img_arr)