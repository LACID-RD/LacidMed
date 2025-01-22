import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pydicom
from scipy.ndimage import label


class Segmentation:
    def __init__(self, 
        image_array: np.array = None, 
        volumetric_array: np.array = None,
    ):
        """
        Initializes segmentation object.
        Args:
            image_array (np.array, optional): The image array to segment. Defaults to None.
            volumetric_array (np.array, optional): The volumetric array to segment. Defaults to None.
        Raises:
            ValueError: If neither image_array nor volumetric_array is provided.
            ValueError: If both image_array and volumetric_array are provided.
            ValueError: If image_array or volumetric_array is not a numpy array.
            ValueError: If volumetric_array has less than 2 slices.
        """
        
        if image_array is None and volumetric_array is None:
            raise ValueError("No image or volume at input.")
        
        if image_array is not None and volumetric_array is not None:
            raise ValueError("Only one input can be provided.")
        
        if not isinstance(image_array, np.ndarray) and image_array is not None:
            raise ValueError("Input image must be a numpy array.")
        
        if not isinstance(volumetric_array, np.ndarray) and volumetric_array is not None:
            raise ValueError("Input volume must be a numpy array.")

        if volumetric_array is not None:
            if volumetric_array.shape[2] <= 1:
                raise ValueError("Input volume must have more than 1 slice.")
        
        self.image_array = image_array
        self.volumetric_array = volumetric_array

    def threshold_segmentation(self, 
        lower_threshold: int = None, 
        upper_threshold: int = None,
    ):
        """
        Generate a segmentation of the array based on a threshold. 
        If no volumetric array is given, volumetric_array_1 is used.
        Args:
            image_array (np.array, optional): The image or volume array to segment. Defaults to volumetric_array_1.
            lower_threshold (int, optional): The lower threshold for the segmentation. Defaults to the minimum value in the array.
            upper_threshold (int, optional): The upper threshold for the segmentation. Defaults to the maximum value in the array.
        Returns: 
            np.array: The segmented image or volume array.
        """
        if self.volumetric_array is None:
            array_2_clip = self.image_array
        else:
            array_2_clip = self.volumetric_array
        if lower_threshold == None: 
            lower_threshold = 0 
            print("Check lower threshold")
        if upper_threshold == None: 
            upper_threshold = np.max(array_2_clip)
            print("Check upper threshold")
        if upper_threshold > np.max(array_2_clip):
            raise ValueError("Upper threshold must be less than the maximum value in the array.")
        if lower_threshold < np.min(array_2_clip):
            raise ValueError("Lower threshold must be greater than the minimum value in the array.")
        if upper_threshold < lower_threshold:
            raise ValueError("Upper threshold must be greater than the lower threshold.")
        clipped_array = np.copy(array_2_clip)
        clipped_array[clipped_array < lower_threshold] = 0
        clipped_array[clipped_array > upper_threshold] = 0
        return clipped_array
    
    def region_growing(
        self, 
        seed_point: list = None, 
        number_of_iterations: int = None,
        multiplier: float = None,
        invert_mask: bool = False,
    ):
        """
        Generate a segmentation of the array based on region growing.
        Args:
            image_array: The image array to segment, converted to a SimpleITK image for segmentation.
            seed_point (list, optional): The seed point for the segmentation. Defaults to [0, 0].
            number_of_iterations (int, optional): The number of iterations for the segmentation. Defaults to 10.
            multiplier (float, optional): The multiplier for the segmentation. Defaults to 4.
            invert_mask (bool, optional): Whether to invert the segmentation mask. Defaults to False.
        Returns:
            np.array: The segmented image array.
        """
        if multiplier is None:
            # default multiplier set to 4.
            multiplier = 4 
            print("It is recomended to use a custom multiplier. Default is 4.")
        if number_of_iterations is None:
            # default number of iterations set to 10.
            number_of_iterations = 10
            print("It is recomended to use a custom number of iterations. Default is 10.")                
        if seed_point is None:
            if self.image_array is not None:
                seed_point = [0, 0]
                print("It is recomended to use a custom seed point. Default is (0, 0).")
            else:
                seed_point = [0, 0, 0]
                print("It is recomended to use a custom seed point. Default is (0, 0, 0).")
        elif len(seed_point) == 3:
            # convert to [z, x, y], wich is the format of SimpleITK.
            z = seed_point[2]
            x = seed_point[0]
            y = seed_point[1]
            seed_point = [z, x, y]
        if self.image_array is not None:
            sitk_image = sitk.GetImageFromArray(self.image_array)
            if len(seed_point) != 2:
                raise ValueError("Seed point must have 2 dimensions.")
        else:
            sitk_image = sitk.GetImageFromArray(self.volumetric_array)
            if len(seed_point) != 3:
                raise ValueError("Seed point must have 3 dimensions.")
        seed_index = sitk_image.TransformPhysicalPointToIndex(seed_point)
        seg_filter = sitk.Image(sitk_image.GetSize(), sitk.sitkUInt8)
        seg_filter.CopyInformation(sitk_image)
        seg_filter[seed_index] = 1
        seg_result = sitk.ConfidenceConnected(
                    sitk_image, 
                    seedList=[seed_index],
                    numberOfIterations=number_of_iterations,
                    multiplier=multiplier,
                    initialNeighborhoodRadius= 1,
                    replaceValue=1
                    )
        if invert_mask:
            seg_result = sitk.Not(seg_result)
        seg_array = sitk.GetArrayFromImage(seg_result)
        return seg_array

    def find_largest_component(self, volume: np.ndarray = None):
        """
        Finds and segments the largest interconnected mass in a 3D binary volume.
        Args:
            volume (np.ndarray): 3D binary volume. Optional if volumetric_array is provided (mostly used for the backround_remover_volumetric method). 
            
        Returns:
            np.ndarray: A mask of the largest interconnected mass.
        """
        if volume is not None:
            labeled_volume, num_features = label(volume)    
        else:
            labeled_volume, num_features = label(self.volumetric_array)
        largest_component_label = 0
        largest_component_size = 0
        for label_num in range(1, num_features + 1):
            component_size = np.sum(labeled_volume == label_num)
            if component_size > largest_component_size:
                largest_component_size = component_size
                largest_component_label = label_num
        largest_component_mask = (labeled_volume == largest_component_label)        
        return largest_component_mask

    
    def background_remover_volumetric(self, background_seed_point=[0,0,0], background_multiplier=3, background_number_of_iterations=50):
        """
        Subtracts the background from the image array.
        Args:
            image_array (np.ndarray): The image array to subtract the background from.
            background_seed_point (list, optional): The seed point for the background segmentation. Defaults to [0, 0].
            background_multiplier (float, optional): The multiplier for the background segmentation. Defaults to 5.
            background_number_of_iterations (int, optional): The number of iterations for the background segmentation. Defaults to 100.
        Returns:
            np.ndarray: The image array with the background subtracted.
        """
        region_growing_mask = self.region_growing(seed_point=background_seed_point, multiplier=background_multiplier, number_of_iterations=background_number_of_iterations, invert_mask=True)
        filtered_array = np.multiply(region_growing_mask, self.volumetric_array)
        largest_component_mask = self.find_largest_component(filtered_array)
        largest_component = np.multiply(largest_component_mask, self.volumetric_array)
        return largest_component

    def canny_volumetric(self, low_threshold, high_threshold):
        edges_volume = np.zeros_like(self.volumetric_array)        
        for i in range(self.volumetric_array.shape[2]):
            edges_volume[i] = cv2.Canny(self.volumetric_array[i].astype(np.uint8), low_threshold, high_threshold)
        return edges_volume










