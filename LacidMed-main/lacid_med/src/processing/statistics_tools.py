from typing import Tuple
import numpy as np
from scipy import stats


class StatisticsTools:
    @staticmethod
    def pearson_r(img_array_1: np.ndarray, img_array_2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate the Pearson correlation coefficient between two arrays.

        Parameters:
            img_array_1 (array-like): First input array.
            img_array_2 (array-like): Second input array.
            alternative_hypothesis (str, optional): The alternative hypothesis for the test. Default is "two-sided".

        Returns:
            float: Pearson correlation coefficient.
            float: Two-tailed p-value.
        """
        return stats.pearsonr(img_array_1.flatten(), img_array_2.flatten())

    @staticmethod
    def spearman_rho(img_array_1: np.ndarray, img_array_2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate the Spearman rank-order correlation coefficient between two arrays.

        Parameters:
            img_array_1 (array-like): First input array.
            img_array_2 (array-like): Second input array.
            alternative_hypothesis (str, optional): The alternative hypothesis for the test. Default is "two-sided".

        Returns:
            float: Spearman rank-order correlation coefficient.
            float: Two-tailed p-value.
        """
        return stats.spearmanr(img_array_1.flatten(), img_array_2.flatten())
    
        
