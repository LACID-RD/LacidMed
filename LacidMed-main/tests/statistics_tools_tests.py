import unittest
from lacid_med.src.processing.statistics_tools import StatisticsTools
import numpy as np
from scipy import stats


class TestStatisticsTools(unittest.TestCase):
    def test_pearson_r(self):
        img_array_1 = np.random.rand(100)
        img_array_2 = np.random.rand(100)
        corr, p_value = StatisticsTools.pearson_r(img_array_1, img_array_2)
        self.assertAlmostEqual(corr, np.corrcoef(img_array_1, img_array_2)[0, 1], places=4)
        self.assertAlmostEqual(p_value, stats.pearsonr(img_array_1, img_array_2)[1], places=4)

    def test_spearman_rho(self):
        img_array_1 = np.random.rand(100)
        img_array_2 = np.random.rand(100)
        corr, p_value = StatisticsTools.spearman_rho(img_array_1, img_array_2)
        self.assertAlmostEqual(corr, stats.spearmanr(img_array_1, img_array_2)[0], places=4)
        self.assertAlmostEqual(p_value, stats.spearmanr(img_array_1, img_array_2)[1], places=4)


if __name__ == '__main__':
    unittest.main()