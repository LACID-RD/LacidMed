import numpy as np
import matplotlib.pyplot as plt


class HistogramGenerator:
    def __init__(self, array_2D=None, array_3D=None):
        self.array_2D = array_2D if array_2D is not None else None
        self.array_3D = array_3D if array_3D is not None else None

    def create_histogram_2D(
        self,
        offset: int = 0,
        show: bool = True,
        plot_title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        x_step: int = None,
        y_step: int = None,
    ):
        """
        Generates a 2D histogram from the provided 2D array and returns the bins and frequencies.

        Args:
            offset (int, optional): The offset to apply to the histogram bins and frequencies. Defaults to 0.
            show (bool, optional): Whether to display the histogram using matplotlib. Defaults to True.

        Returns:
            tuple: bins and frequencies of the histogram
        """
        if self.array_2D is None:
            return None, None

        arr_max = int(np.max(self.array_2D))
        hist, bins = np.histogram(self.array_2D.flatten(), bins=arr_max, density=False)

        if offset < len(bins):
            bins = bins[offset + 1 :]
            hist = hist[offset:]
        else:
            bins = []
            hist = []

        if show:
            try:
                plt.plot(bins, hist)
                if x_step is not None:
                    plt.xticks(np.arange(0, arr_max+x_step,x_step))            
                if y_step is not None:
                    plt.yticks(np.arange(0, np.max(hist)+y_step,y_step))
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(plot_title)
                plt.grid()
                plt.show()
            except Exception as e:
                print("Error plotting histogram:", str(e))

        return hist, bins

    def create_histogram_of_3d_array(
        self,
        offset: int = 0,
        show: bool = True,
        plot_title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        x_step: int = None,
        y_step: int = None,
        
    ):
        """
        Generates a 3D histogram from the provided 3D array and returns the bins and frequencies.

        Args:
            offset (int, optional): The offset to apply to the histogram bins and frequencies. Defaults to 0.
            show (bool, optional): Whether to display the histogram using matplotlib. Defaults to True.

        Returns:
            tuple: bins and frequencies of the histogram
        """
        if self.array_3D is None:
            return None, None

        arr_max = int(np.max(self.array_3D))
        hist, bins = np.histogram(self.array_3D.flatten(), bins=arr_max, density=False)

        if offset < len(bins):
            bins = bins[offset + 1 :]
            hist = hist[offset:]
        else:
            bins = []
            hist = []

        if show:
            try:
                plt.plot(bins, hist)
                if x_step is not None:
                    plt.xticks(np.arange(0, arr_max+x_step,x_step))            
                if y_step is not None:
                    plt.yticks(np.arange(0, np.max(hist)+y_step,y_step))
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(plot_title)
                plt.grid()
                plt.show()
            except Exception as e:
                print("Error plotting histogram:", str(e))

        return hist, bins 
    
    def create_mean_histogram_of_3d_array(
        self,
        offset: int = 0,
        show: bool = True,
        plot_title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        x_step: int = None,
        y_step: int = None,
        
    ):
        """
        Generates a 3D histogram from the provided 3D array and returns the bins and frequencies.

        Args:
            offset (int, optional): The offset to apply to the histogram bins and frequencies. Defaults to 0.
            show (bool, optional): Whether to display the histogram using matplotlib. Defaults to True.

        Returns:
            tuple: bins and frequencies of the histogram
        """
        if self.array_3D is None:
            return None, None

        arr_max = int(np.max(self.array_3D))
        hist, bins = np.histogram(self.array_3D.flatten(), bins=arr_max, density=False) #flaten trnasforma matriz en arrray
        hist_mean = hist / self.array_3D.shape[2]

        if offset < len(bins):
            bins = bins[offset + 1 :]
            hist_mean = hist_mean[offset:]
        else:
            bins = []
            hist_mean = []

        if show:
            try:
                plt.plot(bins, hist_mean)
                if x_step is not None:
                    plt.xticks(np.arange(0, arr_max+x_step,x_step))            
                if y_step is not None:
                    plt.yticks(np.arange(0, np.max(hist_mean)+y_step,y_step))
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.title(plot_title)
                plt.grid()
                plt.show()
            except Exception as e:
                print("Error plotting histogram:", str(e))

        return hist_mean, bins 



    def clip_histogram(self, lower_threshold: None, upper_threshold: int = None):
        """
        Generate a clipped histogram of the volumetric array in the provided range.
        Args:
            lower_threshold (int, optional): The lower threshold to apply to the histogram. Defaults to the minimum value in the array.
            upper_threshold (int, optional): The upper threshold to apply to the histogram. Defaults to the maximum value in the array.

        Returns:
            tuple: bins and frequencies of the histogram
        """
        if lower_threshold == None: 
            lower_threshold = 0 
            print("Check lower threshold")
        if upper_threshold >= np.max(self.array_3D):
            raise ValueError("Upper threshold must be less than the maximum value in the array.")
        if upper_threshold == None: 
            upper_threshold = np.max(self.array_3D)
            print("Check upper threshold")
        if lower_threshold < np.min(self.array_3D):
            raise ValueError("Lower threshold must be greater than the minimum value in the array.")
        if upper_threshold < lower_threshold:
            raise ValueError("Upper threshold must be greater than the lower threshold.")
        clipped_array = np.clip(self.array_3D, lower_threshold, upper_threshold)
        hist, bins = self.create_histogram_of_3d_array(clipped_array)
        return hist, bins
    