import numpy as np
from skimage import exposure

def adjust_window_level(pixel_array, window=1595, level=797.5, output_min=0, output_max=4096):
    """
    Apply window and level to the pixel array.
    """
    lower = level - window / 2
    upper = level + window / 2

    # Clip and normalize
    clipped = np.clip(pixel_array, lower, upper)
    scaled = (clipped - lower) / (upper - lower) * (output_max - output_min) + output_min
    return scaled.astype(np.uint16)

def generate_maps(b_maps, new_b_maps, b_final, new_b_final, truth_maps, truth_final, truth_complete, S0_map, f_map, D_star_map, D_map, RESOLUTION):
    b_values = [0.25, 0.5, 1, 2, 5, 10, 15, 18]

    for i, b in enumerate(b_values):
        b_final[i] = S0_map * (f_map * np.exp(-b * D_star_map) + (1 - f_map) * np.exp(-b * D_map))

    # Match contrast of truth_maps to b_final
    for i in range(len(truth_maps)):
        matched = exposure.match_histograms(truth_maps[i], b_final[i], channel_axis=None)
        truth_final[i] = adjust_window_level(matched)  # Apply window-level after matching

    new_b_values = [0.75, 1.25, 1.75, 3.50, 8, 12, 16]
    for i, b in enumerate(new_b_values):
        new_b_final[i] = S0_map * (f_map * np.exp(-b * D_star_map) + (1 - f_map) * np.exp(-b * D_map))

    return b_final, new_b_final, truth_final
