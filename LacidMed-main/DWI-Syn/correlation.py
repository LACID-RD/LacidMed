from scipy.stats import pearsonr

# Pearson correlation function
def pearson(b_final, truth_final, fil_min, fil_max, col_min, col_max):
    # Extract the region of interest (ROI)
    b_roi = b_final[fil_min:fil_max, col_min:col_max]
    truth_roi = truth_final[fil_min:fil_max, col_min:col_max]

    # Flatten the ROIs
    b_flat = b_roi.flatten()
    truth_flat = truth_roi.flatten()

    # Calculate the Pearson correlation coefficient
    correlation, _ = pearsonr(b_flat, truth_flat)

    return correlation

# Automating the Pearson correlation calculation for specific b-values
def calculate_pearson(b_final, truth_final, fil_min, fil_max, col_min, col_max):
    # Specify the b-values you are interested in
    b_values = [25, 50, 100, 200, 500, 1000, 1500, 1800]  # Original b-values
    indices = [0, 1, 2, 3, 4, 5, 6]  # Corresponding indices in the arrays

    # Loop through each corresponding pair of synthetic and truth images for the specified b-values
    for i, idx in enumerate(indices):
        correlation = pearson(b_final[idx], truth_final[idx], fil_min, fil_max, col_min, col_max)
        print(f"Pearson correlation coefficient for b={b_values[i]}: {correlation}")

