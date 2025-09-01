import numpy as np
import matplotlib.pyplot as plt
from lmfit import Minimizer

def biexponential_fit_and_plot(r_f, r_c, limite_superior, limite_inferior, fil_min, col_min, vaux, b_value, mob, pars):
    # First Step: Fit and Plot without new b-values
    # Collect the pixel intensity values for all b-values for the selected pixel
    t = []
    for k in range(limite_superior - limite_inferior):
        t.append(vaux[limite_inferior + k][r_f + fil_min, r_c + col_min])

    p_p = np.array(t)  # Pixel intensities
    p_p_log = np.nan_to_num(np.log(p_p), nan=0, posinf=0, neginf=0)  # Logarithm of the signal for fitting

    # Fit the biexponential model for this random pixel
    r_fitter = Minimizer(mob, pars, fcn_args=(b_value, p_p_log))
    r_result = r_fitter.minimize()

    # Retrieve the fitted parameters
    f_fitted = r_result.params['f'].value
    D_star_fitted = r_result.params['D_star'].value
    D_fitted = r_result.params['D'].value
    S0_fitted = r_result.params['S_o'].value

    # Generate the fitted biexponential curve using the retrieved parameters
    r_final = S0_fitted * (f_fitted * np.exp(-b_value * D_star_fitted) + (1 - f_fitted) * np.exp(-b_value * D_fitted))

    # Print the fitted parameters
    print("f: ", f_fitted)
    print("D*: ", D_star_fitted)
    print("D: ", D_fitted)

    # Visualization of the data and biexponential fit curve
    plt.plot(b_value, p_p, 'o', color='red', label='Data')
    plt.plot(b_value, r_final, color='black', label='Biexponential Fit')
    plt.legend()
    plt.yscale('log')  # Keep log scale if necessary, adjust based on your desired visualization
    plt.xlabel("b-values [*0.01 s/mm^2]")
    plt.ylabel("Ln(S/S0)")
    plt.title("Fit vs. Data")
    plt.grid(True)

    # Remove the numbers on the y-axis
    plt.gca().set_yticklabels([])  # Option 1: Remove labels
    # plt.yticks([])  # Option 2: This also removes the ticks
    plt.show()

    # Second Step: Fit and Plot again, but add blue points for the new b-values
    # Collect the pixel intensity values for all b-values for the selected pixel (again)
    t = []
    for k in range(limite_superior - limite_inferior):
        t.append(vaux[limite_inferior + k][r_f + fil_min, r_c + col_min])

    p_p = np.array(t)  # Pixel intensities
    p_p_log = np.nan_to_num(np.log(p_p), nan=0, posinf=0, neginf=0)  # Logarithm of the signal for fitting

    # Fit the biexponential model again for this random pixel
    r_fitter = Minimizer(mob, pars, fcn_args=(b_value, p_p_log))
    r_result = r_fitter.minimize()

    # Retrieve the fitted parameters again
    f_fitted = r_result.params['f'].value
    D_star_fitted = r_result.params['D_star'].value
    D_fitted = r_result.params['D'].value
    S0_fitted = r_result.params['S_o'].value

    # Generate the fitted biexponential curve again using the retrieved parameters
    r_final = S0_fitted * (f_fitted * np.exp(-b_value * D_star_fitted) + (1 - f_fitted) * np.exp(-b_value * D_fitted))

    # Define the new b-values and collect their pixel intensities
    new_b_values = [0.75, 1.25, 1.75, 4, 5, 8, 12, 16]
    new_p_p = []

    for new_b in new_b_values:
        # Interpolate the intensities for the new b-values (or collect them if you have actual data)
        new_intensity = S0_fitted * (f_fitted * np.exp(-new_b * D_star_fitted) + (1 - f_fitted) * np.exp(-new_b * D_fitted))
        new_p_p.append(new_intensity)

    # Visualization of the data, biexponential fit curve, and new points
    plt.plot(b_value, p_p, 'o', color='red', label='Data (Pixel Intensities)')
    plt.plot(b_value, r_final, color='black', label='Biexponential Fit')

    # Plot new blue points for b-values 0.75, 1.25, and 1.75
    plt.plot(new_b_values, new_p_p, 'o', color='blue', label='DWI-Syn (New b-values)')

    # Final adjustments to the plot
    plt.legend()
    plt.yscale('log')  # Keep log scale if necessary, adjust based on your desired visualization
    plt.xlabel("b-values [*0.01 s/mm^2]")
    plt.ylabel("Ln(S/S0)")
    plt.title("Fit vs. Data")
    plt.grid(True)

    # Remove the numbers on the y-axis
    plt.gca().set_yticklabels([])  # Option 1: Remove labels
    # plt.yticks([])  # Option 2: This also removes the ticks
    plt.show()
