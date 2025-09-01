import numpy as np

def create_maps(filas, columnas, vaux):
    # Initialize empty lists for b_maps, b_final, truth_maps, and truth_final
    b_maps = []
    new_b_maps = []
    b_final = []
    new_b_final = []
    truth_maps = []
    truth_final = []
    truth_complete = []

    # Define all the b-values to generate maps for
    orden = [25, 50, 100, 200, 500, 1000, 1500, 1800]  # Adding new values

    for num in orden:
        # Append zeros array to each list for b_maps and b_final
        b_map = np.zeros((filas, columnas))
        b_maps.append(b_map)
        b_final.append(np.copy(b_map))  # Initialize with the same size

        # Append zeros array for truth_maps and truth_final
        truth_map = np.zeros((filas, columnas))
        truth_maps.append(truth_map)
        truth_final.append(np.copy(truth_map))
    
    new_b_values = [75, 125, 175, 350, 800, 1200, 1600]

    for num in new_b_values:
        new_b_map = np.zeros((filas, columnas))
        new_b_maps.append(new_b_map)
        new_b_final.append(np.copy(new_b_map))

    # Copy truth images into truth_complete
    for i in range(len(vaux)):
        truth_complete.append(vaux[i])

    return b_maps, new_b_maps, b_final, new_b_final, truth_maps, truth_final, truth_complete
