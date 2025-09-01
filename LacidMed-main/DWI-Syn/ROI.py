import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def ROI(vaux, vauxx, num_b):
    
    filas = vauxx[1].Rows
    columnas = vauxx[1].Columns
    
    fil_min = 50
    fil_max = 195
    col_min = 70
    col_max = 185

    fig_0, axes = plt.subplots(nrows=1, ncols=num_b, figsize=(6, 50))  # show images
    plt.set_cmap("gray")

    for i in range(num_b):
        axes[i].imshow(vaux[i])
        axes[i].axis("off")
        axes[i].add_patch(mpatches.Rectangle((col_min, fil_min), col_max - col_min, fil_max - fil_min, fill=False, edgecolor='red', linewidth=1))
    plt.show()



    return filas, columnas, fil_min, fil_max, col_min, col_max