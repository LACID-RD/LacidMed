# # import matplotlib.pyplot as plt

# # def plot_DWI(b_final, new_b_final, truth_final, fil_min, fil_max, col_min, col_max):
# #     # First set of subplots (for ROI)
# #     b_values = [25, 50, 100, 200, 500, 1000, 1500, 1800]
# #     fig_2, axes_2 = plt.subplots(nrows=len(b_values), ncols=2, figsize=(16, 8))
# #     fig_2.patch.set_facecolor('black')  # Set figure background to black
    
# #     # Original b-values
    
# #     # Display ROI in the grid for each b-value
# #     for i in range(len(b_final)):
# #         #b_roi = b_final[i][fil_min:fil_max, col_min:col_max]
# #         #truth_roi = truth_final[i][fil_min:fil_max, col_min:col_max]

# #         # Plot DWI-Syn ROI
# #         axes_2[i][0].imshow(b_final[i], cmap='gray')
# #         axes_2[i][0].set_title(f"DWI-Syn para b={b_values[i]}", color='white')
# #         axes_2[i][0].axis("off")
# #         axes_2[i][0].set_facecolor('black')  # Set axis background to black

# #         # Plot DWI original ROI
# #         axes_2[i][1].imshow(truth_final[i], cmap='gray')
# #         axes_2[i][1].set_title(f"DWI original para b={b_values[i]}", color='white')
# #         axes_2[i][1].axis("off")
# #         axes_2[i][1].set_facecolor('black')  # Set axis background to black

# #     plt.tight_layout()
# #     plt.show()

# #     # Second set of subplots for ROI (original layout)
# #     fig_3, axes_3 = plt.subplots(nrows=2, ncols=len(b_values), figsize=(16, 8))
# #     fig_3.patch.set_facecolor('black')  # Set figure background to black
    
# #     for i in range(len(b_values)):
# #         #b_roi = b_final[i][fil_min:fil_max, col_min:col_max]
# #         #truth_roi = truth_final[i][fil_min:fil_max, col_min:col_max]

# #         # Plot DWI-Syn ROI
# #         axes_3[0][i].imshow(b_final[i], cmap='gray')
# #         axes_3[0][i].set_title(f"DWI-Syn para b={b_values[i]}", color='white')
# #         axes_3[0][i].axis("off")
# #         axes_3[0][i].set_facecolor('black')

# #         # Plot DWI original ROI
# #         axes_3[1][i].imshow(truth_final[i], cmap='gray')
# #         axes_3[1][i].set_title(f"DWI original para b={b_values[i]}", color='white')
# #         axes_3[1][i].axis("off")
# #         axes_3[1][i].set_facecolor('black')

# #     plt.tight_layout()
# #     plt.show()

# #     # Plot for the first set of b-values (75, 125, 175) for ROI
# #     new_b_values = [75, 125, 175, 400]
# #     fig_1, axes_1 = plt.subplots(nrows=1, ncols=len(new_b_values), figsize=(12, 4))
# #     fig_1.patch.set_facecolor('black')  # Set figure background to black

# #     # Plot the first set of b-values for DWI-Syn ROI
# #     for i, b_val in enumerate(new_b_values):
# #         #b_roi = new_b_final[i][fil_min:fil_max, col_min:col_max]
# #         axes_1[i].imshow(new_b_final[i], cmap='gray')
# #         axes_1[i].set_title(f"DWI-Syn para b={b_val}", color='white')
# #         axes_1[i].axis("off")
# #         axes_1[i].set_facecolor('black')

# #     plt.tight_layout()
# #     plt.show()

# #     # # Plot for the second set of b-values (400, 800, 1000) for ROI
# #     # new_b_values_2 = [400, 800, 1000]
# #     # fig_2, axes_2 = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
# #     # fig_2.patch.set_facecolor('black')  # Set figure background to black

# #     # # Plot the second set of b-values for DWI-Syn ROI
# #     # for i, b_val in enumerate(new_b_values_2):
# #     #     if i < 2:
# #     #         b_roi = b_final[i + 6][fil_min:fil_max, col_min:col_max]
# #     #     else:
# #     #         b_roi = b_final[11][fil_min:fil_max, col_min:col_max]  # For b=1000
# #     #     axes_2[i].imshow(b_roi, cmap='gray')
# #     #     axes_2[i].set_title(f"DWI-Syn para b={b_val}", color='white')
# #     #     axes_2[i].axis("off")
# #     #     axes_2[i].set_facecolor('black')

# #     # plt.tight_layout()
# #     # plt.show()

import matplotlib.pyplot as plt

def plot_DWI(b_final, new_b_final, truth_final, fil_min, fil_max, col_min, col_max):
    # Lista de b-values
    b_values = [25, 50, 100, 200, 500, 1000, 1500, 1800]

    # Figura 1: Subplots (ROI) en 2 columnas
    fig_2, axes_2 = plt.subplots(nrows=len(b_values), ncols=2, figsize=(16, 8))
    fig_2.patch.set_facecolor('black')

    for i in range(len(b_final)):
        # Recortar ROI
        b_roi = b_final[i][fil_min:fil_max, col_min:col_max]
        truth_roi = truth_final[i][fil_min:fil_max, col_min:col_max]

        # DWI-Syn
        axes_2[i][0].imshow(b_roi, cmap='gray')
        axes_2[i][0].set_title(f"DWI-Syn b={b_values[i]}", color='white')
        axes_2[i][0].axis("off")
        axes_2[i][0].set_facecolor('black')

        # DWI original
        axes_2[i][1].imshow(truth_roi, cmap='gray')
        axes_2[i][1].set_title(f"DWI b={b_values[i]}", color='white')
        axes_2[i][1].axis("off")
        axes_2[i][1].set_facecolor('black')

    plt.tight_layout()
    plt.show()

    # Figura 2: Layout original en 2 filas
    fig_3, axes_3 = plt.subplots(nrows=2, ncols=len(b_values), figsize=(16, 8))
    fig_3.patch.set_facecolor('black')

    for i in range(len(b_values)):
        b_roi = b_final[i][fil_min:fil_max, col_min:col_max]
        truth_roi = truth_final[i][fil_min:fil_max, col_min:col_max]

        # DWI-Syn
        axes_3[0][i].imshow(b_roi, cmap='gray')
        axes_3[0][i].set_title(f"DWI-Syn b={b_values[i]}", color='white')
        axes_3[0][i].axis("off")
        axes_3[0][i].set_facecolor('black')

        # DWI original
        axes_3[1][i].imshow(truth_roi, cmap='gray')
        axes_3[1][i].set_title(f"DWI b={b_values[i]}", color='white')
        axes_3[1][i].axis("off")
        axes_3[1][i].set_facecolor('black')

    plt.tight_layout()
    plt.show()

    # Figura 3: Nuevos b-values (ROI)
    new_b_values = [75, 125, 175, 350, 800, 1200, 1600]
    fig_1, axes_1 = plt.subplots(nrows=1, ncols=len(new_b_values), figsize=(12, 4))
    fig_1.patch.set_facecolor('black')

    for i, b_val in enumerate(new_b_values):
        new_b_roi = new_b_final[i][fil_min:fil_max, col_min:col_max]
        axes_1[i].imshow(new_b_roi, cmap='gray')
        axes_1[i].set_title(f"DWI-Syn para b={b_val}", color='white')
        axes_1[i].axis("off")
        axes_1[i].set_facecolor('black')

    plt.tight_layout()
    plt.show()


# ##USAR ESTE CODIGO PARA IMRPIMIR DE UNA CARPETA EN ESPECIFICO

# import os
# import re
# import pydicom
# import matplotlib.pyplot as plt

# # Sorting function
# def numericalSort(value):
#     numbers = re.compile(r'(\d+)')
#     parts = numbers.split(value)
#     parts[1::2] = map(int, parts[1::2])
#     return parts

# def sort(input_path):
#     directory = input_path
#     directory_files = []
#     for file in sorted(os.listdir(directory), key=numericalSort):    
#         path = os.path.join(directory, file)
#         if file.endswith(".dcm"):                
#             directory_files.append(path)
#         else:
#             print("This file is not a DICOM file: " + path)
#     return directory_files

# # DICOM reading function
# def leer_dcm(directory):
#     vauxx = []
#     vaux = []
#     for i in range(len(directory)):
#         ds = pydicom.dcmread(directory[i])
#         vauxx.append(ds)
#         image_data = ds.pixel_array
#         vaux.append(image_data)
#     return vauxx, vaux

# # ROI limits
# fil_min = 70
# fil_max = 185
# col_min = 70
# col_max = 185

# # Directory with DICOM images
# input_path = r"E:\FUESMEN\DWI_Generator\SRC\1.5T Optimized\Ground_truth\VARGAS\registered\Generated\shifted\prueba 2"

# # Sort and read the images
# sorted_files = sort(input_path)
# headers, images = leer_dcm(sorted_files)

# # b-values for titles
# b_values = [75, 125, 175, 350, 800, 1200, 1600]

# # Crop to ROI
# cropped_images = [img[fil_min:fil_max, col_min:col_max] for img in images]

# # Plotting
# fig, axs = plt.subplots(1, len(cropped_images), figsize=(15, 5))
# fig.patch.set_facecolor('black')

# for idx, ax in enumerate(axs):
#     ax.imshow(cropped_images[idx], cmap='gray')
#     ax.set_title(f"DWI-Syn b={b_values[idx]}", color='white')
#     ax.axis('off')
#     ax.set_facecolor('black')

# plt.tight_layout()
# plt.show()
