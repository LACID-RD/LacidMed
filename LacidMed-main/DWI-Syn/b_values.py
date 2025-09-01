import pydicom
import numpy as np

def b_values(vauxx, directory_files, num_b):
    orden = []

    for i in range(num_b):
        path_imagen = directory_files[i]  # Use sorted file paths
        dicom_b = pydicom.dcmread(path_imagen)  # Read DICOM file
        
        try:
            # Philips Ingenia: Uses Private Tag (0x2001, 0x1003)
            b = dicom_b[0x2001, 0x1003].value
        except KeyError:
            # General Electric: Uses (0x0043, 0x1039)
            b = dicom_b[0x0043, 0x1039].value
            out1 = str(b)
            out2 = list(out1)
            
            if int(out2[1]) == 0:
                b = 0
            else:
                b = int(out2[10]) + 10 * int(out2[9]) + 100 * int(out2[8]) + 1000 * int(out2[7])

        orden.append(int(b))  # Convert b-value to integer

    # Normalize b-values to avoid LMFIT issues
    orden = np.array(orden) / 100
    print(orden)

    return orden
