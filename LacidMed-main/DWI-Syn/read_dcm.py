##CODE TO READ DCM IMAGES AND STORE THEM##
#The "leer_dcm" function will iterate over the provided directory.
#The directory must contain the DCM images to be registered.
#Throughout the function, ONLY the pixel array will be stored in the "vaux" list.
#The code returns "vauxx" (the full DICOM datasets) and "vaux" (the pixel arrays of the unregistered images).

import pydicom

def leer_dcm(directory):

    #Initialize empty lists
    vauxx = []
    vaux = []

    #Iterate over the provided directory
    for i in range(len(directory)):
        ds = pydicom.dcmread(directory[i])
        vauxx.append(ds)
        #Extract only the pixel array and store it in "vaux"
        image_data = ds.pixel_array
        vaux.append(image_data)

    return vauxx, vaux
