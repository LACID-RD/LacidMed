##CODE TO READ DCM IMAGES AND STORE THEM##
#The "leer_dcm" function will iterate over the directory you provide
#The directory must contain the DCM images to be registered
#Throughout the function, ONLY the pixel array will be stored in the "images" list
#The code returns "images", the list of all pixel arrays of the unregistered images
#and "len_images", which is simply the length of that list (number of images)

import pydicom

def leer_dcm(directory):

    #Initialize the empty list
    images = []

    #Iterate over the provided directory
    for i in range(len(directory)):
        ds = pydicom.dcmread(directory[i])
        #Extract only the pixel array and store it in "images"
        image_data = ds.pixel_array
        images.append(image_data)

    len_images = len(images)

    return images, len_images
