##THIS CODE IS USED FOR REGISTRATION##
#In the main function all other functions will be called
#If any variable is unclear, check its function to understand it

#WARNING: this code requires distinct images. For example, in a PWI
#the first images, where the contrast has not progressed yet, will not be registered
#and it may throw an error.

#REMEMBER: in "leer_dcm" and in "save" you must set/use the appropriate directory path

from dcm import leer_dcm
from compare import registration    
from plot import plot
from save import save
from sorted_files import main2
import matplotlib.pyplot as plt
import os
    
#"images" is the list with the unregistered images
#"len_images" is the length of this list

directory_files = main2()

images, len_images = leer_dcm(directory_files)


#corr_list is the correlation difference (it should always be positive)
#H_x is the translation in x
#H_y is the translation in y
#rot_list is the rotation
#reg_img is the list with the registered images
corr_list, H_x, H_y, rot_list, reg_img = registration(images,len_images)

#Show all images (test to verify registration)
for i in range(len(reg_img)):
   plt.figure(i)
   plt.imshow(reg_img[i], cmap="gray")
   plt.title(f"Image {i+1}")
   plt.axis('off')
   plt.show()

#print the correlation list to verify it is positive
print([float(x) for x in corr_list])

#call the plot function to plot translation and rotation
plot(H_x, H_y, rot_list)   

# Save the pixel arrays to new DICOM files
save(reg_img)
