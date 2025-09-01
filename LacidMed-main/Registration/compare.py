#In this code we compare three feature-detection algorithms: BRISK, ORB, and SIFT.
#This part registers with all three algorithms, checks which one registered best,
#and stores that image in "reg_img". It is NOT the best overall algorithm—it's the best registered image per step.
#For example: if image 2 registers best with BRISK and image 3 with SIFT, the corresponding result is kept.
#This function returns the key information the rest of the code uses.

#Note: if the correlation difference (or anything else) is unclear, check
#"registration_brisk" first.

from registration_brisk import reg_br
from registration_orb import reg_or
from registration_sift import reg_si

def registration(images, len_images):
    # Create the required lists
    corr_list = []  # correlation difference
    H_x = []        # x translation
    H_y = []        # y translation
    rot_list = []   # rotation
    reg_img = [images[0]]  # registered images, starting with the original reference image

    print(len(images))
    # "reg_br", "reg_si", and "reg_or" are BRISK, SIFT, and ORB respectively
    # Each returns translation, rotation, correlation difference, and its list of registered images
    H_x_br, H_y_br, corr_list_br, rot_list_br, reg_img_br = reg_br(images, len_images)
    H_x_si, H_y_si, corr_list_si, rot_list_si, reg_img_si = reg_si(images, len_images)
    H_x_or, H_y_or, corr_list_or, rot_list_or, reg_img_or = reg_or(images, len_images)

    # Iterate over the length of the correlation-difference lists
    # Using BRISK's length (len(corr_list_br)) as the loop bound, but any of the three would work
    for i in range(1, len(corr_list_br) + 1):
        # "i" indexes the corresponding image. Example: "i=1" is the correlation difference between
        # image 2 and image 1 (registered vs. unregistered). We take the maximum correlation difference
        # and assume that is the best registration for that step, then append to corr_list.
        max_corr = max(corr_list_br[i - 1], corr_list_or[i - 1], corr_list_si[i - 1])
        corr_list.append(max_corr)

        # If the max comes from BRISK, store all its info in the lists
        if max_corr == corr_list_br[i - 1]:
            H_x.append(H_x_br[i])
            H_y.append(H_y_br[i])
            reg_img.append(reg_img_br[i])
            rot_list.append(rot_list_br[i])

        # Same for SIFT
        elif max_corr == corr_list_si[i - 1]:
            H_x.append(H_x_si[i])
            H_y.append(H_y_si[i])
            reg_img.append(reg_img_si[i])
            rot_list.append(rot_list_si[i])

        # Same for ORB
        elif max_corr == corr_list_or[i - 1]:
            H_x.append(H_x_or[i])
            H_y.append(H_y_or[i])
            reg_img.append(reg_img_or[i])
            rot_list.append(rot_list_or[i])

    return corr_list, H_x, H_y, rot_list, reg_img
