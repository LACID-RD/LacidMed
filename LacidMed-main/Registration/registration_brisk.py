#This code implements the BRISK algorithm for feature detection.
#It will return the images registered according to this algorithm.

import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.stats import pearsonr
from math import atan, degrees
from skimage.metrics import structural_similarity as ssim

def reg_br(images, num_images):

    #Create the lists that will hold the outputs. The ones starting with 0 correspond to img1 (unregistered)
    H_list_x_br = [0]  # translation in x
    H_list_y_br = [0]  # translation in y
    corr_list_br = []  # correlation difference
    rot_list_br = [0]  # rotation
    Registrated_images_br = [images[0]]  # registered images (img1 remains unregistered)

    for i in range(1, num_images):
        img1 = images[0]  # Use the last registered image
        img2 = images[i]
        
        #Convert to grayscale
        img1 = cv2.normalize(img1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img2 = cv2.normalize(img2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        #Initialize the BRISK descriptor
        brisk = cv2.BRISK_create()

        #Detect keypoints and descriptors for each image (feature detection)
        keypoints1, descriptors1 = brisk.detectAndCompute(img1, None)
        keypoints2, descriptors2 = brisk.detectAndCompute(img2, None)

        #Initialize descriptor matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        #Store the matches in "matches" (pun intended)
        matches = bf.match(descriptors1, descriptors2)

        #Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        #Extract the matching keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        #Find the homography matrix
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

        #Compute the rotation angle
        theta = atan(H[1, 0] / H[0, 0])
        rot_angle = degrees(theta)

        #Apply the transformation to "img2"
        img2_aligned = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

        #Convert to [0, 1] for SSIM
        img1_ssim = img1.astype(np.float32) / 255.0
        img2_ssim = img2.astype(np.float32) / 255.0
        aligned_ssim = img2_aligned.astype(np.float32) / 255.0

        #Compute SSIM
        ssim_before = ssim(img1_ssim, img2_ssim, data_range=1.0)
        ssim_after = ssim(img1_ssim, aligned_ssim, data_range=1.0)

        ssim_improvement = ssim_after - ssim_before  # SSIM improvement after registration

        #Append to lists
        H_list_x_br.append(H[0, 2])
        H_list_y_br.append(H[1, 2])
        corr_list_br.append(ssim_improvement)
        rot_list_br.append(rot_angle)
        Registrated_images_br.append(img2_aligned)

    return H_list_x_br, H_list_y_br, corr_list_br, rot_list_br, Registrated_images_br
