#Registración con algoritmo SIFT de feature detection
#para el código explicado revisar "registration_brisk".

import cv2
import numpy as np
from scipy.stats import pearsonr
from math import atan, degrees
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def reg_si(images, num_images):

    H_list_x_si = [0]
    H_list_y_si = [0]
    corr_list_si = []
    rot_list_si =[0]
    Registrated_images_si = [images[0]]

    for i in range(1, num_images):
        img1 = images[0]  # Use the last registered image
        img2 = images[i]
        

        # Convert the images to 8-bit grayscale
        img1 = cv2.normalize(img1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img2 = cv2.normalize(img2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Initialize the SIFT descriptor
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        # Initialize a Brute-Force Matcher
        bf = cv2.BFMatcher()

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find Homography
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)

        # Compute rotation angle in degrees
        theta = atan(H[1,0]/H[0,0])
        rot_angle = degrees(theta)

        # Warp img2 to img1
        img2_aligned = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

        # Convertir a [0, 1] para SSIM
        img1_ssim = img1.astype(np.float32) / 255.0
        img2_ssim = img2.astype(np.float32) / 255.0
        aligned_ssim = img2_aligned.astype(np.float32) / 255.0

        # Calcular SSIM
        ssim_before = ssim(img1_ssim, img2_ssim, data_range=1.0)
        ssim_after = ssim(img1_ssim, aligned_ssim, data_range=1.0)

        ssim_improvement = ssim_after - ssim_before  # Mejora de SSIM tras registración


        H_list_x_si.append(H[0, 2])
        H_list_y_si.append(H[1, 2])
        corr_list_si.append(ssim_improvement)
        rot_list_si.append(rot_angle)
        Registrated_images_si.append(img2_aligned)


    return H_list_x_si, H_list_y_si, corr_list_si, rot_list_si, Registrated_images_si
