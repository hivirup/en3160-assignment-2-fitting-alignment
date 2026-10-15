import cv2 as cv
import numpy as np

# Load template image (a)
template = cv.imread('a2images/pcbs/AECC1.jpg', cv.IMREAD_GRAYSCALE)
sift = cv.SIFT_create()
kp_temp, des_temp = sift.detectAndCompute(template, None)

# List of the other PCB images
test_images = ['AAAA1.jpg', 'ADAD1.jpg', 'AECC2.jpg', 'AECC3.jpg', 'AECC4.jpg']

for img_name in test_images:
    test_img = cv.imread(f'a2images/pcbs/{img_name}', cv.IMREAD_GRAYSCALE)
    
    # (a) Detect SIFT features
    kp_test, des_test = sift.detectAndCompute(test_img, None)
    
    # (b) Match features using BFMatcher
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_test, des_temp, k=2)
    
    # Lowe's ratio test to keep only strong matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)
            
    # Extract coordinates of good matches
    src_pts = np.float32([kp_test[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_temp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography using RANSAC
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    
    # Draw matches
    matches_img = cv.drawMatches(test_img, kp_test, template, kp_temp, good_matches, None, 
                                 matchesMask=mask.ravel().tolist(), 
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(f'matches_{img_name}', matches_img)
    
    # (c) Warp image and compute difference
    h, w = template.shape
    aligned_img = cv.warpPerspective(test_img, H, (w, h))
    
    # Optional: Match mean brightness & contrast of aligned_img to template 
    # (Only over the non-black warped region so borders don't skew the math)
    valid_mask = aligned_img > 0
    mean_t, std_t = cv.meanStdDev(template, mask=valid_mask.astype(np.uint8))
    mean_a, std_a = cv.meanStdDev(aligned_img, mask=valid_mask.astype(np.uint8))
    
    aligned_norm = np.clip((aligned_img - mean_a) * (std_t / (std_a + 1e-6)) + mean_t, 0, 255).astype(np.uint8)
    aligned_norm[~valid_mask] = 0
    
    # Compute raw absolute difference (NO binary threshold!)
    diff_img = cv.absdiff(template, aligned_norm)
    
    # Save the raw grayscale difference image (d)
    cv.imwrite(f'diff_{img_name}', diff_img)
    
    print(f"Processed {img_name}")