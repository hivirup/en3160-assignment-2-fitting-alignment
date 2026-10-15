import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1. Load your chosen images (Replace these with your file paths)
target_img = cv.imread('a2images/brick_wall.jpg')
insert_img = cv.imread('a2images/poster.jpg')

# Resize target for easier viewing if it's too large (optional)
# target_img = cv.resize(target_img, (1000, 800))

# 2. Get dimensions of the image to insert
h_ins, w_ins = insert_img.shape[:2]
# Source points: The 4 corners of the insert image (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
src_pts = np.array([[0, 0], [w_ins - 1, 0], [w_ins - 1, h_ins - 1], [0, h_ins - 1]], dtype=np.float32)

# 3. Interactive Point Selection
dst_pts = []
window_name = "Click 4 points (Top-Left, Top-Right, Bottom-Right, Bottom-Left)"

def select_points(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN and len(dst_pts) < 4:
        dst_pts.append([x, y])
        cv.circle(target_img_copy, (x, y), 20, (0, 255, 0), -1) # Increased circle size for high-res visibility
        cv.imshow(window_name, target_img_copy)

target_img_copy = target_img.copy()

# Add this line to make the window resizable to fit your screen
cv.namedWindow(window_name, cv.WINDOW_NORMAL) 

cv.imshow(window_name, target_img_copy)
cv.setMouseCallback(window_name, select_points)

print("Click 4 corners on the image in this order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")
print("Press any key after selecting the 4 points.")
cv.waitKey(0)
cv.destroyAllWindows()

# Convert clicked points to numpy array
dst_pts = np.array(dst_pts, dtype=np.float32)

if len(dst_pts) == 4:
    # 4. Compute Homography
    H, _ = cv.findHomography(src_pts, dst_pts)

    # 5. Warp the insert image to fit the target perspective
    h_tgt, w_tgt = target_img.shape[:2]
    warped_insert = cv.warpPerspective(insert_img, H, (w_tgt, h_tgt))

    # 6. Blend the images
    # Create a mask of the warped image
    mask = np.zeros((h_tgt, w_tgt), dtype=np.uint8)
    cv.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)
    
    # Invert mask to black out the target area in the background
    mask_inv = cv.bitwise_not(mask)
    background = cv.bitwise_and(target_img, target_img, mask=mask_inv)
    
    # Add the warped image to the blacked-out area
    final_result = cv.add(background, warped_insert)

    # Save and display
    cv.imwrite('homography_result_1.jpg', final_result)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cv.cvtColor(final_result, cv.COLOR_BGR2RGB))
    plt.title('Superimposed Image')
    plt.axis('off')
    plt.show()
else:
    print("Error: You must click exactly 4 points.")