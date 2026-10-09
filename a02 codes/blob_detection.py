import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

# 1. Load the image exactly as requested
im = cv.imread('a2images/the_berry_farms_sunflower_field.jpeg', cv.IMREAD_REDUCED_COLOR_4)

# Convert to grayscale and normalize for accurate LoG calculations
gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
gray = gray / 255.0 

# 2. Define scale-space parameters
# You will need to tune these based on the actual size of the sunflowers in the image
sigma_min = 2.0
sigma_max = 15.0
k = 1.25 # Scale multiplier

sigmas = []
log_responses = []

sigma = sigma_min
while sigma <= sigma_max:
    sigmas.append(sigma)
    
    # Calculate kernel size based on sigma (ensure it is odd)
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    
    # Apply Gaussian Blur
    blurred = cv.GaussianBlur(gray, (ksize, ksize), sigma)
    
    # Apply Laplacian (cv.CV_64F prevents overflow/underflow issues)
    laplacian = cv.Laplacian(blurred, cv.CV_64F, ksize=3)
    
    # Scale normalization: multiply by sigma^2 to maintain response magnitude across scales
    normalized_log = laplacian * (sigma ** 2)
    log_responses.append(normalized_log)
    
    sigma *= k

# Stack responses into a 3D volume (y, x, scale)
log_volume = np.stack(log_responses, axis=-1)

# Take the absolute value to detect both dark and bright blobs (sunflower centers are often dark)
log_volume = np.abs(log_volume)

# 3. Scale-space extrema detection
# Use a 3D max filter to find local maxima across (x, y) and the scale dimension
neighborhood_size = 3
local_max = maximum_filter(log_volume, size=neighborhood_size)

# Extrema are located where the raw volume matches the local maximum
extrema_mask = (log_volume == local_max)

# Apply a threshold to discard weak responses (noise)
# Tune the 0.1 multiplier based on how many false positives you get
threshold = 0.1 * np.max(log_volume) 
threshold_mask = (log_volume > threshold)

# Combine masks to get final blob locations
blob_mask = extrema_mask & threshold_mask
y_coords, x_coords, scale_indices = np.where(blob_mask)

# 4. Draw circles and extract parameters
output_im = im.copy()
detected_blobs = [] 

for y, x, s_idx in zip(y_coords, x_coords, scale_indices):
    s = sigmas[s_idx]
    r = int(np.sqrt(2) * s) # Radius calculation for LoG blobs
    detected_blobs.append({'x': x, 'y': y, 'radius': r, 'sigma': s})
    
    # Draw green circle
    cv.circle(output_im, (x, y), r, (0, 255, 0), 2)

# 5. Sort to find the largest circles
detected_blobs.sort(key=lambda b: b['radius'], reverse=True)

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(cv.cvtColor(output_im, cv.COLOR_BGR2RGB))
plt.title('Detected Sunflowers')
plt.axis('off')
plt.show()

# Print the required report parameters
print(f"Range of sigma values used: {sigma_min} to {sigmas[-1]:.2f}")
print("\nParameters of the largest circles:")
for i, blob in enumerate(detected_blobs[:5]): 
    print(f"  Rank {i+1}: Center = ({blob['x']}, {blob['y']}), Radius = {blob['radius']}, Sigma = {blob['sigma']:.2f}")