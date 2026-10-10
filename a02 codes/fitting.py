import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- 1. Generate Noisy Data (From Listing 1) ---
N = 100
half_n = N // 2
r = 10
x0_gt, y0_gt = 2, 3 # Center
s = r / 16
t = np.random.uniform(0, 2*np.pi, half_n)
n = s * np.random.randn(half_n)
x, y = x0_gt + (r + n)*np.cos(t), y0_gt + (r + n)*np.sin(t)
X_circ = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

s = 1.
m, b = -1, 2
x = np.linspace(-12, 12, half_n)
y = m*x + b + s*np.random.randn(half_n)
X_line = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

X = np.vstack((X_circ, X_line)) # All points

# --- 2. RANSAC Line Fitting ---
def fit_line_tls(points):
    # Total Least Squares line fitting using SVD
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered)
    a, b = vh[-1] # Normal vector
    d = -(a * centroid[0] + b * centroid[1])
    return a, b, d

best_line_inliers = []
best_line_sample = None
best_line_model = None
line_threshold = 1.5 

for _ in range(1000):
    # Sample 2 points
    idx = np.random.choice(X.shape[0], 2, replace=False)
    p1, p2 = X[idx]
    
    # Calculate line normal [a,b] and d
    dir_vec = p2 - p1
    normal = np.array([-dir_vec[1], dir_vec[0]])
    norm = np.linalg.norm(normal)
    if norm == 0: continue
    a, b = normal / norm # Enforce ||[a,b].T|| = 1
    d = -(a * p1[0] + b * p1[1])
    
    # Calculate errors (normal distance)
    distances = np.abs(X[:, 0]*a + X[:, 1]*b + d)
    inliers = np.where(distances < line_threshold)[0]
    
    if len(inliers) > len(best_line_inliers):
        best_line_inliers = inliers
        best_line_sample = X[idx]
        best_line_model = (a, b, d)

# Re-fit line using all inliers (Total Least Squares)
X_line_inliers = X[best_line_inliers]
a_opt, b_opt, d_opt = fit_line_tls(X_line_inliers)

# --- 3. Remove Line Inliers and RANSAC Circle Fitting ---
remnant_mask = np.ones(X.shape[0], dtype=bool)
remnant_mask[best_line_inliers] = False
X_remnant = X[remnant_mask]

def fit_circle_3pts(p1, p2, p3):
    # Algebraic circle fit for 3 points
    temp = p2[0]**2 + p2[1]**2
    bc = (p1[0]**2 + p1[1]**2 - temp) / 2
    cd = (temp - p3[0]**2 - p3[1]**2) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    if abs(det) < 1e-6: return None
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0])*cd - (p2[0] - p3[0])*bc) / det
    r = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return cx, cy, r

best_circ_inliers = []
best_circ_sample = None
best_circ_model = None
circ_threshold = 1.5

for _ in range(1000):
    # Sample 3 points from remnant
    idx = np.random.choice(X_remnant.shape[0], 3, replace=False)
    pts = X_remnant[idx]
    
    model = fit_circle_3pts(pts[0], pts[1], pts[2])
    if model is None: continue
    cx, cy, r_est = model
    
    # Calculate radial errors
    distances = np.sqrt((X_remnant[:, 0] - cx)**2 + (X_remnant[:, 1] - cy)**2)
    errors = np.abs(distances - r_est)
    inliers = np.where(errors < circ_threshold)[0]
    
    if len(inliers) > len(best_circ_inliers):
        best_circ_inliers = inliers
        best_circ_sample = pts
        best_circ_model = model

# Re-fit circle using scipy.minimize on all inliers
X_circ_inliers = X_remnant[best_circ_inliers]

def calc_circ_error(params, points):
    cx, cy, r_est = params
    distances = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
    return np.sum((distances - r_est)**2)

# Initial guess from RANSAC
opt_res = minimize(calc_circ_error, best_circ_model, args=(X_circ_inliers,))
cx_opt, cy_opt, r_opt = opt_res.x

# --- 4. Plotting (Part C) ---
fig, ax = plt.subplots(1,1, figsize=(8,8))

# Background points
ax.scatter(X[:,0], X[:,1], c='cyan', label='All points', s=15)

# Line Plotting
ax.scatter(X_line_inliers[:,0], X_line_inliers[:,1], c='yellow', label='Line inliers', s=20)
ax.scatter(best_line_sample[:,0], best_line_sample[:,1], c='black', marker='x', label='Best sample for line', s=50)

x_vals = np.array(ax.get_xlim())
y_vals = -(a_opt * x_vals + d_opt) / b_opt
ax.plot(x_vals, y_vals, color='purple', label='RANSAC line')

# Circle Plotting
ax.scatter(X_circ_inliers[:,0], X_circ_inliers[:,1], c='magenta', label='Circle inliers', s=20)
ax.scatter(best_circ_sample[:,0], best_circ_sample[:,1], c='red', label='Best sample for circle', s=30)

circle_ransac = plt.Circle((cx_opt, cy_opt), r_opt, color='blue', fill=False, label='RANSAC circle')
ax.add_patch(circle_ransac)
ax.plot(cx_opt, cy_opt, '+', color='blue')

# Ground Truth Overlay
circle_gt = plt.Circle((x0_gt,y0_gt), r, color='g', fill=False, label='Ground_truth_circle')
ax.add_patch(circle_gt)
y_gt = m*x_vals + b
ax.plot(x_vals, y_gt, color='g', linestyle='--', label='Ground_truth_line')

plt.legend()
plt.axis('equal')
plt.savefig('ransac_results.jpeg', bbox_inches='tight', pad_inches=0.1)
plt.show()