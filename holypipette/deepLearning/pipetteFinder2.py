"""
pipette_tip.py  –  locate a pipette tip (blue dot) in a phase-contrast image.

Usage:
    python pipette_tip.py microscope_image.png
"""
import sys, cv2, numpy as np



def locate_tip_pca(edge_pts):
    """
    Robustly returns the pipette apex by:
      1. Computing principal components of all shaft edge pixels.
      2. Picking the extreme point along the major axis.
    """
    if len(edge_pts) < 10:
        raise RuntimeError("Too few edge points for PCA")

    # centre the cloud and run PCA (eig of covariance)
    pts   = edge_pts.astype(np.float32)
    mean  = pts.mean(axis=0, keepdims=True)
    cov   = np.cov((pts - mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis  = eigvecs[:, np.argmax(eigvals)]          # major component

    # project points onto that axis and take the extreme end
    proj  = (pts - mean) @ axis
    idx   = np.argmin(proj) if proj.mean() > 0 else np.argmax(proj)
    return pts[idx].astype(int)


def _dominant_angle(deg, bins=180):
    """helper: returns modal angle in degrees wrapped to [-90,90)"""
    wrapped = (deg + 90) % 180 - 90
    hist, edges = np.histogram(wrapped, bins=bins, range=(-90, 90))
    return edges[np.argmax(hist)]


def _longest_k(lines, k=2):
    lines = sorted(lines, key=lambda L: np.hypot(L[2]-L[0], L[3]-L[1]), reverse=True)
    return lines[:k]


def locate_tip_hough(gray,
                     canny_lohi=(50, 150),
                     hough_thresh=60,
                     min_len=60,
                     max_gap=15,
                     angle_tol=10):
    """
    1. Canny → Hough to get line segments for the two glass walls.
    2. Keep the two longest segments with nearly identical angle.
    3. Intersect their infinite lines and return that point.
    """
    # edges and Hough line segments
    edges = cv2.Canny(gray, *canny_lohi, apertureSize=3)
    segs  = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=hough_thresh,
                            minLineLength=min_len,
                            maxLineGap=max_gap)
    if segs is None:
        raise RuntimeError("No Hough lines detected")

    # angle of each segment
    segs  = segs[:, 0, :]                       # shape (N,4)
    angles = np.degrees(np.arctan2(segs[:,3]-segs[:,1],
                                   segs[:,2]-segs[:,0]))

    dom_ang = _dominant_angle(angles)
    # select segments within ±angle_tol of dominant
    mask  = np.abs(((angles + 90) % 180 - 90) - dom_ang) < angle_tol
    shaft = segs[mask]
    if len(shaft) < 2:
        raise RuntimeError("Could not isolate two shaft walls")

    (x1,y1,x2,y2), (u1,v1,u2,v2) = _longest_k(shaft, 2)

    # analytic intersection of two infinite lines
    A = np.array([[x2-x1, u1-u2],
                  [y2-y1, v1-v2]], dtype=float)
    b = np.array([u1-x1, v1-y1], dtype=float)
    denom = np.linalg.det(A)
    if abs(denom) < 1e-6:
        raise RuntimeError("Shaft walls appear parallel")

    t, s = np.linalg.solve(A, b)
    tip  = np.array([x1 + t*(x2-x1), y1 + t*(y2-y1)], dtype=int)
    return tip

def load_image(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(img_path)
    return gray

def get_edge_mask(gray):
    edges = cv2.Canny(gray, 40, 120)
    mask  = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    return mask

def find_edge_corners(gray, mask, max_corners=150, quality=0.01, min_dist=5):
    corners = cv2.goodFeaturesToTrack(gray,
                                      maxCorners=max_corners,
                                      qualityLevel=quality,
                                      minDistance=min_dist)
    if corners is None:
        raise RuntimeError("No corners detected")
    edge_pts = np.array([c.ravel() for c in corners
                         if mask[int(c[0][1]), int(c[0][0])] > 0])
    if edge_pts.size == 0:
        raise RuntimeError("No corners overlapped the pipette-edge mask")
    return edge_pts



def visualize(gray, edge_pts, tip_pt):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for pt in edge_pts:
        pt_int = tuple(np.round(pt).astype(int))
        cv2.circle(vis, pt_int, 2, (0, 255, 0), -1)
    tip_pt_int = tuple(np.round(tip_pt).astype(int))
    cv2.circle(vis, tip_pt_int, 8, (255, 0, 0), -1)
    cv2.imshow("Tip detection", vis)
    cv2.waitKey(0)

def main(img_path, use='pca'):
    gray      = load_image(img_path)
    mask      = get_edge_mask(gray)
    edge_pts  = find_edge_corners(gray, mask)

    if use == 'pca':
        tip_pt = locate_tip_pca(edge_pts)
    elif use == 'hough':
        tip_pt = locate_tip_hough(gray)
    else:
        raise ValueError("use must be 'pca' or 'hough'")

    print(f"Pipette tip ≈ pixel {tuple(tip_pt)}   (method: {use})")
    visualize(gray, edge_pts, tip_pt)

if __name__ == '__main__':
    IMG = r"C:\Users\sa-forest\Documents\GitHub\Neuron_Detection\data\EmoryPipetteDataBase\camera_frames\3577_1742332992.76673.webp"  # <-- Hardcoded path for testing
    main(IMG, use='pca')     # swap to 'hough' if you prefer that variant

# if __name__ == "__main__":
#     # img_path =r"C:\Users\sa-forest\Documents\GitHub\Neuron_Detection\data\EmoryPipetteDataBase\camera_frames\3248_1742332973.216286.webp"  # <-- Hardcoded path for testing
#     # img_path = r"C:\Users\sa-forest\Documents\GitHub\Neuron_Detection\data\EmoryPipetteDataBase\camera_frames\3577_1742332992.76673.webp"
#     img_path = r"C:\Users\sa-forest\Documents\GitHub\Neuron_Detection\data\DinoTrainingData\32007_1743700624.756302.webp"

#     main(img_path)
