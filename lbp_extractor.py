import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp_features(image, radius=1, n_points=8):
    """
    Extract LBP features from an image.
    
    Parameters:
    - image: Input image in BGR or grayscale.
    - radius: Radius of circle (in pixels) for LBP.
    - n_points: Number of points to consider around each pixel.
    
    Returns:
    - hist: Normalized histogram of LBP features.
    """
    # If the image has 3 channels, convert it to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Compute the LBP representation of the image using the "uniform" method
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    
    # Calculate the number of bins. For uniform LBP, the number of bins is n_points + 2
    n_bins = int(n_points + 2)
    
    # Build a histogram of the LBP values
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    
    return hist
