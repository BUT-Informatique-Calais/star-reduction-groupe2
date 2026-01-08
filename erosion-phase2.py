# SAE - Star reduction
#
# Groupe 2 :
# - AMEDRO Louis (Osiris-Sio)
# - HERBAUX Jules (Lirei159)
# - PACE--BOULNOIS Lysandre (NovaChocolat)

from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

# =================================================================
# CONFIGURATION VARIABLES
# =================================================================
import sys
if len(sys.argv) > 1:
    FITS_FILE = sys.argv[1]
else:
    print("Veuillez choisir une image FITS.")
    # Use input to ask for file if not provided
    user_file = input("Entrez le chemin du fichier FITS (par défaut ./examples/m31_star.fits) : ").strip()
    if user_file:
        FITS_FILE = user_file
    else:
        FITS_FILE = "./examples/m31_star.fits"

# Erosion Parameters

EROSION_SIZE = 3  # Kernel size (3x3, 5x5, etc.)
EROSION_ITER = 4  # Number of erosion iterations (more = smaller stars)

# Mask Parameters (Step A)
MASK_BLOCK_SIZE = 21  # Neighborhood size (must be odd: 11, 21, 31...)
MASK_C_VALUE = -10  # Constant C (lower value = more stars detected)

# Blur Parameters (Step B)
BLUR_SIZE = 15  # Softness of the transition (must be odd: 5, 9, 15...)
# =================================================================

# Open and read the FITS file
hdul = fits.open(FITS_FILE)

# Display information about the file
hdul.info()

# Access the data from the primary HDU
data = hdul[0].data

# Access header information
header = hdul[0].header

# Handle both monochrome and color images
if data.ndim == 3:
    # Color image - need to transpose to (height, width, channels)
    if data.shape[0] == 3:  # If channels are first: (3, height, width)
        data = np.transpose(data, (1, 2, 0))
    # If already (height, width, 3), no change needed

    # Normalize the entire image to [0, 1] for matplotlib
    data_normalized = (data - data.min()) / (data.max() - data.min())

    # Save the data as a png image (no cmap for color images)
    plt.imsave("./results/original.png", data_normalized)

    # Global normalization to preserve colors
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype("uint8")

    # RGB to BGR conversion for OpenCV
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
else:
    # Monochrome image
    plt.imsave("./results/original.png", data, cmap="gray")

    # Convert to uint8 for OpenCV
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype("uint8")

# Define a kernel for erosion
kernel = np.ones((EROSION_SIZE, EROSION_SIZE), np.uint8)
# Perform erosion
eroded_image = cv.erode(image, kernel, iterations=EROSION_ITER)


###### Phase 2:


### Step A: Create Star Mask
if len(image.shape) == 3:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
else:
    gray = image

# Threshold
mask = cv.adaptiveThreshold(
    gray,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    MASK_BLOCK_SIZE,  # Neighborhood size (e.g. 21x21)
    MASK_C_VALUE,  # Keep only pixels significantly brighter than average
)

### Step B: Localized Reduction
# Blurred mask using Gaussian kernel
mask_blurred = cv.GaussianBlur(mask, (BLUR_SIZE, BLUR_SIZE), 0)

# Use float32 to avoid image depth errors
M = mask_blurred.astype(np.float32) / 255.0

if len(image.shape) == 3:
    M = np.stack([M, M, M], axis=2)

# Explicit conversion to float32 for calculation
Ioriginal = image.astype(np.float32)
Ierode = eroded_image.astype(np.float32)

# Calculate final image
final_image_float = (M * Ierode) + ((1.0 - M) * Ioriginal)

# Reconvert to uint8 BEFORE saving to avoid warnings
final_image = np.clip(final_image_float, 0, 255).astype(np.uint8)

### Final saves
cv.imwrite("./results/eroded.png", eroded_image)
cv.imwrite("./results/final_phase2.png", final_image)
cv.imwrite("./results/star_mask.png", mask)

# Close the file
hdul.close()
