# SAE - Star reduction
#
# Groupe 2 :
# - AMEDRO Louis (Osiris-Sio)
# - HERBAUX Jules (Lirei159)
# - PACE--BOULNOIS Lysandre (NovaChocolat)
#
# =========================================================================================
# Used methods :
#
# 1. ADAPTATIVE THRESHOLD (cv.adaptiveThreshold) : stars local detection, efficient
#    even on light areas like galaxies core.
#
# 2. MORPHOLOGICAL OPENING (cv.morphologyEx) : mask cleaning to delete numeric
#    noise (isolated pixels) before treatment.
#
# 3. DILATION (cv.dilate) : mask expansion to cover colored halos.
#
# 4. INPAINTING (cv.inpaint) : Smart reconstruction of masked zones to use
#    the surroundings (sky / galaxy).
#
# 5. ALPHA BLENDING (Fusion) : weighted mix to lower the stars brightness
#    without deleting the full image.
# =========================================================================================

from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

# =================================================================
# CONFIGURATION VARIABLES
# =================================================================
FITS_FILE = "examples/m31_star.fits"

# Preventive erosion settings (lower peaks of light)
IMAGE_EROSION_SIZE = 3  # 3x3 zone
IMAGE_EROSION_ITER = 1  # Iteration

# Mask settings (Detection) :
MASK_BLOCK = 31
MASK_C = -2  # High sensitivity to catch weaker stars
OPENING_KERNEL_SIZE = 3  # Mask cleaning
MASK_DILATE_ITER = 3  # Orange/white halos cover

# Inpainting & Final Rendering settings :
INPAINT_RADIUS = 5  # Rayon de reconstruction
REDUCTION_ALPHA = 0.6  # Reduction intensity (0.6 = 60% star reduction)
BLUR_SIZE = 15  # Transition blur (for fusion)
# =================================================================

# 1. Creating output directory
if not os.path.exists("./results"):
    os.makedirs("./results")

# 2. Opening and reading FITS file
hdul = fits.open(FITS_FILE)
data = hdul[0].data

# 3. Preparation and save ORIGINAL image
# Normalization of FITS sata (0.0 to 1.0)
data_norm = (data - data.min()) / (data.max() - data.min())

if data.ndim == 3:
    if data.shape[0] == 3:  # adjusting axes if needed
        data_norm = np.transpose(data_norm, (1, 2, 0))
    plt.imsave("./results/original.png", data_norm)
    # Conversion to uint8 BGR for OpenCV
    image = (data_norm * 255).astype("uint8")
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
else:
    plt.imsave("./results/original.png", data_norm, cmap="gray")
    image = (data_norm * 255).astype("uint8")

##### Phase 1 : Full image erosion
# We slightly lower all sparkly points intensity
kernel_img = np.ones((IMAGE_EROSION_SIZE, IMAGE_EROSION_SIZE), np.uint8)
image_eroded_step1 = cv.erode(image, kernel_img, iterations=IMAGE_EROSION_ITER)

## step A : Creating and cleaning the star mask
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
mask = cv.adaptiveThreshold(
    gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, MASK_BLOCK, MASK_C
)

# Opening to delete the mask parasital noise
kernel_m = np.ones((OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE), np.uint8)
mask_cleaned = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_m)

# Dilating to encompass halos aorund the stars
mask_dilated = cv.dilate(mask_cleaned, kernel_m, iterations=MASK_DILATE_ITER)

#### PHASE 2 : Inpainting (Restoration for the ERODED image)
# Creating an image where stars are fully deleted
print("Calcul de l'Inpainting...")
eroded_final = cv.inpaint(
    image_eroded_step1, mask_dilated, INPAINT_RADIUS, cv.INPAINT_TELEA
)

# Sauving intermediate results (0 stars)
cv.imwrite("./results/eroded.png", eroded_final)

##### Phase 3 : Final Fusion (Alpha Blending for reduction)
# Mask border softening for a natural fusion
mask_blurred = cv.GaussianBlur(mask_dilated, (BLUR_SIZE, BLUR_SIZE), 0)
M = mask_blurred.astype(np.float32) / 255.0

if image.ndim == 3:
    M = np.stack([M] * 3, axis=-1)

# Transition to float for fusion calculation
Ioriginal = image.astype(np.float32)
Ieroded = eroded_final.astype(np.float32)

# Reduction application (Compromise between the original and the empty)
# Formula : M * strenght * Empty_Image + (1 - M * Strenght) * Beginning_Image
final_image_float = (M * REDUCTION_ALPHA * Ieroded) + (
    1.0 - (M * REDUCTION_ALPHA)
) * Ioriginal
final_image = np.clip(final_image_float, 0, 255).astype(np.uint8)

# 4. Results final saving
cv.imwrite("./results/star_mask.png", mask_dilated)
cv.imwrite("./results/final_phase3.png", final_image)

hdul.close()
print("Terminé ! Les 4 fichiers sont disponibles dans le dossier ./results/")
