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

# Open and read the FITS file
fits_file = "./examples/m31_star.fits"
hdul = fits.open(fits_file)

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

    # Normalisation globale pour préserver les couleurs
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype("uint8")

    # Conversion RGB vers BGR pour OpenCV
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
else:
    # Monochrome image
    plt.imsave("./results/original.png", data, cmap="gray")

    # Convert to uint8 for OpenCV
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype("uint8")

# Define a kernel for erosion
kernel = np.ones((5, 5), np.uint8)
# Perform erosion
eroded_image = cv.erode(image, kernel, iterations=1)


###### Phase 2 :


### Étape A : Création du masque d’étoiles
if len(image.shape) == 3:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
else:
    gray = image

mask = cv.adaptiveThreshold(
    gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, -20
)

### Étape B : Réduction localisée
mask_blurred = cv.GaussianBlur(mask, (5, 5), 0)

# Utilisation de float32 pour éviter les erreurs de profondeur d'image
M = mask_blurred.astype(np.float32) / 255.0

if len(image.shape) == 3:
    M = np.stack([M, M, M], axis=2)

# Conversion explicite en float32 pour le calcul
Ioriginal = image.astype(np.float32)
Ierode = eroded_image.astype(np.float32)

# Calcul de l'image finale
final_image_float = (M * Ierode) + ((1.0 - M) * Ioriginal)

# Reconversion en uint8 AVANT la sauvegarde pour éviter les Warnings
final_image = np.clip(final_image_float, 0, 255).astype(np.uint8)

### Sauvegardes finales
cv.imwrite("./results/eroded.png", eroded_image)
cv.imwrite("./results/final_phase2.png", final_image)
cv.imwrite("./results/star_mask.png", mask)

# Close the file
hdul.close()
