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
import os

# =========================================================================================
# résumé des méthodes utilisées :
# 1. SEUILLAGE ADAPTATIF (cv.adaptiveThreshold) : Détecte les étoiles en calculant un seuil
#    local, permettant d'isoler les points brillants même sur le fond lumineux de la galaxie.
#
# 2. DILATATION MORPHOLOGIQUE (cv.dilate) : Élargit les zones du masque pour englober
#    totalement les halos (souvent oranges/jaunes) qui entourent les coeurs d'étoiles.
#
# 3. RESTAURATION PAR INPAINTING (cv.INPAINT_TELEA) : Reconstruit l'image "eroded" en
#    remplaçant les étoiles par les textures de la galaxie environnante (méthode de Telea).
#
# 4. MÉLANGE ALPHA FLOU (Alpha Blending) : Fusionne l'original et la version restaurée
#    via un masque flouté (GaussianBlur) pour garantir des transitions invisibles.
# =========================================================================================

# =================================================================
# VARIABLES DE CONFIGURATION
# =================================================================
FITS_FILE = "./examples/m31_star.fits"

# Paramètres de détection (Masque)
MASK_BLOCK = 31  # Voisinage large pour bien englober l'étoile
MASK_C = -5  # Sensibilité stricte pour ne prendre que les étoiles
MASK_DILATE_ITER = 1  # Augmenter à 3 ou 4 pour couvrir tout le halo des étoiles

# Paramètres d'Inpainting (La nouvelle méthode "magique")
INPAINT_RADIUS = 10  # Rayon de reconstruction autour de l'étoile

# Paramètres de flou pour la fusion finale (Phase 2/3)
BLUR_SIZE = 15
# =================================================================

if not os.path.exists("./results"):
    os.makedirs("./results")

hdul = fits.open(FITS_FILE)
data = hdul[0].data

# Préparation de l'image
if data.ndim == 3:
    if data.shape[0] == 3:
        data = np.transpose(data, (1, 2, 0))
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype("uint8")
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
else:
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype("uint8")

# étape A : Création du masque de détection
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
mask = cv.adaptiveThreshold(
    gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, MASK_BLOCK, MASK_C
)

# On dilate un peu le masque pour être sûr de couvrir TOUT le halo des étoiles
kernel_dilate = np.ones((3, 3), np.uint8)
mask_dilated = cv.dilate(mask, kernel_dilate, iterations=MASK_DILATE_ITER)

# phase 1 : Inpainting (L'amélioration de l'eroded)
# Cette méthode remplace les étoiles par la texture du ciel voisin
print("Calcul de l'Inpainting (cela peut prendre quelques secondes)...")
eroded_inpainted = cv.inpaint(image, mask_dilated, INPAINT_RADIUS, cv.INPAINT_TELEA)

# 2. Résultat : ERODED (Ici, on ne voit quasiment plus que la galaxie, mais un peu flou, avec un reste des étoiles larges)
cv.imwrite("./results/eroded.png", eroded_inpainted)

# phase 2/3 : Fusion Finale
# Floutage pour la transition
mask_blurred = cv.GaussianBlur(mask_dilated, (BLUR_SIZE, BLUR_SIZE), 0)
M = mask_blurred.astype(np.float32) / 255.0

if image.ndim == 3:
    M = np.stack([M] * 3, axis=-1)

Ioriginal = image.astype(np.float32)
Ieroded = eroded_inpainted.astype(np.float32)

# Fusion : on garde l'original partout, sauf là où il y avait des étoiles
final_image_float = (M * Ieroded) + ((1.0 - M) * Ioriginal)
final_image = np.clip(final_image_float, 0, 255).astype(np.uint8)

# Sauvegardes
plt.imsave(
    "./results/original.png",
    cv.cvtColor(image, cv.COLOR_BGR2RGB) if image.ndim == 3 else image,
    cmap="gray",
)
cv.imwrite("./results/star_mask.png", mask_dilated)
cv.imwrite("./results/final_phase3.png", final_image)

hdul.close()
print("Terminé !")
