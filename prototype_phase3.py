# SAE - Star reduction
#
# Groupe 2 :
# - AMEDRO Louis (Osiris-Sio)
# - HERBAUX Jules (Lirei159)
# - PACE--BOULNOIS Lysandre (NovaChocolat)
#
# =========================================================================================
# Résumé des méthodes utilisées :
#
# 1. SEUILLAGE ADAPTATIF (cv.adaptiveThreshold) : Détection locale des étoiles, efficace
#    même sur les zones lumineuses comme le coeur de la galaxie.
#
# 2. OPENING MORPHOLOGIQUE (cv.morphologyEx) : Nettoyage du masque pour supprimer le bruit
#    numérique (pixels isolés) avant le traitement.
#
# 3. DILATATION (cv.dilate) : Élargissement du masque pour couvrir les halos colorés.
#
# 4. INPAINTING (cv.inpaint) : Reconstruction intelligente des zones masquées en utilisant
#    les textures environnantes (fond du ciel / galaxie).
#
# 5. ALPHA BLENDING (Fusion) : Mélange pondéré pour réduire la luminosité des étoiles
#    sans les supprimer totalement dans l'image finale.
# =========================================================================================

from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

# =================================================================
# VARIABLES DE CONFIGURATION
# =================================================================
FITS_FILE = "examples/m31_star.fits"

# Paramètres Érosion préventive (affaiblit les pics de lumière)
IMAGE_EROSION_SIZE = 3  # Zone de 3x3
IMAGE_EROSION_ITER = 1  # Itération

# Paramètres Masque (Détection) :
MASK_BLOCK = 31
MASK_C = -2  # Sensibilité élevée pour capturer les étoiles faibles
OPENING_KERNEL_SIZE = 3  # Nettoyage du masque
MASK_DILATE_ITER = 3  # Couverture des halos oranges/blancs

# Paramètres Inpainting & Rendu Final :
INPAINT_RADIUS = 5  # Rayon de reconstruction
REDUCTION_ALPHA = 0.6  # Intensité de la réduction (0.6 = étoiles atténuées de 60%)
BLUR_SIZE = 15  # Flou de transition pour la fusion
# =================================================================

# 1. Création du dossier de sortie
if not os.path.exists("./results"):
    os.makedirs("./results")

# 2. Ouverture et lecture du fichier FITS
hdul = fits.open(FITS_FILE)
data = hdul[0].data

# 3. Préparation et sauvegarde de l'image ORIGINAL
# Normalisation des données FITS (0.0 à 1.0)
data_norm = (data - data.min()) / (data.max() - data.min())

if data.ndim == 3:
    if data.shape[0] == 3:  # Ajustement des axes si nécessaire
        data_norm = np.transpose(data_norm, (1, 2, 0))
    plt.imsave("./results/original.png", data_norm)
    # Conversion en uint8 BGR pour OpenCV
    image = (data_norm * 255).astype("uint8")
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
else:
    plt.imsave("./results/original.png", data_norm, cmap="gray")
    image = (data_norm * 255).astype("uint8")

##### Phase 1 : Érosion de l'image complète
# On réduit légèrement l'intensité de tous les points brillants
kernel_img = np.ones((IMAGE_EROSION_SIZE, IMAGE_EROSION_SIZE), np.uint8)
image_eroded_step1 = cv.erode(image, kernel_img, iterations=IMAGE_EROSION_ITER)

## étape A : Création et nettoyage du masque d'étoiles
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
mask = cv.adaptiveThreshold(
    gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, MASK_BLOCK, MASK_C
)

# Opening pour supprimer le bruit parasite du masque
kernel_m = np.ones((OPENING_KERNEL_SIZE, OPENING_KERNEL_SIZE), np.uint8)
mask_cleaned = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_m)

# Dilatation pour englober les halos autour des étoiles
mask_dilated = cv.dilate(mask_cleaned, kernel_m, iterations=MASK_DILATE_ITER)

#### PHASE 2 : Inpainting (Restauration pour l'image ERODED)
# Création d'une image où les étoiles sont totalement supprimées
print("Calcul de l'Inpainting...")
eroded_final = cv.inpaint(
    image_eroded_step1, mask_dilated, INPAINT_RADIUS, cv.INPAINT_TELEA
)

# Sauvegarde du résultat intermédiaire (Zéro étoiles)
cv.imwrite("./results/eroded.png", eroded_final)

##### Phase 3 : Fusion Finale (Alpha Blending pour réduction)
# Adoucissement des bords du masque pour une fusion naturelle
mask_blurred = cv.GaussianBlur(mask_dilated, (BLUR_SIZE, BLUR_SIZE), 0)
M = mask_blurred.astype(np.float32) / 255.0

if image.ndim == 3:
    M = np.stack([M] * 3, axis=-1)

# Passage en flottant pour les calculs de fusion
Ioriginal = image.astype(np.float32)
Ieroded = eroded_final.astype(np.float32)

# Application de la réduction (Compromis entre l'original et le vide)
# Formule : M * Force * Image_Vide + (1 - M * Force) * Image_Originale
final_image_float = (M * REDUCTION_ALPHA * Ieroded) + (
    1.0 - (M * REDUCTION_ALPHA)
) * Ioriginal
final_image = np.clip(final_image_float, 0, 255).astype(np.uint8)

# 4. Sauvegardes finales des résultats
cv.imwrite("./results/star_mask.png", mask_dilated)
cv.imwrite("./results/final_phase3.png", final_image)

hdul.close()
print("Terminé ! Les 4 fichiers sont disponibles dans le dossier ./results/")
