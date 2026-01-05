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

    # Normalisation globale pour préserver les couleurs (balance des blancs)
    # OpenCV utilise BGR ou RGB, ici on garde les canaux tels quels mais normalisés ensemble

    image = (
        (data - data.min()) / (data.max() - data.min()) * 255
    ).astype("uint8")
    
    # Conversion RGB vers BGR pour OpenCV
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
else:
    # Monochrome image
    plt.imsave("./results/original.png", data, cmap="gray")

    # Convert to uint8 for OpenCV
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype("uint8")



# Définition du noyau pour l'érosion
# Une taille de 5x5 a été choisie pour une réduction plus marquée des étoiles
kernel = np.ones((5, 5), np.uint8)
# Application de l'érosion (Phase 1)
eroded_image = cv.erode(image, kernel, iterations=1)

# Étape A : Création du masque d’étoiles (Phase 2)
# Conversion en niveaux de gris pour la détection du masque
if len(image.shape) == 3:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
else:
    gray = image

# Utilisation d'un seuillage adaptatif pour isoler les étoiles
# Le but est d'obtenir un masque binaire : blanc (255) pour les étoiles, noir (0) pour le fond.
# Paramètres :
# - 255 : Valeur max (blanc)
# - ADAPTIVE_THRESH_GAUSSIAN_C : Méthode de calcul du seuil local
# - THRESH_BINARY : Inversion non nécessaire ici grâce à la correction de la constante
# - 11 : Taille du voisinage
# - -2 : Constante "C". Une valeur négative signifie que le seuil est SUPERIEUR à la moyenne locale.
#        Cela permet de ne capturer que les pixels nettement plus brillants que le fond (les étoiles).
#        -2 est plus sensible que -20, capturant ainsi les étoiles faibles.
mask = cv.adaptiveThreshold(
    gray, 
    255, 
    cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv.THRESH_BINARY, 
    11, 
    -2
)

# Étape B : Réduction localisée (Phase 2)
# 1. Nous avons l'image érodée (eroded_image) et l'image originale (image/Ioriginal).
# 2. Adoucissement du masque pour des transitions naturelles (évite les artefacts de coupure)
mask_blurred = cv.GaussianBlur(mask, (5, 5), 0)

# Normaliser le masque entre 0 et 1 pour les calculs (float)
M = mask_blurred.astype(float) / 255.0

# Si l'image est en couleur, nous avons besoin que le masque ait 3 canaux pour la multiplication
if len(image.shape) == 3:
    M = np.stack([M, M, M], axis=2)

# Convertir les images en float pour le calcul
Ioriginal = image.astype(float)
Ierode = eroded_image.astype(float)

# 3. Calcul de l'image finale
# Formule : Ifinal = (M * Ierode) + ((1 - M) * Ioriginal)
final_image_float = (M * Ierode) + ((1.0 - M) * Ioriginal)


# Reconversion en uint8
final_image = np.clip(final_image_float, 0, 255).astype("uint8")

# Ajout d'une étape de netteté (Sharpening) pour compenser le flou
# Création d'un noyau de convolution pour accentuer les détails
sharpen_kernel = np.array([[0, -1, 0], 
                           [-1, 5, -1], 
                           [0, -1, 0]])
# Application du filtre de netteté sur l'image finale
final_image = cv.filter2D(final_image, -1, sharpen_kernel)

# Save the eroded image
cv.imwrite("./results/eroded.png", eroded_image)
# Sauvegarde de l'image finale avec réduction localisée
cv.imwrite("./results/final_phase2.png", final_image)
# Sauvegarde du masque pour vérification
cv.imwrite("./results/star_mask.png", mask)

# Close the file
hdul.close()
