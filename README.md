[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zP0O23M7)

# Project Documentation

## Groupe 2 :

- AMEDRO Louis (Osiris-Sio)

- HERBAUX Jules (Lirei159)

- PACE--BOULNOIS Lysandre (NovaChocolat)

## Fonctionnalités
- **Lecture de fichiers FITS** : Support des images monochromes et couleurs (RGB).
- **Interface Graphique (GUI)** : Basée sur PyQt6, elle permet de visualiser instantanément l'effet des réglages.
- **Algorithme de réduction** :
  1. **Érosion** : Crée une version "sans étoiles" de l'image.
  2. **Masquage** : Détecte les étoiles via un seuillage adaptatif.
  3. **Fusion** : Mélange l'image érodée (sur les étoiles) et l'image originale (sur le fond du ciel) pour un rendu naturel.

## Installation

### Pré-requis
Une installation standard de Python (3.10 ou plus récent) sous Windows est recommandée.

> **Note** : Évitez d'utiliser le Python fourni par MSYS2/MinGW, car il peut poser des problèmes de compatibilité avec les bibliothèques graphiques comme PyQt6 ou OpenCV.

### Configuration
Il est fortement conseillé d'utiliser un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate
# On Windows:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies

```bash
pip install -r requirements.txt
```

## Utilisation

Pour lancer l'interface graphique :

```powershell
# Assurez-vous que votre environnement virtuel est activé
python gui_star_reduction.py
```

Par défaut, l'application charge l'image `examples/m31_star.fits`. Une fois l'interface lancée, vous verrez l'image traitée au centre et un panneau de contrôle sur la droite.

### Instructions
1. **Visualisation** : L'image affichée est le résultat du traitement en temps réel.
2. **Ajustement** : Déplacez les curseurs (sliders) dans le panneau de droite. Chaque modification relance le calcul et met à jour l'image instantanément.
3. **Workflow recommandé** :
   - Commencez par ajuster **"Taille de bloc seuil"** et **"Constante seuil"** pour isoler correctement les étoiles (le masque).
   - Augmentez **"Flou du masque"** pour rendre la transition autour des étoiles invisible.
   - Enfin, réglez l'**"Érosion"** pour déterminer à quel point les étoiles doivent être réduites/effacées.

### Paramètres disponibles
- **Taille du noyau d'érosion** : Définit la force de la réduction brute des étoiles.
- **Itérations d'érosion** : Nombre de passes d'érosion (réduit davantage les grosses étoiles).
- **Taille de bloc seuil** : Ajuste la sensibilité de la détection des étoiles (zones locales).
- **Constante seuil (C)** : Affine la détection (valeur plus faible = plus d'étoiles détectées).
- **Flou du masque** : Adoucit les bords du masque pour éviter les artefacts de coupure autour des étoiles.


## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Examples files

Example files are located in the `examples/` directory. You can run the scripts with these files to see how they work.

- Example 1 : `examples/HorseHead.fits` (Black and whiteFITS image file for testing)
- Example 2 : `examples/test_M31_linear.fits` (Color FITS image file for testing)
- Example 3 : `examples/test_M31_raw.fits` (Color FITS image file for testing)
