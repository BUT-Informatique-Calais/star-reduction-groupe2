[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zP0O23M7)

# Documentation du projet

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

### Dépendances

```bash
pip install -r requirements.txt
```

### Configuration

Il est fortement conseillé d'utiliser un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate
# Sur Windows:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

Pour lancer l'application avec le menu d'accueil :

```powershell
# Assurez-vous que votre environnement virtuel est activé
python launcher.py
```

Ou vous avez un exécutable dans l'espace "releases" sur GitHub.

L'écran d'accueil vous propose trois modes :

1.  **Mode Temps Réel** : L'interface de réduction interactive (décrite ci-dessous).
2.  **Mode Comparaison** : Permet de comparer deux images grâce au MSE, SSIM et à un nuage différentiel.
3.  **Générer Images (Batch)** : Génère les images suivantes dans le dossier souhaité : 'original.png', 'star_mask.png', 'eroded.png' et 'final_phase3.png'

### Instructions (Mode Temps Réel)

1. **Visualisation** : L'image affichée est le résultat du traitement en temps réel.

2. **Ajustement** : Déplacez les curseurs (sliders) dans le panneau de droite. Chaque modification relance le calcul et met à jour l'image instantanément.
3. **Workflow recommandé** :
   - Commencez par ajuster **"Taille de bloc seuil"** et **"Constante seuil"** pour isoler correctement les étoiles (le masque).
   - Augmentez **"Flou du masque"** pour rendre la transition autour des étoiles invisible.
   - Enfin, réglez l'**"Érosion"** pour déterminer à quel point les étoiles doivent être réduites/effacées.

### Paramètres disponibles

##### Paramètres de Masque (Détection)

- **Seuil Bloc (impair) :** Ajuste la taille de la zone locale pour distinguer les étoiles du fond du ciel.
- **Constante C :** Règle la sensibilité de la détection (plus elle est basse, plus on détecte d'étoiles faibles).
- **Nettoyage (Ouverture) :** Supprime le bruit numérique et les petits pixels isolés du masque.
- **Dilatation (Halos) :** Élargit le masque pour couvrir entièrement le halo coloré autour des étoiles.

##### Paramètres de Traitement

- **Inpainting (Rayon) :** Définit la distance utilisée pour reconstruire le fond de l'image à la place des étoiles.

##### Paramètres de Fusion

- **Intensité Réduction (%) :** Contrôle le dosage entre l'image originale et l'image corrigée (60% atténue l'étoile sans l'effacer).
- **Flou Transition :** Adoucit les bords du masque pour rendre l'intégration des corrections invisible.

## Exemples de fichier FITS

Les fichiers d’exemple sont situés dans le répertoire `examples/`. Vous pouvez exécuter le script `launcher.py` avec ces fichiers pour voir comment ils fonctionnent :

- Exemple Recommandé : `examples/m31_star.fits`
- Exemple 1 : `examples/HorseHead.fits` (Fichier image noir et blanc FITS pour test)
- Exemple 2 : `examples/test_M31_linear.fits` (Fichier image FITS couleur pour test)
- Exemple 3 : `examples/test_M31_raw.fits` (Fichier image FITS couleur pour test)
