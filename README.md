[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zP0O23M7)

# Project documentation

## Group 2 :

- AMEDRO Louis (Osiris-Sio)

- HERBAUX Jules (Lirei159)

- PACE--BOULNOIS Lysandre (NovaChocolat)

## Features

- **FITS files reading** : Monochrome and colored image support (RGB).
- **Graphical Interface (GUI)** : Based on PyQt6, it allows to visualize instantly settings effects.
- **Reduction algorythm** :
  1. **Erosion** : Create a "starless" image.
  2. **Masking** : Detects stars by an adaptative threshold.
  3. **Fusion** : Blend the eroded image (on the stars) and the original image (on the sky) for a natural rendering.

## Setup

### Prerequisites

Standard Python installation (3.10 or newer) Windows recommanded.

> **Note** : Avoid using Python given by MSYS2/MinGW, because it can cause compatibility problems with graphical library like PyQt6 ou OpenCV.

### Dependencies

```bash
pip install -r requirements.txt
```

### Configuration

Using a virtual environment is recommended :

```bash
python -m venv venv
source venv/bin/activate
# On Windows:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# venv\Scripts\activate
pip install -r requirements.txt
```

## Use

To launch the Application with the home menu :

```powershell
# Make sure your virtual environment is on
python launcher.py
```

Or there is an executable in the release section on GitHub.

The home screen has three modes :

1.  **Real Time Mode** : Interactive reduction of the interface (described below).
2.  **Comparison Mode** : Allows to compare two images with [MSE](https://scikit-image.org/docs/0.25.x/api/skimage.metrics.html#skimage.metrics.mean_squared_error), [SSIM](https://scikit-image.org/docs/0.25.x/api/skimage.metrics.html#skimage.metrics.structural_similarity) and a differential cloud.
3.  **Generate Images (Batch)** : Generate the following images in the wanted directory : 'original.png', 'star_mask.png', 'eroded.png' et 'final_phase3.png'

### Instructions (Real Time Mode)

1. **Visualization** : The displayed image is the real time treatment result.

2. **Adjustment** : Move the sliders in the right panel. Each modification restart the calculation and update instantly the image.
3. **Recommanded workflow** :
   - Start by adjusting **"Threshold bloc size"** and **"Threshold Constant"** to correctly isolate stars (mask).
   - Increase **"Mask blur"** to make the transition between stars invisible.
   - Finally, settle the **"Erosion"** to determine how much stars need to be lowered/erased.

### Available settings

##### Mask settings (Detection)

- **Threshold bloc (odd) :** Adjust local area size to distinguish stars from the sky.
- **Constant C :** Settle detection sensibility (lower she is, more we can detect weaker stars).
- **Cleaning (Opening) :** Erase numeric noise and small pixels isolated from the mask.
- **Dilation (Halos) :** Expands the mask to cover entirely the colored halo around stars.

##### Treatment settings

- **Inpainting (Radius) :** Define used distance to rebuild background image instead of stars.

##### Fusion settings

- **Reduction intensity (%) :** Control dosage between original image and corrected image (attenuates 60% of the star without erasing it).
- **Transition blur :** Softens mask border to make the correction integration invisible.

## FITS files example

Examples files are in the directory `examples/`. You can execute the script `launcher.py` with these files to see how they work :

- Recommanded example : `examples/m31_star.fits`
- Example 1 : `examples/HorseHead.fits` (Black and white FITS image file to test)
- Example 2 : `examples/test_M31_linear.fits` (Colored FITS image file to test)
- Example 3 : `examples/test_M31_raw.fits` (Colored FITS image file to test)
