

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.118.0-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![U-Net](https://img.shields.io/badge/Model-U--Net-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly)
![Xarray](https://img.shields.io/badge/Xarray-Scientific%20Data-1679C1)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.3-11557C)
![Pillow](https://img.shields.io/badge/Pillow-11.2.1-3776AB)
![Satellite Data](https://img.shields.io/badge/Data-Satellite%20Data-2E8B57)
![CMEMS](https://img.shields.io/badge/Data-CMEMS-0066CC)
![NetCDF](https://img.shields.io/badge/Data-NetCDF-4B8BBE)
![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-181717?logo=git)
![License](https://img.shields.io/badge/License-MIT-green)

# AEOLUS Ocean Subsurface Temperature Prediction

![](.assets\image.png)

This Project implements a deep learning pipeline to predict ocean subsurface temperatures across multiple depth levels using surface ocean physics data and scatterometer wind observations.

---

## 1. Input Datasets & Variables

### Ocean Physics Reanalysis Data (`0.083°` Resolution)
* **`thetao`**: Sea Water Potential Temperature (°C)
* **`so`**: Sea Water Salinity
* **`uo` / `vo`**: Eastward & Northward Sea Water Velocity (Surface Currents)
* **`zos`**: Sea Surface Height
* **`mlotst`**: Mixed Layer Thickness
* **`bottomT`**: Sea Floor Potential Temperature

### Scatterometer Wind Observation Data (`0.25°` / `20km` Resolution)
* **`eastward_wind`**: 10m Stress-Equivalent Eastward Wind Component ($U$)
* **`northward_wind`**: 10m Stress-Equivalent Northward Wind Component ($V$)

---


The model interpolates and predicts subsurface temperature (`thetao`) across 15 standard depth levels:
* **Standard Depths (m):** `0`, `5`, `10`, `20`, `30`, `50`, `75`, `100`, `125`, `150`, `200`, `300`, `500`, `700`, `1000`

---

## Model Architecture & Pipeline Highlights

1. **Input Preprocessing & Alignment:**
   * Resampling and coordinate alignment between grid resolutions (`0.083°` vs `0.25°`).
   * Adding normalized latitude/longitude channels (**CoordConv**) for location-aware spatial feature extraction.
   * Explicit land-sea mask integration to restrict loss and inference to ocean areas.

2. **Model Network:**
   * **Residual Attention U-Net (CBAM):** Features Channel and Spatial Attention modules within convolutional blocks to improve geospatial pattern recognition.
   * **Depth-Residual Head:** Predicts deeper temperature layers as residuals relative to Sea Surface Temperature (SST).

3. **Loss Function & Training:**
   * **Depth-Weighted Masked Loss:** Applies specialized weighting across depth levels to handle high variance in shallow vs. deep waters.
   * Gradient clipping and best-checkpoint saving mechanisms.

4. **Evaluation & Output:**
   * **Skill Metrics:** Per-depth evaluation using RMSE, MAE, and $R^2$ scores.
   * **Visualizations:** 2D horizontal depth slices, vertical profile plots (Temperature vs. Depth), and 3D ocean cuboid visualizations.
   * **Export:** Formatted predictions saved as NetCDF files.






