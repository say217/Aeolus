#  AEOLUS — Ocean Subsurface Temperature Prediction

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.118.0-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![Model](https://img.shields.io/badge/Model-Residual%20Attention%20U--Net-orange)
![CBAM](https://img.shields.io/badge/Attention-CBAM-purple)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly)
![Xarray](https://img.shields.io/badge/Xarray-Scientific%20Data-1679C1)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10.3-11557C)
![Pillow](https://img.shields.io/badge/Pillow-11.2.1-3776AB)
![CMEMS](https://img.shields.io/badge/Data-CMEMS-0066CC)
![NetCDF](https://img.shields.io/badge/Data-NetCDF-4B8BBE)
![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-181717?logo=git)
![License](https://img.shields.io/badge/License-MIT-green)

![](https://i.postimg.cc/zfFynx8G/Screenshot-2026-09-23-011311.png)

## Overview

**AEOLUS** is a deep-learning framework for reconstructing **three-dimensional ocean subsurface temperature** from daily surface ocean observations.

The system learns the relationship between observable surface ocean conditions and the vertical temperature structure of the upper-to-mid ocean.

The current Proof-of-Concept focuses on the **North Indian Ocean**:

* **Latitude:** 5°N – 30°N
* **Longitude:** 45°E – 105°E
* **Spatial resolution:** 0.25° × 0.25°
* **Temporal resolution:** Daily
* **Maximum reconstruction depth:** 1000 m
* **Output depth levels:** 15

The model receives surface ocean and atmospheric variables and produces a complete depth-wise temperature field.

```text
Surface Ocean State
        │
        ├── SST
        ├── SSS
        ├── SSH
        ├── Surface Current U
        ├── Surface Current V
        ├── Wind U
        ├── Wind V
        ├── Latitude
        └── Longitude
                │
                ▼
      Residual Attention U-Net
              + CBAM
                │
                ▼
        Depth-Residual Head
                │
                ▼
     15 Temperature Fields
                │
                ▼
      3D Subsurface Ocean
       Temperature Volume
```

---

#  Problem Statement

Direct measurements of subsurface ocean temperature are spatially and temporally sparse. Observing systems such as **ARGO floats, moored buoys, gliders, and research vessels** provide valuable vertical measurements, but they cannot continuously cover an entire ocean basin.

Satellite and surface ocean products, in contrast, provide broad spatial coverage and frequent observations.

AEOLUS investigates whether surface ocean conditions can be used to reconstruct the hidden vertical temperature structure of the ocean using deep learning.

The system therefore learns:

> **Surface Ocean State → Subsurface Temperature Profile**

rather than relying on direct subsurface observations during inference.

---

![](https://i.postimg.cc/VL82h48J/Screenshot-2026-09-23-011626.png)

---

![](https://i.postimg.cc/YCJ6Xn5x/Screenshot-2026-09-23-011531.png)
---
![](https://i.postimg.cc/VNQt9Sth/Screenshot-2026-09-23-011503.png)
---

#  Study Region

The current implementation focuses on the **North Indian Ocean**.

| Parameter           | Configuration |
| ------------------- | ------------- |
| Northern Boundary   | 30°N          |
| Southern Boundary   | 5°N           |
| Western Boundary    | 45°E          |
| Eastern Boundary    | 105°E         |
| Output Resolution   | 0.25° × 0.25° |
| Temporal Resolution | Daily         |
| Maximum Depth       | 1000 m        |

The region includes major oceanic areas such as the **Arabian Sea, Bay of Bengal, and surrounding North Indian Ocean waters**.

---

# Input Datasets

##  CMEMS Global Ocean Physics Reanalysis

**Product:** `GLOBAL_MULTIYEAR_PHY_001_030`

**Dataset:** `cmems_mod_glo_phy_my_0.083deg_P1D-m`

The original ocean physics product has approximately **0.083° / 1/12° spatial resolution** and daily temporal resolution.

The variables used by the current model include:

| Variable  | Description                     | Model Role                         |
| --------- | ------------------------------- | ---------------------------------- |
| `thetao`  | Sea Water Potential Temperature | Surface input + target             |
| `so`      | Sea Water Salinity              | Surface input                      |
| `uo`      | Eastward Sea Water Velocity     | Surface current input              |
| `vo`      | Northward Sea Water Velocity    | Surface current input              |
| `zos`     | Sea Surface Height              | Surface input                      |
| `mlotst`  | Mixed Layer Thickness           | Available auxiliary ocean variable |
| `bottomT` | Sea Floor Potential Temperature | Available auxiliary variable       |

For the current seven-channel model input, the primary surface variables are:

```text
SST
SSS
SSH
Surface U Current
Surface V Current
Surface U Wind
Surface V Wind
```
![](https://i.postimg.cc/sxNQM1sd/Screenshot-2026-09-23-011429.png)
---
![](https://i.postimg.cc/gJbK0S4g/Screenshot-2026-09-23-011555.png)
---

#  Scatterometer Wind Dataset

The wind observations are obtained from the global ocean surface wind/stress product:

**WIND_GLO_PHY_L4_NRT_012_004**

Selected variables:

* `eastward_wind`
* `northward_wind`

The wind data provide surface atmospheric forcing information that can help the model learn relationships associated with:

* Vertical mixing
* Surface forcing
* Ocean-atmosphere interaction
* Transport
* Thermocline variability

The wind observations are spatially harmonized with the ocean dataset before being supplied to the neural network.

---

#  Data Harmonization Pipeline

The original datasets have different spatial resolutions.

```text
Ocean Reanalysis
0.083° × 0.083°
        │
        ▼
Spatial Regridding
        │
        ▼
0.25° × 0.25°
```

```text
Scatterometer Wind
~0.25°
        │
        ▼
Spatial / Temporal Alignment
        │
        ▼
0.25° × 0.25°
```

Both datasets are then aligned using their common timestamps.

### Current PoC

The current experiment uses six common daily observations:

```text
18 June 2026
19 June 2026
20 June 2026
21 June 2026
22 June 2026
23 June 2026
```

Each day is treated as an independent spatial training example in the current 2D CNN implementation.

```text
Day 1 → X₁ → Model → Y₁
Day 2 → X₂ → Model → Y₂
Day 3 → X₃ → Model → Y₃
Day 4 → X₄ → Model → Y₄
Day 5 → X₅ → Model → Y₅
Day 6 → X₆ → Model → Y₆
```

The model therefore does **not currently use temporal sequence modeling**.

---

# Input Representation

The final model input contains:

###  Ocean / Atmospheric Variables

```text
1. SST
2. SSS
3. SSH
4. Surface Current U
5. Surface Current V
6. Surface Wind U
7. Surface Wind V
```

### Coordinate Channels

Normalized:

```text
8. Latitude
9. Longitude
```

These coordinate channels implement a **CoordConv-style representation**, allowing the network to distinguish geographically different regions that may have similar local surface conditions.

### Final Input Tensor

```text
9 × 101 × 241
```

where:

```text
9   = input channels
101 = latitude dimension
241 = longitude dimension
```

---

#  Target Representation

The target is generated from subsurface `thetao` temperature.

The temperature field is interpolated to the following 15 standard depths:

```text
0 m
5 m
10 m
20 m
30 m
50 m
75 m
100 m
125 m
150 m
200 m
300 m
500 m
700 m
1000 m
```

The resulting target tensor is:

```text
15 × 101 × 241
```

Each output channel corresponds to one depth.

```text
Channel 0  → 0 m
Channel 1  → 5 m
Channel 2  → 10 m
Channel 3  → 20 m
...
Channel 14 → 1000 m
```

---

#  Model Architecture

AEOLUS uses a **Residual Attention U-Net with CBAM**.

The architecture is designed specifically for spatial reconstruction of the subsurface temperature field.

```text
                 INPUT
            9 × 101 × 241
                    │
                    ▼
          ┌─────────────────┐
          │ Residual Encoder│
          │ + CBAM Attention│
          └────────┬────────┘
                   │
             Downsampling
                   │
                   ▼
          ┌─────────────────┐
          │ Residual Encoder│
          │ + CBAM Attention│
          └────────┬────────┘
                   │
             Downsampling
                   │
                   ▼
          ┌─────────────────┐
          │ Residual Encoder│
          │ + CBAM Attention│
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Dilated Bottleneck│
          │ Dilation 2 + 4  │
          └────────┬────────┘
                   │
             Upsampling
                   │
          ┌────────▼────────┐
          │ Residual Decoder│
          │ + Skip Connection│
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Residual Decoder│
          └────────┬────────┘
                   │
                   ▼
          Depth-Residual Head
                   │
                   ▼
           15 × 101 × 241
```

### Model parameters

**13,015,651 trainable parameters**

---

#  Residual Convolution Blocks

Residual connections allow the network to learn feature corrections rather than requiring every layer to learn a complete transformation.

Advantages include:

* Improved gradient propagation
* More stable optimization
* Deeper feature extraction
* Better preservation of spatial information

---

#  CBAM Attention

The model incorporates **Convolutional Block Attention Module (CBAM)**.

CBAM contains two complementary attention mechanisms:

### Channel Attention

Learns which feature channels are important.

This allows the network to emphasize useful information from variables and learned representations.

### Spatial Attention

Learns which spatial regions are important.

This can help emphasize structures such as:

* Ocean fronts
* Mesoscale structures
* Coastal gradients
* Strong temperature gradients
* Eddy-like features

Conceptually:

```text
Feature Map
     │
     ├──── Channel Attention
     │
     ▼
Refined Features
     │
     ├──── Spatial Attention
     │
     ▼
Attention-Weighted Features
```

---

#  Dilated Bottleneck

The bottleneck uses dilated convolutions with dilation rates:

```text
Dilation = 2
Dilation = 4
```

This increases the effective receptive field without requiring a proportional increase in the number of convolutional layers.

The objective is to allow the network to capture larger spatial structures associated with oceanographic phenomena such as mesoscale variability.

---

#  U-Net Skip Connections

The U-Net architecture combines:

* High-level contextual features
* Fine-resolution spatial features

through encoder-decoder skip connections.

This is useful for reconstructing spatially detailed temperature maps while retaining larger-scale ocean structures.

---

#  Depth-Residual Prediction

Instead of predicting every temperature layer completely independently, the model uses a **depth-residual formulation relative to SST**.

Conceptually:

```text
Predicted Temperature(depth)
=
SST + Learned Depth Residual
```

For the surface:

```text
T(0 m) = SST
```

For deeper levels:

```text
T(depth) = SST + ΔT(depth)
```

where `ΔT(depth)` is learned by the neural network.

This gives the model an explicit physical reference point at the surface and allows it to focus on learning the vertical temperature differences.

---

#  Loss Function

Training uses a:

### Depth-Weighted Masked Hybrid MSE/MAE Loss

The loss combines MSE and MAE while accounting for the validity of ocean pixels.

A land-sea mask is used to prevent invalid land pixels from contributing to the loss.

The depth weighting helps prevent the optimization from being dominated by particular depth ranges.

Conceptually:

```text
Prediction
     │
     ▼
Valid Ocean Mask
     │
     ▼
Depth Weighting
     │
     ▼
MSE + MAE
     │
     ▼
Total Training Loss
```

---

#  Training

The model is trained using:

* **Optimizer:** AdamW
* **Learning-rate scheduler:** ReduceLROnPlateau
* **Gradient clipping**
* **Best-checkpoint saving**
* **Masked evaluation metrics**
* **CUDA acceleration**

The best model checkpoint is retained for inference.

---

#  Evaluation Metrics

The system evaluates predictions using:

### Mean Squared Error

Measures the squared prediction error.

### Root Mean Squared Error

```text
RMSE = √MSE
```

RMSE is reported in °C.

### Mean Absolute Error

Measures the average absolute temperature difference.

### R²

Measures the proportion of variance explained by the predictions relative to the evaluation target.

Metrics are calculated both:

* Across the complete valid ocean domain
* Independently for each depth level

---

#  Unseen-Data Evaluation

The model was subsequently evaluated on unseen data using the same input format:

```text
Surface Ocean Physics
+
Scatterometer Wind
        │
        ▼
     AEOLUS
        │
        ▼
15-depth temperature prediction
```

The resulting prediction tensor has:

```text
15 × 101 × 241
```

The current unseen-data evaluation produced:

|  Depth | RMSE (°C) | MAE (°C) |     R² |
| -----: | --------: | -------: | -----: |
|    0 m |         — |        — |      — |
|    5 m |    0.1261 |   0.0972 | 0.9263 |
|   10 m |    0.1591 |   0.1018 | 0.8627 |
|   20 m |    0.1783 |   0.1215 | 0.8260 |
|   30 m |    0.3442 |   0.2454 | 0.5515 |
|   50 m |    0.4585 |   0.3401 | 0.8275 |
|   75 m |    0.5659 |   0.4347 | 0.8829 |
|  100 m |    0.6708 |   0.5203 | 0.8864 |
|  125 m |    0.6879 |   0.5407 | 0.8847 |
|  150 m |    0.6281 |   0.4774 | 0.8819 |
|  200 m |    0.4012 |   0.2966 | 0.8598 |
|  300 m |    0.3191 |   0.2449 | 0.4176 |
|  500 m |    0.2496 |   0.1912 | 0.2457 |
|  700 m |    0.2047 |   0.1573 | 0.4489 |
| 1000 m |    0.1597 |   0.1222 | 0.3837 |

### Overall Test Metrics

```text
MSE  = 0.1711
RMSE = 0.4136 °C
MAE  = 0.2746 °C
R²   = 0.9977
Valid Pixels = 291,042
```

> **Important:** The 0 m layer is directly anchored to the input SST by the depth-residual formulation, so independent surface-layer error metrics are not meaningful and are therefore reported as NaN.

---

#  Prediction Visualization

AEOLUS provides several visualization modes.

##  Horizontal Depth Maps

Each depth is displayed as an independent temperature field.

```text
0 m       → Surface Temperature
5 m       → Subsurface Temperature
10 m      → Subsurface Temperature
...
1000 m    → Deep Temperature
```

These maps allow spatial temperature structures to be examined at individual depths.

---

##  Vertical Temperature Profiles

Temperature can be extracted at selected geographic coordinates and plotted as:

```text
Temperature (°C)
       │
       │ ●
       │  ●
       │   ●
       │    ●
       │     ●
       └────────────────
            Depth (m)
```

This allows the predicted vertical thermal structure to be compared against the target/reanalysis profile.

---

#  3D Ocean Temperature Cuboid

The 15 predicted temperature fields can be assembled into a three-dimensional visualization.

```text
                  Surface
        ┌────────────────────────┐
       /│       0 m             /│
      / │──────────────────────/ │
     /  │      50 m            /  │
    ┌────────────────────────┐   │
    │  │     100 m            │   │
    │  │──────────────────────│───│
    │  │     200 m            │   │
    │  │──────────────────────│───│
    │  │     500 m            │   │
    │  │──────────────────────│───│
    │  │    1000 m            │  /
    │  └──────────────────────│ /
    └─────────────────────────┘/
              Depth ↓
```

The visualization uses:

* Longitude → X axis
* Latitude → Y axis
* Depth → negative Z axis
* Temperature → color
* Interactive rotation/zoom
* 15 horizontal temperature planes

This provides a volumetric representation of the reconstructed ocean temperature structure.

---

#  NetCDF Output

Predictions can be exported as **NetCDF** for further scientific analysis.

The output contains:

```text
time
depth
latitude
longitude
temperature
```

Conceptually:

```text
temperature(time, depth, latitude, longitude)
```

This allows the generated temperature fields to be consumed by scientific Python workflows and other oceanographic tools.

---





#  Current Proof-of-Concept Limitations

The current implementation is a **Proof-of-Concept**, not yet a basin-scale operational reconstruction system.

The primary limitation is the short training period.

The current experiment contains only:

```text
6 daily observations
18–23 June 2026
```

This is insufficient to capture:

* Seasonal variability
* Interannual variability
* Monsoon transitions
* Different thermocline regimes
* Long-term ocean variability
* Rare extreme events
* Broader mesoscale variability

The current 2D CNN also treats each day independently and therefore does not explicitly model temporal dependencies.

Consequently, the current results demonstrate the **technical feasibility of spatial subsurface temperature reconstruction**, but they should not be interpreted as evidence of generalization across all seasons and ocean states.

---

# Future Development

The next stage of AEOLUS should expand the temporal and observational coverage.

### Longer Training Dataset

Instead of six days:

```text
Months → Years → Multi-year dataset
```

This would expose the model to substantially greater ocean variability.

### Temporal Deep Learning

Potential architectures include:

* ConvLSTM
* Temporal CNN
* 3D CNN
* Temporal Transformer
* Vision Transformer
* Spatiotemporal Transformer

The input could become:

```text
T × Channels × H × W
```

instead of processing each day independently.

### Independent ARGO Validation

A major future validation step is comparison against independent **ARGO observations** rather than relying only on reanalysis targets.

This would provide a stronger test of whether the reconstructed temperature profiles represent observed ocean conditions.

### Satellite-Derived Surface Inputs

Future versions can incorporate genuinely satellite-observed variables such as:

* Satellite SST
* Satellite SSH/SLA
* Satellite ocean color
* Scatterometer winds
* Satellite-derived surface products

This would move the system closer to the original objective of reconstructing subsurface conditions primarily from satellite observations.

### Uncertainty Estimation

Future models could estimate:

```text
Temperature Prediction
+
Prediction Uncertainty
```

using approaches such as:

* Deep ensembles
* Monte Carlo dropout
* Probabilistic neural networks
* Quantile regression
* Diffusion-based reconstruction

---

#  Scientific Objective

The long-term objective of AEOLUS is to investigate whether widely available surface ocean observations can be transformed into a useful representation of the hidden three-dimensional thermal structure of the ocean.

The conceptual relationship is:

```text
             SURFACE OBSERVATIONS
                     │
       ┌─────────────┼─────────────┐
       │             │             │
      SST           SSS           SSH
       │             │             │
    Currents       Winds       Coordinates
       │             │             │
       └─────────────┼─────────────┘
                     ▼
             Deep Learning
                     │
              Satellite/Ocean
                Representation
                     │
                     ▼
          Subsurface Reconstruction
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
     0–50 m       50–200 m      200–1000 m
       │             │             │
       └─────────────┼─────────────┘
                     ▼
             3D Temperature
                  Field
```

---



# 28. License

This project is released under the **MIT License**.
