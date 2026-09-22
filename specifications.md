# 27. Key Specifications

| Component           | Specification                               |
| ------------------- | ------------------------------------------- |
| Project             | AEOLUS                                      |
| Task                | Subsurface Ocean Temperature Reconstruction |
| Region              | North Indian Ocean                          |
| Spatial Resolution  | 0.25° × 0.25°                               |
| Temporal Resolution | Daily                                       |
| Input Channels      | 9                                           |
| Output Channels     | 15                                          |
| Output Depth        | 0–1000 m                                    |
| Model               | Residual Attention U-Net                    |
| Attention           | CBAM                                        |
| Parameters          | 13,015,651                                  |
| Optimizer           | AdamW                                       |
| Scheduler           | ReduceLROnPlateau                           |
| Acceleration        | CUDA                                        |
| Backend             | FastAPI                                     |
| Containerization    | Docker                                      |
| Visualization       | Matplotlib / Plotly                         |
| Scientific Data     | Xarray / NetCDF                             |
| Current PoC Period  | 18–23 June 2026                             |
| Current Test RMSE   | 0.4136 °C                                   |
| Current Test MAE    | 0.2746 °C                                   |
| Current Test R²     | 0.9977                                      |

---