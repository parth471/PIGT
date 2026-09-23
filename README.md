# PIGT — Physics-Informed Graph Transformer

**Physics-Informed Graph Transformer for Bay of Bengal Sea Surface Temperature Forecasting**

PIGT is a spatio-temporal deep learning model that predicts future Sea Surface Temperature (SST) over the Bay of Bengal using historical oceanographic and atmospheric data.

The model combines:

- Spatial Graph Convolutional Network (GCN)
- Temporal Transformer
- Physics-informed SST loss
- Uncertainty estimation
- Temporal attention-based explainability

The current implementation uses a **7-day historical window** to predict **next-day SST** at multiple ocean grid locations.

---

## 1. Project Overview

The PIGT model learns both:

1. **Spatial relationships** between nearby ocean locations using a graph neural network.
2. **Temporal relationships** across historical observations using a Transformer.

```
Historical Ocean Data
        ↓
Spatial Graph
        ↓
Graph Convolutional Network
        ↓
Temporal Transformer
        ↓
SST Prediction
        ↓
Physics-Informed Training
        ↓
Uncertainty & Explainability
```

---

## 2. Study Region

The implementation focuses on the Bay of Bengal.

| | Range |
|---|---|
| Latitude | 10°N – 20°N |
| Longitude | 80°E – 90°E |

- Original spatial grid: **121 × 121** locations
- Final graph after removing invalid/land locations: **12,204 ocean nodes**

---

## 3. Dataset

Environmental data sources:

- Copernicus Marine
- ERA5
- Bathymetry data

Data is aligned spatially and temporally before being converted into historical sequences.

**Temporal resolution:** Daily

---

## 4. Input Features

The model uses **13 input features**:

| No. | Feature |
|---|---|
| 1 | SST |
| 2 | Salinity |
| 3 | U Current |
| 4 | V Current |
| 5 | Significant Wave Height (SWH) |
| 6 | Wind Speed |
| 7 | Wind Direction |
| 8 | Pressure |
| 9 | Bathymetry |
| 10 | Latitude |
| 11 | Longitude |
| 12 | Sin(Day of Year) |
| 13 | Cos(Day of Year) |

Feature order is fixed per the dataset metadata.

---

## 5. Temporal Input & Model I/O

The model uses the previous **7 days** of observations to predict **day t+1**.

```
Day t-6 ... Day t
      ↓
    PIGT
      ↓
 Day t+1 SST
```

**Input tensor:**
```
X = [B, T, N, F]
```
- `B` = batch size
- `T` = 7 historical time steps
- `N` = 12,204 ocean nodes
- `F` = 13 input features

So: `X = [B, 7, 12204, 13]`

**Prediction target:** Future SST for every valid ocean node.

**Output tensor:**
```
Prediction = [B, N]  →  [B, 12204]
```

---

## 6. Spatial Graph

Ocean locations are represented as nodes in a **k-nearest-neighbor (kNN) graph**.

| Parameter | Value |
|---|---|
| Nodes | 12,204 |
| Neighbors (k) | 8 |
| Edges | 97,632 |

The GCN uses this graph to propagate spatial information between neighboring ocean locations.

---

## 7. Graph Convolutional Network (Spatial Encoder)

```
13 Input Features
        ↓
Feature Projection
        ↓
64-Dimensional Representation
        ↓
GCN Layer → GCN Layer
        ↓
Spatial Representation
```

| Parameter | Value |
|---|---|
| Hidden dimension | 64 |
| GCN layers | 2 |

---

## 8. Temporal Transformer

For each ocean node, representations from the 7 historical time steps are passed through a Transformer to learn temporal relationships.

| Parameter | Value |
|---|---|
| Hidden dimension | 64 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Dropout | 0.1 |
| Max time steps | 7 |

---

## 9. Full Model Architecture

```
Input [B, 7, N, 13]
        ↓
Feature Projection
        ↓
Spatial GCN
        ↓
Spatial Features
        ↓
Temporal Transformer
        ↓
Spatio-Temporal Representation
        ↓
SST Prediction Head
        ↓
Output [B, N]
```

---

## 10. Physics-Informed Learning

The model includes a physics-informed loss based on the SST advection–diffusion relationship:

```
∂T/∂t + (u · ∇)T = κ∇²T
```

where `T` = SST, `u` = ocean velocity, `κ` = thermal diffusivity.

**Physical residual:**
```
R = ∂T/∂t + (u · ∇)T − κ∇²T
```

**Physics loss:**
```
Physics Loss = mean(R²)
```

**Total training objective:**
```
Total Loss = Prediction Loss + λ × Physics Loss
```
Current `λ = 0.1`.

```
Historical Data → PIGT Model → Predicted SST
                                    ├──→ Prediction Loss
                                    └──→ Physics Loss
                                              ↓
                                        Total Loss
                                              ↓
                                       Backpropagation
                                              ↓
                                        Model Update
```

---

## 11. Uncertainty Estimation

The model provides predictive uncertainty alongside the SST prediction, combining:

- **Aleatoric uncertainty** — a predicted mean (μ) and log-variance (σ²), trained with a Gaussian negative log-likelihood.
- **Epistemic uncertainty** — estimated via Monte Carlo Dropout across multiple stochastic forward passes.

```
             PIGT
      ┌───────┼───────┐
      ↓       ↓       ↓
    Pass 1  Pass 2 ... Pass M
      └───────┼───────┘
              ↓
     Prediction Distribution
```

Reported quantities: predictive mean, aleatoric variance, epistemic variance, total variance, and standard deviation.

```
Total Uncertainty = Aleatoric Uncertainty + Epistemic Uncertainty
```

---

## 12. Explainability

Temporal explainability is derived from Transformer attention across the 7 historical time steps, identifying which past days receive the most weight during prediction.

```
t-6 ── Attention
t-5 ── Attention
t-4 ── Attention
t-3 ── Attention
t-2 ── Attention
t-1 ── Attention
t   ── Attention
```

---

## 13. Model Evaluation

Comparison is made between the **GCN baseline** and **PIGT**, using:

- MAE
- RMSE
- R²

Evaluation is performed on held-out test data.

**GCN Baseline:** spatial-only model (no temporal Transformer), used as a reference point.
```
Input → GCN → SST Prediction
```

---

## 14. Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Gradient clip | 1.0 |
| Hidden dimension | 64 |
| GCN layers | 2 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Dropout | 0.1 |
| History window | 7 days |
| Forecast horizon | 1 day |

---

## 15. Project Structure

```
PIGT-tiyasa/
│
├── config.py
├── requirements.txt
│
├── download_copernicus_marine.py
├── download_era5.py
│
├── 01_inspect_raw_data.py
├── 02_align_and_merge.py
├── 03_build_dataset.py
├── 04_inspect_final_dataset.py
├── 04_verify_final_dataset.py
│
├── data/
│   └── final/
│       └── metadata.json
│
├── src/
│   ├── models/
│   │   ├── build_graph.py
│   │   ├── gcn_baseline.py
│   │   ├── pigt_model.py
│   │   ├── pigt_uncertainty.py
│   │   ├── temporal_transformer.py
│   │   ├── train_gcn_baseline.py
│   │   ├── train_pigt.py
│   │   ├── train_pigt_physics.py
│   │   ├── evaluate_models.py
│   │   ├── test_pigt_model.py
│   │   ├── test_pigt_real_graph.py
│   │   └── test_temporal_transformer.py
│   │
│   ├── physics/
│   │   ├── __init__.py
│   │   └── physics_loss.py
│   │
│   └── explainability/
│       ├── __init__.py
│       ├── temporal_attention.py
│       └── visualize_results.py
│
├── tests/
│   ├── test_physics_loss.py
│   ├── test_uncertainty_model.py
│   └── test_explainability.py
│
└── results/
    ├── figures/
    └── test_plots/
```

---

## 16. Setup

```bash
python -m venv pigt_env

# Windows
pigt_env\Scripts\activate

# Linux/macOS
source pigt_env/bin/activate

pip install -r requirements.txt
```

---

## 17. Data Preparation

```bash
python download_copernicus_marine.py
python download_era5.py

python 01_inspect_raw_data.py
python 02_align_and_merge.py
python 03_build_dataset.py
python 04_verify_final_dataset.py
```

---

## 18. Build the Graph

```bash
python src/models/build_graph.py
```

Produces: 12,204 nodes, 97,632 directed edges, 8 nearest neighbors per node.

---

## 19. Run Tests

```bash
python src/models/test_temporal_transformer.py
python src/models/test_pigt_model.py
python src/models/test_pigt_real_graph.py
python tests/test_physics_loss.py
python tests/test_uncertainty_model.py
python tests/test_explainability.py
```

---

## 20. Train the Models

```bash
# GCN baseline
python src/models/train_gcn_baseline.py

# Standard PIGT
python src/models/train_pigt.py

# Physics-informed PIGT
python src/models/train_pigt_physics.py
```

---

## 21. Evaluate the Models

```bash
python src/models/evaluate_models.py
```

Compares the GCN baseline and PIGT using MAE, RMSE, and R².

---

## 22. Results and Visualizations

Generated outputs, stored under `results/`:

- Temporal attention profile
- Forecast uncertainty bounds
- Spatial uncertainty map
- Test attention profile
- Test spatial map
- Test uncertainty bounds

---

## 23. End-to-End Workflow

```
Copernicus Marine + ERA5 + Bathymetry
                ↓
        Data Preprocessing
                ↓
     Temporal Sequence Creation
                ↓
      7-Day Historical Window
                ↓
         Spatial kNN Graph
                ↓
        GCN Spatial Encoder
                ↓
        Temporal Transformer
                ↓
      Next-Day SST Prediction
                ↓
        Physics-Informed Loss
                ↓
       Uncertainty Estimation
                ↓
     Temporal Attention Analysis
                ↓
      Bay of Bengal SST Forecast
```

---

## 24. Model Summary

**Input:** `[B, 7, 12204, 13]`

```
Feature Projection
        ↓
2-Layer GCN (64 hidden dims)
        ↓
2-Layer Temporal Transformer (4 attention heads)
        ↓
SST Prediction Head
        ↓
Output: [B, 12204]
```

The physics-informed variant additionally uses `Prediction Loss + 0.1 × Physics Loss`.

The uncertainty-aware implementation provides predictive mean and uncertainty estimates via aleatoric modeling and Monte Carlo Dropout.

The explainability component uses temporal Transformer attention to analyze the importance of historical timesteps.
