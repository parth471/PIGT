# PIGT — Person 2 Implementation & Handoff Specification

**Module:** Graph Learning + Temporal Transformer + Training/Evaluation  
**Target Audience:** Person 3 (Physics-Informed Loss, Uncertainty Quantification, and Explainability Layers)

---

## 1. Project Scope & Division of Responsibilities

| Role | Domain & Deliverables | Status |
| :--- | :--- | :--- |
| **Person 1** | Data ingestion, Copernicus Marine & ERA5 alignment, quality control, $12,204$ ocean-node masking, chronological train/val/test rolling sequence generation. | ✅ Verified Complete |
| **Person 2** | Geographic $k$-NN graph construction ($k=8$), Spatial GCN, Temporal Transformer, baseline model, training/validation loops, test evaluation. | ✅ Implemented & Verified |
| **Person 3** | Physics-informed loss formulation (thermal conservation, advection/diffusion constraints), uncertainty quantification, explainability/attention visualization. | 🚀 Ready for Implementation |

---

## 2. Person 1 Dataset Contract (Verified)

All final dataset files reside in `data/final/`:

- **Input Sequences ($X$):** Shape `[B, T=7, N=12204, F=13]`
  - Train: `X_train.npy` $(549, 7, 12204, 13)$
  - Validation: `X_val.npy` $(118, 7, 12204, 13)$
  - Test: `X_test.npy` $(118, 7, 12204, 13)$
- **Target Horizons ($Y$):** Shape `[B, N=12204, 5]` ($1$ day ahead)
  - Train: `Y_train.npy` $(549, 12204, 5)$
  - Validation: `Y_val.npy` $(118, 12204, 5)$
  - Test: `Y_test.npy` $(118, 12204, 5)$
  - **Note:** SST is target channel `0` (`Y[:, :, 0]`).
- **Feature Order ($F=13$):**
  1. `sst` (Sea Surface Temperature, $^\circ\text{C}$)
  2. `salinity` (Practical Salinity, $\text{psu}$)
  3. `u_current` (Zonal Eastward Velocity, $\text{m/s}$)
  4. `v_current` (Meridional Northward Velocity, $\text{m/s}$)
  5. `swh` (Significant Wave Height, $\text{m}$)
  6. `wind_speed` (ERA5 10m Wind Speed, $\text{m/s}$)
  7. `wind_direction` (ERA5 Meteorological Wind Direction, deg)
  8. `pressure` (ERA5 Mean Sea-Level Pressure, $\text{hPa}$)
  9. `bathymetry` (Ocean depth, $\text{m}$)
  10. `latitude` (Normalized coordinate, $[-1, 1]$)
  11. `longitude` (Normalized coordinate, $[-1, 1]$)
  12. `sin_day_of_year` (Cyclical calendar encoding)
  13. `cos_day_of_year` (Cyclical calendar encoding)
- **Quality Assurance:** $0$ NaNs or Infs across all splits.

---

## 3. Person 2 Graph & Model Architecture

### Graph Topology (`data/graph/graph.pt`)
- **Nodes ($N$):** $12,204$ valid ocean points in the Bay of Bengal ($10^\circ\text{N}-20^\circ\text{N}, 80^\circ\text{E}-90^\circ\text{E}$).
- **Connectivity:** Geographic 8-Nearest Neighbors ($k=8$).
- **Edges:** $97,632$ directed edges (`edge_index` shape $[2, 97632]$) with Euclidean distances (`edge_attr` shape $[97632, 1]$).
- **Node Ordering:** Node index $i$ in `graph.pt` corresponds identically to row $i$ in `node_lat.npy`, `node_lon.npy`, and index $i$ in dimension $N$ of $X$ and $Y$.

### Model Interface (`src/models/pigt_model.py`)
```python
model = PIGTModel(
    in_features=13,
    hidden_dim=64,
    num_heads=4,
    num_transformer_layers=2,
    dropout=0.1,
    max_time_steps=7,
)

# Forward call:
# x: [B, T=7, N=12204, F=13]
# edge_index: [2, 97632]
# Returns: prediction [B, N=12204] (next-day SST)
prediction = model(x, edge_index)
```

### Baseline Model (`src/models/gcn_baseline.py`)
- Standard 2-layer GCN (`GCNConv`) operating on the latest single historical day ($T=1$, $[N, 13]$) to benchmark the temporal attention gain.

---

## 4. Handoff Contract for Person 3

Person 3 can cleanly plug in downstream components as follows:

1. **Physics-Informed Loss ($L_{\text{total}} = L_{\text{data}} + \lambda L_{\text{physics}}$):**
   - Features in $X[:, -1, :, :]$ provide the full physical ocean state at day $t$ (`sst` at index 0, `salinity` at index 1, `u_current` at index 2, `v_current` at index 3, `wind_speed` at index 5, `bathymetry` at index 8, `latitude`/`longitude` at indices 9-10).
   - Prediction $\hat{Y}$ is SST at day $t+1$.
   - Spatial finite differences or graph Laplacian operations can be computed directly using `edge_index` and `edge_attr` from `data/graph/graph.pt`.

2. **Uncertainty Quantification:**
   - The prediction head in [src/models/pigt_model.py](file:///Users/tiyasapaul/Desktop/projects/PIGT/src/models/pigt_model.py) can be expanded to output Gaussian parameter heads $(\mu, \sigma^2)$ or ensemble/Monte-Carlo Dropout distributions.

3. **Explainability & Attention Weights:**
   - Person 3 can extract cross-temporal attention matrices from `model.temporal_transformer` to interpret seasonal and meteorological lag influences on forecasting accuracy.

---

## 5. Execution Reference

```bash
# 1. Build spatial graph
python src/models/build_graph.py

# 2. Run model sanity and unit tests
python src/models/test_temporal_transformer.py
python src/models/test_pigt_model.py
python src/models/test_pigt_real_graph.py

# 3. Train models
python src/models/train_gcn_baseline.py
python src/models/train_pigt.py

# 4. Comparative evaluation
python src/models/evaluate_models.py
```
