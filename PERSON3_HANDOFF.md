# PIGT Person 3 Handoff

Prepared from the repository and local artifacts on 2026-10-06. This is a handoff of the current state, not a claim that final model training is complete.

## 1. Project overview

PIGT is an ocean forecasting project for the Bay of Bengal. The current pipeline combines ocean observations, wave and atmospheric variables, static geography, and temporal features. The current eight-target PIGT model takes seven daily snapshots over the ocean-node graph and predicts eight dynamic variables for the next day at every node.

## 2. Current verified dataset

The saved NumPy array headers and `data/final/metadata.json` were inspected. The repository verifier has also passed its shape and NaN/Inf checks for these artifacts.

```text
X_train: (549, 7, 12204, 13)
Y_train: (549, 12204, 8)

X_val:   (118, 7, 12204, 13)
Y_val:   (118, 12204, 8)

X_test:  (118, 7, 12204, 13)
Y_test:  (118, 12204, 8)
```

The task is **7 historical daily steps → next-day forecast**. There are 12,204 spatial ocean nodes, 13 input channels, and 8 output channels. Metadata describes the common period as 2022-11-01 through 2024-12-31 (792 daily timesteps) and the split fractions as 70% train, 15% validation, 15% test. `03_build_dataset.py` builds windows in chronological order and cuts the ordered sequence array into those splits. Training samples are shuffled only inside the already-created training split during optimization; the split itself is not randomized. The verifier's temporal-leakage check is **hard-coded as `check(True, ...)`**. It does not inspect dates or independently prove that leakage was prevented; independently verify target dates before treating that check as proof.

The current input feature order is:

1. `sst`
2. `salinity`
3. `u_current`
4. `v_current`
5. `swh`
6. `wind_speed`
7. `wind_direction`
8. `pressure`
9. `bathymetry`
10. `latitude`
11. `longitude`
12. `sin_day_of_year`
13. `cos_day_of_year`

The current target order is:

1. `sst`
2. `salinity`
3. `u_current`
4. `v_current`
5. `swh`
6. `wind_speed`
7. `wind_direction`
8. `pressure`

All six saved split arrays are `float32`. The latest `04_verify_final_dataset.py` run reported no NaNs or infinities in any of X/Y train, validation, or test arrays. The check verifies array values and shapes; its temporal-leakage assertion has the limitation above.

## 3. Graph

- File: `data/graph/graph.pt`
- Nodes: 12,204
- Edges: 97,632, or 8 outgoing nearest-neighbor edges per node (`k=8`). The saved graph has `edge_index` shape `[2, 97632]` and also stores edge distances and `k`.
- `src/models/train_pigt.py` and `src/models/evaluate_models.py` load and reuse this graph.
- Graph node indices correspond to the `node_lat.npy` / `node_lon.npy` order used to create the dataset's node series. Keep this order identical to the N dimension of every X/Y array. Rebuilding the graph from differently ordered coordinates would silently connect the wrong nodes.

The graph is generated from the final node coordinates by `src/models/build_graph.py`. The graph file is ignored by Git (`*.pt`); it is present in this working copy but is not part of the pushed handoff commit.

## 4. PIGT architecture

Current implementation: `src/models/pigt_model.py`.

1. Input is `[B, 7, N, 13]`; the implementation checks the feature count and supports at most seven time steps.
2. A linear feature projection maps 13 channels to 64 hidden channels: `[B, 7, N, 64]`.
3. Two spatial `GCNConv(64, 64)` layers with ReLU operate on each of the B×7 copies of the shared node graph. The graph copies are assembled by offsetting node indices.
4. A learned temporal position embedding is added. The model rearranges the representation to `[B×N, 7, 64]`, so the two-layer, four-head Transformer encoder attends over time independently for each node. Each encoder layer has a 256-wide feed-forward block; the final historical step is retained.
5. Layer normalization and a `64 → 64 → 8` head (linear, GELU, dropout 0.1, linear) yield `[B, N, 8]`.

The current PIGT head predicts **all eight target channels** in the target order listed above. The separate `src/models/temporal_transformer.py` contains similar transformer code but is not used by `pigt_model.py`; it has deliberately not been refactored.

## 5. Eight-target training implementation

Current implementation: `src/models/train_pigt.py`.

- Reads only `X_train`, `Y_train`, `X_val`, and `Y_val`. It does not load or use test arrays for training or checkpoint selection.
- Uses every Y channel in train and validation batches. Expected batch tensors are prediction/target `[B, 12204, 8]`.
- Fits no scalers during model training. It loads the saved target `StandardScaler` objects from `data/final/feature_scaler.pkl` and validates their order against both the expected list and `metadata.json`.
- Loss is computed in standardized space. For the seven linear channels it is ordinary squared error. For `wind_direction`, it converts the normalized prediction and target back to degrees using the saved scaler's verified `mean_` and `scale_`, finds the shortest angular difference by wrapping the radian delta with `atan2(sin(delta), cos(delta))`, converts it back to degrees, divides by the scaler scale, and squares it. It then averages the eight per-channel mean squared errors equally.
- Uses batch size 1, AdamW (`lr=0.001`, `weight_decay=1e-5`), gradient clipping at 1.0, and a fixed seed of 42. Training sequences are shuffled within the train split.
- Validation loss selects the best checkpoint. Early stopping patience is 5 epochs without validation improvement.
- Maximum epochs are controlled by `PIGT_MAX_EPOCHS`; the default is 50. Early stopping may stop before that maximum.
- Optional `PIGT_MAX_TRAIN_SAMPLES` limits the training arrays to their first N sequences. It does not change validation or test data.
- `PIGT_CHECKPOINT_NAME` and `PIGT_RUN_LABEL` allow separate named runs. The trainer refuses to use `pigt_best.pt` or `pigt_8target_best.pt` when supplied as a checkpoint name, and refuses to overwrite any existing selected checkpoint file.
- With no override, the best full-run model is saved as `results/checkpoints/pigt_8target_best.pt`. That full-training checkpoint does **not** exist in the inspected working copy yet.

## 6. Wind direction

Wind direction is an angle, so 359° and 1° are close, not 358° apart. The **target loss** handles this wraparound using the shortest angular delta, and the evaluator reports circular MAE/RMSE in degrees. It omits ordinary R² for wind direction because ordinary scalar R² is not appropriate for this circular error.

The **input feature** `wind_direction` is still linearly derived/scaled like a scalar, so its representation has a discontinuity at 0°/360°. This is a known limitation and a possible future improvement (for example, input sine/cosine encoding); it is not a blocker for the current prototype. Changing that representation would require a coordinated dataset/model feature-contract update.

## 7. Existing baselines

These are the previously run, normalized-scale **SST-only** baseline results from the old single-output checkpoints:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| GCN | 0.081559 | 0.111338 | 0.970501 |
| Old single-target PIGT | 0.202974 | 0.253037 | 0.847633 |

Both models were trained/evaluated using only target channel 0 (`sst`). These normalized-scale SST results are **not directly comparable to RAPID5's physical-unit per-target metrics**, and they do not establish a final eight-target model comparison.

## 8. RAPID 8-target PIGT experiment

The RAPID5 run used the existing eight-target model and loss, the first 40 of 549 training sequences, the unchanged full 118-sequence validation split, and the unchanged full 118-sequence test split. It ran exactly five epochs. The model/data architecture and dataset were not changed for this run.

| Epoch | Train loss | Validation loss | Epoch time |
|---:|---:|---:|---:|
| 1 | 0.594700 | 1.534872 | 156.83 s |
| 2 | 0.337046 | 1.302286 | 160.74 s |
| 3 | 0.294080 | 1.121152 | 160.83 s |
| 4 | 0.258521 | **1.005399** | 162.90 s |
| 5 | 0.245536 | 1.484061 | 165.48 s |

The sum of recorded epoch times is **806.78 seconds**. Each epoch timer spans both training and its full validation pass. **Epoch 4** had the best validation loss and was selected for the checkpoint; epoch 5 regressed. Validation improved over epoch 1.

## 9. RAPID5 test results

Every result below is labeled **RAPID5 / PRELIMINARY / HANDOFF ONLY**. The listed physical metrics come from the committed test report; the overall summary is in standardized units.

| Target | MAE | RMSE | R² |
|---|---:|---:|---:|
| SST | 0.696292 °C | 0.856401 °C | -0.005978 |
| Salinity | 0.503058 (1e-3) | 0.722937 (1e-3) | 0.877849 |
| U current | 0.118136 m/s | 0.155309 m/s | 0.684504 |
| V current | 0.125431 m/s | 0.170986 m/s | 0.721132 |
| SWH | 0.343139 m | 0.481446 m | 0.278276 |
| Wind speed | 1.747509 m/s | 2.254257 m/s | 0.219820 |
| Pressure | 2.035804 hPa | 2.880997 hPa | 0.085955 |

Wind direction: circular MAE **51.329564°**, circular RMSE **70.670697°**; R² omitted.

Overall standardized MAE is **0.514779**. The requested/previously summarized overall standardized RMSE is **0.706054**. The committed JSON records `0.7060520915096168` (0.706052 to six decimal places), a difference of about 0.000002; use the JSON value as the artifact source of record and preserve this discrepancy when copying the supplied summary figure.

This model was severely undertrained: it saw 40/549 training sequences for five epochs only. These values are **not final PIGT performance** and must not be presented as such.

## 10. Checkpoints and result files

| Path | Meaning / state |
|---|---|
| `results/checkpoints/pigt_8target_rapid5.pt` | Current best eight-target rapid checkpoint; saved at epoch 4. This is the only rapid checkpoint/report pair included in the latest commit. |
| `results/evaluation/pigt_8target_rapid5_metrics.json` | Committed RAPID5 test metrics: label, checkpoint/split, sample count, input/output shape contract, target order, per-target metrics, and overall standardized summary. |
| `results/checkpoints/pigt_best.pt` | Old single-target SST PIGT checkpoint; leave untouched. |
| `results/checkpoints/gcn_baseline_best.pt` | SST-only GCN baseline checkpoint. |
| `results/checkpoints/pigt_8target_rapid.pt` | Earlier one-epoch, 40-sequence eight-target rapid checkpoint; separate from RAPID5 and not the current best. |
| `results/evaluation/pigt_8target_rapid_metrics.json` | Earlier one-epoch rapid evaluation report; present locally but untracked. |
| `results/checkpoints/pigt_8target_best.pt` | Expected output of full eight-target training; absent at inspection time. |
| `results/evaluation/pigt_8target_test_metrics.json` | Default full-model evaluator output; absent at inspection time. |

The evaluator reads `X_test`/`Y_test`, loads `pigt_8target_best.pt` by default, calculates per-target physical-unit metrics (circular metrics for direction), and writes the JSON report. For rapid checkpoints, override its checkpoint and report paths with `PIGT_CHECKPOINT_NAME` and `PIGT_REPORT_PATH`. It does not save full prediction tensors.

## 11. Current Git/repository state

- Branch: `parth_branch`
- Remote: `origin` → `https://github.com/parth471/PIGT.git`
- HEAD: `d7d55d74f6ef14cc54ef05dbab741b09ab316035` (`Add eight-target PIGT training and rapid results`)
- At inspection, local HEAD and `origin/parth_branch` matched (`ahead 0, behind 0`).
- The latest commit intentionally includes nine paths: `03_build_dataset.py`, `04_verify_final_dataset.py`, `config.py`, `data/final/metadata.json`, `src/models/evaluate_models.py`, `src/models/pigt_model.py`, `src/models/train_pigt.py`, `results/checkpoints/pigt_8target_rapid5.pt`, and `results/evaluation/pigt_8target_rapid5_metrics.json`.
- Existing local uncommitted state before creating this document: deleted `requirements.txt` and untracked `results/evaluation/pigt_8target_rapid_metrics.json`. This handoff document is also new/untracked until separately committed. There were no staged changes at inspection.
- `.gitignore` excludes `.venv/`, `*.npy`, `*.pkl`, and `*.pt` (among other generated data). `git ls-files` confirmed that only `data/final/metadata.json` is tracked under `data/final`/`data/graph`; the data arrays, scaler, graph and virtual environment are **not on GitHub**. The RAPID5 checkpoint was explicitly force-added to the commit. Person 3 must obtain the ignored dataset artifacts and graph separately (or rebuild them) before training/evaluating locally.
- The deletion of `requirements.txt` is local and uncommitted; the pushed HEAD still contains it. That committed manifest lists common data packages and `torch`, but does **not** list `torch-geometric`, which the GCN models require.
- The committed README is stale: it instructs `cd scripts` although scripts are at repository root and documents five targets `[B,N,5]`. The root requirements file is absent only in this local working tree because of its uncommitted deletion.

## 12. What Person 3 should do next

1. **Verify checkout and environment.** From repository root:

   ```powershell
   git status --short --branch
   .\.venv\Scripts\python.exe -c "import numpy, torch, torch_geometric, sklearn, xarray; print(torch.__version__, torch_geometric.__version__)"
   ```

   The inspected local environment was Python 3.10.11, NumPy 2.2.6, PyTorch 2.13.0+cpu, PyG 2.8.0.post1, scikit-learn 1.7.2, and xarray 2025.6.1. A clone will not include `.venv`; create a compatible environment and install PyG explicitly because the committed manifest omits it. Obtain the ignored dataset and graph artifacts before running project commands.

2. **Verify the dataset artifacts** after they are available:

   ```powershell
   .\.venv\Scripts\python.exe .\04_verify_final_dataset.py
   ```

   Independently inspect target-date split boundaries as well; the verifier's leakage check is not sufficient.

3. **Treat RAPID5 strictly as a preliminary proof of concept.** Inspect `results/evaluation/pigt_8target_rapid5_metrics.json` and retain its preliminary label in any handoff/report.

4. **If compute is limited, prioritize a 2024-only experiment before full multi-year training.** There is currently no date-filter option for train samples. `PIGT_MAX_TRAIN_SAMPLES` selects the earliest N existing training sequences, not 2024 data. The saved arrays also lack per-sample date arrays. Do not use that option as a substitute; first implement and verify date-aware target-date selection and preserve valid chronological validation/test periods. No safe one-year command exists until that support is added.

5. **Run stronger/full multi-year training when adequate compute is available.** The trainer defaults to a 50-epoch maximum and patience 5, reads all 549 train sequences, uses all 118 validation sequences, and writes the best validation checkpoint to `pigt_8target_best.pt`. From repository root, with the dataset and graph present:

   ```powershell
   Remove-Item Env:PIGT_MAX_TRAIN_SAMPLES, Env:PIGT_CHECKPOINT_NAME, Env:PIGT_RUN_LABEL -ErrorAction SilentlyContinue
   $env:PIGT_MAX_EPOCHS = '50'
   .\.venv\Scripts\python.exe -u .\src\models\train_pigt.py
   ```

   It refuses to overwrite an existing selected checkpoint. CPU training is expensive: the 40-sequence rapid run took about 160 seconds per epoch including validation. A full epoch over 549 train sequences plus validation will take substantially longer; use an available GPU and monitor unbuffered output.

6. **Evaluate the best validation-selected checkpoint on the untouched test split** only after training:

   ```powershell
   .\.venv\Scripts\python.exe -u .\src\models\evaluate_models.py
   ```

   Default report: `results/evaluation/pigt_8target_test_metrics.json`. The current evaluator loads the test arrays only for this post-training evaluation.

7. **Produce final per-target metrics and fair comparisons.** Compare a final model against baselines on matching splits, target channels, units, and scaling. The old GCN/PIGT numbers are SST-only normalized metrics, so they are not directly comparable to the new physical-unit eight-target table.

8. **Generate report plots/results.** No current script in the inspected model pipeline writes the requested final plots; create those after final metrics are established.

9. **Do not report RAPID5 as final performance.** Its 40-sample/five-epoch training is a handoff prototype only.

## 13. Known limitations / risks

- RAPID5 used only 40 of 549 training sequences (about 7.3%) and only five epochs; it is severely undertrained.
- PIGT is computationally expensive on CPU. The measured rapid epoch times imply that full training may take many hours on CPU.
- The wind-direction input feature remains linearly scaled and discontinuous at 0°/360°, even though target loss and test metrics are circular-aware.
- The verifier's “no leakage” line is unconditional; independently verify split dates and scaler fitting boundaries.
- README setup paths and five-target description are stale. The local worktree lacks `requirements.txt` due an uncommitted deletion; the committed manifest omits `torch-geometric`.
- Existing `test_pigt_model.py` and `test_pigt_real_graph.py` still assert a single-target output shape `[B,N]`. They are stale for the current eight-target head and need updating before they can validate it.
- The full eight-target model has not yet been trained on all 549 sequences for a suitable duration; final model performance has not been established.
- Data arrays, scaler, graph and `.venv` are ignored and absent from GitHub. Ensure artifact transfer/availability for Person 3.

## 14. Reproduction checklist

- [ ] Checkout `parth_branch` at or beyond `d7d55d74f6ef14cc54ef05dbab741b09ab316035`.
- [ ] Create a compatible Python environment; install PyTorch Geometric explicitly.
- [ ] Obtain the ignored final arrays, scaler, node coordinates and `data/graph/graph.pt`.
- [ ] Run `04_verify_final_dataset.py`; independently inspect chronological target dates.
- [ ] Confirm graph has 12,204 nodes and `[2, 97632]` edges; preserve node order.
- [ ] Confirm `pigt_model.py` outputs `[B,N,8]` for eight targets in the expected order.
- [ ] Inspect RAPID5 checkpoint metadata: epoch 4, 40 training sequences, five-epoch run label.
- [ ] Treat `pigt_8target_rapid5_metrics.json` as preliminary only.
- [ ] Run stronger/full training to create `pigt_8target_best.pt`; retain validation-selected checkpoint.
- [ ] Run the evaluator once on the untouched test split and save final per-target metrics.
- [ ] Generate final comparison tables/plots; do not mix units or compare RAPID5 as final.

## 15. One-paragraph handoff summary

PIGT currently has a verified 792-day Bay of Bengal dataset with chronological 549/118/118 sequence splits, seven days of 13 features as input, and eight next-day targets over 12,204 nodes. The GCN + temporal Transformer PIGT now outputs all eight channels and has circular-aware wind-direction target loss/evaluation. The best available eight-target checkpoint is `pigt_8target_rapid5.pt`, selected at epoch 4 after training only 40 sequences for five epochs; its metrics are preliminary, not final. Full multi-year eight-target training and final untouched-test evaluation remain to be done. The verifier's leakage assertion is hard-coded, the 2024-only training selection is not currently supported, and a clone needs the ignored arrays/scaler/graph plus a compatible environment. Start by obtaining those artifacts, verifying split dates independently, then plan a date-aware 2024 experiment or stronger full multi-year training before making final performance claims.
