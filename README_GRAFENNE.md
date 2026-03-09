# GRAFENNE — Results

## Run Configuration

| Parameter | Value |
|---|---|
| Nodes (N) | 128 |
| Initial Features (F_init) | 12 |
| Max Features (F_max) | 16 |
| Hidden Dim | 32 |
| Feature Presence | 40–100% per node |
| Seed | 123 |
| Device | CUDA (GPU) |

## Training Results

### 2-Phase Model (v1)

| Epoch | Loss |
|---|---|
| 0 | 1.11 |
| 40 | 0.75 |
| 80 | 0.25 |
| 120 | 0.09 |
| 160 | 0.03 |

| Metric | Value |
|---|---|
| Parameters | 15 |
| Training Epochs | 200 |
| Learning Rate | 0.01 |
| Test Accuracy | 16/27 (59.3%) |

### Checkpoint: 2-Phase -> 3-Phase (v2)

| Metric | Value |
|---|---|
| v2 Parameters | 25 |
| Loaded from v1 | 15 |
| Missing (new layers) | 10 |
| cls2 Weight Norm (before) | 0.8318 |
| cls2 Weight Norm (after) | 0.8318 |
| v2 Test Accuracy (before fine-tune) | 12/27 (44.4%) |

### 3-Phase Fine-Tuning with EWC (lambda=100.0)

| Epoch | CE Loss | EWC Penalty | Total Loss |
|---|---|---|---|
| 0 | 1.235 | 0.00000 | 1.235 |
| 20 | 0.156 | 0.00100 | 0.256 |
| 40 | 0.051 | 0.00059 | 0.110 |
| 60 | 0.019 | 0.00038 | 0.057 |
| 80 | 0.009 | 0.00025 | 0.035 |

| Metric | Value |
|---|---|
| Training Epochs | 100 |
| Learning Rate | 0.005 |
| Test Accuracy (after ft) | 16/27 (59.3%) |

### Dynamic Feature Expansion (12 -> 16)

| Epoch | Loss |
|---|---|
| 0 | 0.10 |
| 20 | 0.25 |
| 40 | 0.07 |
| 60 | 0.03 |
| 80 | 0.01 |

| Metric | Value |
|---|---|
| Training Epochs | 100 |
| Learning Rate | 0.005 |
| Final Test Accuracy | 17/27 (63.0%) |
