# OSNet ReID — Player Re-Identification for Broadcast Football

Re-identification of football players across frames of broadcast match videos, built on top of the **OSNet** backbone and trained on the **SoccerNet v3 ReID** dataset.

## Results

| Split | mAP | Rank-1 |
|---|---|---|
| Validation | 83.7% | 78.5% |
| Test | 83.4% | 78.0% |

Evaluated with the SoccerNet action-to-replay metric (only Rank-1 and mAP are meaningful; Rank-5/Rank-10 are not defined for this metric).

---

## Architecture

- **Backbone**: OSNet-x1.0 — Omni-Scale Network, 2.19M parameters, 512-dim feature output
- **Loss**: Triplet loss (margin=0.5, weight=0.9) + Cross-Entropy with label smoothing (weight=0.5)
- **Sampling**: `RandomIdentitySampler_Hierarchical` — samples N identities × K instances per batch, with two-level hierarchy across cameras and actions
- **Input size**: 256×128 (H×W)

---

## Dataset — SoccerNet v3 ReID

| Split | Identities | Images | Cameras |
|---|---|---|---|
| Train | 161,443 | 248,234 | 9,189 |
| Query (valid) | 11,638 | 11,638 | 1,751 |
| Gallery (valid) | 29,534 | 34,355 | 1,751 |
| Query (test) | 11,777 | 11,777 | 1,715 |
| Gallery (test) | 30,059 | 34,989 | 1,715 |
| Query (challenge) | — | 9,021 | — |
| Gallery (challenge) | — | 26,082 | — |

Dataset access requires a SoccerNet password (request at [silvio.giancola@kaust.edu.sa](mailto:silvio.giancola@kaust.edu.sa)).

---

## Installation

```bash
bash install.sh
```

This will:
1. Install PyTorch with CUDA 12.4 (if not already installed)
2. Install Python dependencies
3. Build the Cython rank evaluation extension
4. Register the package via `.pth` file

**Requirements**: Python 3.8+, CUDA 12.x, NVIDIA GPU

---

## Download Dataset

```bash
python3 tools/download_data.py --password <your_soccernet_password>
```

Downloads train / valid / test / challenge splits into `benchmarks/baseline/datasets/soccernetv3/`.

---

## Training

```bash
# Default config (RTX 3090, 20 epochs)
bash train.sh

# Custom config
bash train.sh benchmarks/baseline/configs/reid_config.yaml

# Override specific params
bash train.sh benchmarks/baseline/configs/rtx3090_config.yaml train.max_epoch 30
```

### Config files

| File | Description |
|---|---|
| `configs/rtx3090_config.yaml` | Optimized for RTX 3090 — batch 256, 16 workers, AMP, cudnn.benchmark |
| `configs/reid_config.yaml` | General config — batch 128, 4 workers |
| `configs/baseline_config.yaml` | Minimal baseline |

### Key training settings (rtx3090_config)

```yaml
train:
  batch_size: 256
  lr: 0.0003
  max_epoch: 20
  stepsize: [15, 18]      # MultiStepLR milestones
  gamma: 0.1
  amp: True               # Automatic Mixed Precision (FP16 forward, FP32 loss)
  cudnn_benchmark: True

sampler:
  num_instances: 8        # 32 identities × 8 images per batch
```

---

## Evaluation

```bash
# Evaluate a checkpoint
bash test.sh checkpoints/checkpoint.pth

# Custom output directory
bash test.sh checkpoints/checkpoint.pth logs/my_eval
```

Produces:
- `logs/eval/test.log-<timestamp>` — full evaluation log
- `logs/eval/ranking_results_<dataset>_<timestamp>.json` — ranking export for external SoccerNet evaluator

---

## Project Structure

```
osnet_reid/
├── benchmarks/baseline/
│   ├── main.py                  # Training and evaluation entry point
│   ├── default_config.py        # All config defaults
│   └── configs/
│       ├── rtx3090_config.yaml
│       ├── reid_config.yaml
│       └── baseline_config.yaml
├── osnet_reid/
│   ├── models/osnet.py          # OSNet-x1.0 backbone
│   ├── data/
│   │   ├── datamanager.py       # Train/test data pipeline
│   │   ├── sampler.py           # RandomIdentitySampler_Hierarchical
│   │   └── transforms.py        # Augmentation (flip, crop, patch, erase)
│   ├── losses/
│   │   ├── hard_mine_triplet_loss.py
│   │   └── cross_entropy_loss.py
│   ├── engine/
│   │   └── image/triplet.py     # Forward/backward with AMP support
│   └── metrics/
│       └── rank_cylib/          # Cython-accelerated CMC/mAP computation
├── tools/
│   ├── download_data.py         # SoccerNet dataset downloader
│   ├── evaluate_soccernetv3_reid.py  # External evaluator
│   ├── visualize_actmap.py      # Activation map visualization
│   └── parse_test_res.py        # Parse results from log files
├── checkpoints/                 # Saved model weights
├── install.sh
├── train.sh
└── test.sh
```

---

## Data Augmentation

Applied during training:

| Transform | Description |
|---|---|
| Random flip | Horizontal flip with p=0.5 |
| Random crop | Enlarge to 288×144, crop back to 256×128 |
| Random patch | Swap random patches between images |
| Random erase | Randomly erase rectangular regions |

---

## Tools

| Script | Usage |
|---|---|
| `tools/download_data.py` | Download SoccerNet ReID splits |
| `tools/evaluate_soccernetv3_reid.py` | Evaluate ranking JSON against ground truth |
| `tools/visualize_actmap.py` | Visualize model attention maps |
| `tools/parse_test_res.py` | Extract metrics from training logs |
| `tools/create_model_soups.py` | Average multiple checkpoints (model soups) |
| `tools/compute_mean_std.py` | Compute dataset normalization stats |
