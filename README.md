# SICAPv2 Gleason Grade Segmentation with Attention U-Net

This project performs 4-class pixel-level segmentation of prostate histopathology
patches (SICAPv2 dataset) using a U-Net model.

Segmentation classes:

- **0**: Non-cancerous
- **1**: Gleason Grade 3 (GG3)
- **2**: Gleason Grade 4 (GG4)
- **3**: Gleason Grade 5 (GG5)

## Repository Structure

```
src/            # Core Python modules (dataset, model, losses, metrics, training)
notebooks/      # Jupyter notebooks for exploration and visualization
experiments/    # Outputs: cross-validation folds + final test results
report/         # Figures and tables for project write-up
data/           # Local copy of SICAPv2 dataset (not tracked by Git)
```

## Dataset

Download SICAPv2 from Mendeley and place its contents here:

```
data/SICAPv2/
    images/
    masks/
    partition/
        Test/
        Validation/
    wsi_labels.xlsx
```

> **Important:** The dataset is NOT included in this repository due to size.

> **Dataset Link:** https://data.mendeley.com/datasets/9xxm58dvs3/1

## Training Strategy

This project uses:

- Official patient-based partitions
- 4-fold cross-validation (Val1–Val4)
- Final training on the full training set
- Evaluation on the held-out global Test set

## Training via CLI (Single Run or Cross‑Validation)

All training, whether a single fold or full 4‑fold cross‑validation is performed using the same command-line interface:

```bash
python -m src.run_training --fold <FOLD> --epochs 40 --batch_size 4 --num_workers 0
```

**Arguments:**

- `--fold` — one of {Val1, Val2, Val3, Val4, final}
- `--epochs` — number of epochs
- `--batch_size` — mini‑batch size
- `--lr` — learning rate (default: 1e‑4)
- `--num_workers` — number of DataLoader workers (macOS users may need 0)

> **Note:** `num_workers=0` is the safest and most compatible setting across all platforms (Windows, macOS, Linux). Increase to `2–4` only if your system supports multi-process data loading without issues (Linux usually does; Windows often does not).

- `--data_root` — dataset directory (default: data/SICAPv2)
- `--out_dir` — output directory for experiment results (default: experiments/)

The script automatically:

- detects CUDA / MPS / CPU
- loads the correct Excel partition for the selected fold
- trains the Attention U‑Net
- evaluates on the corresponding validation or test set
- saves results to `experiments/<FOLD>/`

Outputs include:

- `best_model.pth`
- `val_metrics.json` (or `test_metrics.json` for `--fold final`)

---

### Running All 4 Folds (Cross‑Validation)

To run 4‑fold patient‑based cross‑validation, simply execute the training command once per fold:

```bash
python -m src.run_training --fold Val1 --epochs 40 --batch_size 4 --num_workers 0
python -m src.run_training --fold Val2 --epochs 40 --batch_size 4 --num_workers 0
python -m src.run_training --fold Val3 --epochs 40 --batch_size 4 --num_workers 0
python -m src.run_training --fold Val4 --epochs 40 --batch_size 4 --num_workers 0
```

Each run will populate:

```
experiments/Val1/
experiments/Val2/
experiments/Val3/
experiments/Val4/
```

We shall divide folds among ourselves.

---

### Training the Final Model

After cross‑validation, we can train the final model on all available training data:

```bash
python -m src.run_training --fold final --epochs 40 --batch_size 4 --num_workers 0
```

This uses the global Test set for evaluation and produces:

```
experiments/final/
    best_model.pth
    test_metrics.json
```

## Device Compatibility

The training pipeline automatically selects the best available compute device:

- **NVIDIA GPU (CUDA)** — if available on Linux/Windows
- **Apple Silicon GPU (M1/M2/M3 using MPS backend)** — for macOS users
- **CPU fallback** — if no GPU is detected

No manual configuration is required. The training script handles device detection internally:

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

This setup ensures anyone can train and run experiments regardless of hardware.

## Environment Setup

Create a conda environment:

```
# Create the environment from the file
conda env create -f environment.yml

# Activate the environment
conda activate sicap-env
```

If you prefer to create the environment manually instead of using environment.yml, run:

```
conda create -n sicap-env python=3.10
conda activate sicap-env
```

## Authors / Team

- Ronald Kakooza
- Victoria Buchanan
- Ghazal Mirsayyah
