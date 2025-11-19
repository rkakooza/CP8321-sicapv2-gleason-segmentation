# SICAPv2 Gleason Grade Segmentation with U-Net

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
