# AnatoXAI — Anatomically-Aware Chest X-ray Classifier

AnatoXAI compares a standard **DenseNet-121** multi-label classifier against a custom **3-Expert Anatomical Mixture-of-Experts (MoE)** model for multi-label disease classification on chest X-rays, using the **NIH ChestX-ray14** dataset. The MoE model routes each image across three anatomically-grounded experts (Lung, Cardiac, Pleural), and the notebook also includes Grad-CAM visualizations for model interpretability.

## Overview

- **Task**: Multi-label classification of 14 chest X-ray findings
- **Baseline model**: DenseNet-121 (ImageNet-pretrained) with a linear classification head
- **Proposed model**: A 3-expert Mixture-of-Experts built on a shared DenseNet-121 feature extractor, with a learned router that weights contributions from Lung, Cardiac, and Pleural expert heads
- **Interpretability**: Grad-CAM on the DenseNet baseline, plus expert routing-weight analysis for the MoE
- **Dataset**: [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) (`Data_Entry_2017.csv` + PNG images)

## Disease Labels

The model predicts 14 findings:

```
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule,
Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis,
Pleural Thickening, Hernia
```

These are grouped into three anatomical expert domains:

| Expert  | Diseases |
|---------|----------|
| Lung    | Atelectasis, Infiltration, Pneumonia, Consolidation, Emphysema, Fibrosis |
| Cardiac | Cardiomegaly, Edema |
| Pleural | Effusion, Pneumothorax, Pleural Thickening |

## Model Architecture

### 1. DenseNet-121 Baseline
A pretrained DenseNet-121 backbone with its classifier head replaced by a linear layer producing 14 logits (one per disease).

### 2. 3-Expert Anatomical MoE

```
X-ray
  ↓
DenseNet feature extractor
  ↓
1024-d feature vector
  ↓
Router → [Lung, Cardiac, Pleural] softmax weights
  ↓
3 expert MLP heads → 14 logits each
  ↓
Weighted sum (routing weights × expert logits)
  ↓
14 disease logits
```

- The router is intentionally simple: it outputs **3 weights per image** (one per anatomical expert), not per-disease routing.
- Each expert is a small MLP (`Linear → ReLU → Dropout → Linear`) applied to the shared 1024-d DenseNet feature vector.
- An **auxiliary expert-specialization loss** encourages each expert to be accurate specifically on its own anatomical disease subset, added to the main BCE loss with a weight of 0.2.

## Training Details

- **Input size**: 224×224
- **Batch size**: 32
- **Optimizer**: AdamW (`lr=1e-4`, `weight_decay=1e-4`)
- **Loss**: `BCEWithLogitsLoss` with per-class `pos_weight` (clipped to [1, 50]) to handle class imbalance
- **Epochs**: 10 (baseline), 20 (MoE)
- **Augmentation** (train only): random horizontal flip, random rotation (±7°), color jitter
- **Normalization**: ImageNet mean/std
- **Split**: Patient-level `GroupShuffleSplit` (train/val/test ≈ 70/15/15) to prevent patient leakage across splits
- **Best checkpoint**: Selected by highest validation mean AUROC (MoE)

## Evaluation

For both models, the notebook computes, per disease and averaged:
- **AUROC** (Area under ROC curve)
- **AUPRC** (Area under Precision-Recall curve)
- **F1 score** (at 0.5 threshold)

It then produces:
- A summary table/bar chart comparing DenseNet vs. MoE on mean AUROC/AUPRC/F1
- A per-disease AUROC comparison bar chart
- Expert routing-weight analysis (average utilization of each expert across the test set)
- A single-sample inspection view showing routing weights and top predicted diseases

## Interpretability (Grad-CAM)

Grad-CAM is applied to the DenseNet baseline's final dense block to visualize which image regions drive the top predicted class, shown as:
1. Original X-ray
2. Grad-CAM heatmap
3. Heatmap overlaid on the original image

## Repository Structure (expected data layout)

```
DATA_ROOT/
├── Data_Entry_2017.csv        # NIH metadata (Image Index, Finding Labels, Patient ID, ...)
├── images_001/images/*.png
├── images_002/images/*.png
├── ...
└── images_012/images/*.png
```

Update `DATA_ROOT` in the configuration cell to point to your local copy of the NIH ChestX-ray14 dataset.

## Requirements

- Python 3.8+
- PyTorch + torchvision
- numpy, pandas, Pillow, tqdm
- matplotlib, seaborn
- scikit-learn

Install with:

```bash
pip install torch torchvision numpy pandas pillow tqdm matplotlib seaborn scikit-learn
```

A CUDA-capable GPU is strongly recommended for training.

## Usage

1. Download the NIH ChestX-ray14 dataset and update `DATA_ROOT` / `CSV_PATH` in the notebook's configuration cell.
2. Run the notebook top to bottom:
   - Load metadata and build multi-label targets
   - Create patient-level train/val/test splits
   - Train the DenseNet-121 baseline
   - Train the 3-Expert Anatomical MoE
   - Compare results and inspect expert routing
   - Run Grad-CAM on sample predictions
3. For inference on a single new image, use the `predict_single(model, image_path, is_moe=...)` helper defined near the end of the notebook.

## Outputs

Running the full notebook saves the following to an output directory (`thoraxmoe_outputs/` next to the data root):

- `densenet121_baseline.pth` — trained baseline weights
- `thoraxmoe_3expert.pth` — trained MoE weights
- `model_comparison.csv` — overall metric comparison
- `disease_comparison.csv` — per-disease metric comparison
- `expert_routing.csv` — average expert routing weights
- `config.json` — run configuration (seed, hyperparameters, disease/expert lists)

## Notes

- All experiments use a fixed seed (`SEED = 42`) for reproducibility.
- The MoE router is deliberately simple (image-level, 3-way) rather than disease-specific, to keep expert attribution interpretable.
- This is a research/educational notebook; it is **not** validated for clinical use.
