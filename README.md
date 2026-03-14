# Yoga Pose Classification: MediaPipe vs. Deep Learning

Stanford CS229 Final Project — Lara Selin Seyahi

## Overview
This project compares landmark-based classical classifiers and fine-tuned CNNs 
for yoga pose classification on the Yoga-82 dataset. Models are evaluated across 
3 label levels: 6, 20, and 82 classes.

## Approaches
- **Phase 1**: Classical classifiers (Random Forest, SVM, KNN, MLP) on raw 
  MediaPipe landmarks (132-dim)
- **Phase 2**: Same classifiers on engineered geometric features (joint angles, 
  pairwise distances, symmetry measures) combined with raw landmarks (164-dim)
- **Phase 3**: Fine-tuned ResNet-18 and ResNet-50 CNNs on raw images

## Key Results
| Model | 6-class | 20-class | 82-class |
|-------|---------|----------|----------|
| SVM (combined features) | 93.6% | 90.6% | 84.2% |
| ResNet-18 | 89.8% | 84.8% | 78.9% |
| ResNet-50 | 93.0% | 89.9% | 85.3% |

SVM with combined features outperforms ResNet-18 and approaches ResNet-50 at 
the 82-class level, demonstrating that structured geometric representations are 
highly competitive with deep learning for pose classification.

## Dataset
[Yoga-82](https://sites.google.com/view/yoga-82/home) — a fine-grained yoga 
pose dataset with 3-level hierarchical labels. Download the dataset and place 
it in a `yoga-82/` folder.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Phase 1: Classical classifiers on raw landmarks
python classify_phase1.py

# Phase 2: Feature engineering
python feature_engineering_phase2.py

# Phase 3: CNN fine-tuning
python resnet18_phase3.py
python resnet50_phase3.py
```

## Files
- `data_prep.py` — dataset preparation and MediaPipe landmark extraction
- `extract_features.py` — feature extraction
- `classify_phase1.py` — Phase 1 classifiers
- `feature_engineering_phase2.py` — Phase 2 engineered features
- `resnet18_phase3.py` / `resnet18_phase3.ipynb` — ResNet-18 fine-tuning
- `resnet50_phase3.py` / `resnet50_phase3.ipynb` — ResNet-50 fine-tuning
