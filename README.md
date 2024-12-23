# Chest X-Ray Analysis Project

# Important Note on Model Performance
The current model's performance metrics should be understood in the context of medical image analysis and class imbalance:
While the overall metrics might appear lower than previous versions, this represents a more clinically relevant approach to disease detection:

The dataset has significant class imbalance (~94% no findings vs ~6% findings)
Earlier versions achieved higher metrics by being overly conservative, mostly predicting "no finding"
The current model actively attempts to detect positive cases, leading to:

More true positive detections
Higher false positive rate
Lower overall metrics but potentially more clinical utility



Example from current results:

Infiltration: 154 true positives with 299 false positives
Effusion: 129 true positives with 400 false positives

This trade-off is intentional: in medical contexts, it's often preferable to have false positives (which can be verified by doctors) than to miss conditions by always predicting negative.

## Overview
This project implements a deep learning system for analyzing chest X-ray images, focusing on multi-label classification of various thoracic conditions. The system now utilizes DenseNet121 architecture with dual pathways for global and local feature analysis.

## Key Features
- Multi-label classification of 14 different thoracic conditions
- DenseNet121 backbone with global and local paths
- Channel and Spatial Attention (CBAM)
- Feature Pyramid Network (FPN) for multi-scale analysis
- GradCAM visualization for model interpretability
- Robust training pipeline with warmup and adaptive learning rates
- Interactive visualization interface using Gradio
- Comprehensive metrics tracking and evaluation

## Project Structure
```
chest_xray_project/
├── data/
│   ├── raw/          # Original X-ray images
│   ├── processed/    # Preprocessed images
│   └── metadata/     # Dataset metadata
├── src/
│   ├── data/         # Data handling
│   ├── models/       # Model architectures
│   ├── training/     # Training utilities
│   └── visualization/# Visualization tools
├── configs/          # Configuration files
├── notebooks/        # Jupyter notebooks
└── tests/           # Unit tests
```

## Technical Details

### Current Model Architecture (DenseNet Implementation)
- Base: DenseNet121 with pretrained ImageNet weights
- Dual pathway processing:
  - Global path for full image context
  - Local path for detailed features
- Enhanced with:
  - Channel and Spatial Attention (CBAM)
  - Feature Pyramid Network (FPN)
  - Multi-scale feature fusion
  - Advanced classifier head with dropout

### Training Pipeline
- Focal Loss for handling class imbalance
- AdamW optimizer with differential learning rates:
  - DenseNet layers: lr/20
  - BatchNorm layers: lr/20
  - New layers: full learning rate
- Learning rate scheduling with warmup
- Early stopping with patience
- Checkpoint saving and management
- Comprehensive metrics tracking

### Data Processing

#### Dataset Details
- Over 100,000 chest X-ray images from more than 30,000 unique patients
- 14 different thoracic conditions including:
  - Pneumonia
  - Atelectasis
  - Cardiomegaly
  - Effusion
  - Infiltration
  - Mass
  - Nodule
  - Pneumothorax
  - Consolidation
  - Edema
  - Emphysema
  - Fibrosis
  - Pleural Thickening
  - Hernia

#### Processing Pipeline
- Dynamic data loading with caching
- Robust preprocessing:
  - Resizing to 224x224 pixels
  - Normalization using ImageNet statistics
  - Contrast enhancement
- Data augmentation techniques
- Multi-threaded data loading
- Handling of multi-label cases

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- CUDA-capable GPU (recommended)

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using the Model

#### Basic Setup
```python
import torch
from src.models.cnn import ChestXrayModel
from src.training.trainer import XRayTrainer
from torch.utils.data import DataLoader

# Initialize model
model = ChestXrayModel(num_classes=14, pretrained=True)
```

#### Training
```python
trainer = XRayTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=14,
    lr=1e-4,
    device=device
)

best_metrics = trainer.train(
    num_epochs=10,
    early_stopping_patience=7
)
```

## Future Improvements
- Implement mixed precision training
- Optimize training speed
- Enhanced data augmentation
- Better handling of class imbalance
- Advanced feature fusion techniques

## OLD MODEL (check commit history)
Previous implementation using ResNet50 backbone with simpler architecture achieved different performance characteristics. See commit history for details of the previous implementation.

## Acknowledgments
- NIH Clinical Center for the Chest X-ray dataset
- PyTorch team for the deep learning framework
- Medical imaging community for research insights