# Chest X-Ray Analysis Project

## Overview
This project implements a deep learning system for analyzing chest X-ray images, with a primary focus on multi-label classification of various thoracic conditions. The system utilizes state-of-the-art deep learning techniques including attention mechanisms, feature pyramid networks, and advanced visualization tools.

## Key Features
- Multi-label classification of 14 different thoracic conditions
- Advanced attention mechanisms (CBAM) for improved feature extraction
- Feature Pyramid Network (FPN) for multi-scale analysis
- GradCAM visualization for model interpretability
- Robust training pipeline with early stopping and checkpointing
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

### Model Architecture
- Base: ResNet50 with pretrained ImageNet weights
- Enhanced with:
  - Channel and Spatial Attention (CBAM)
  - Feature Pyramid Network (FPN)
  - Multi-scale feature fusion
  - Advanced classifier head with dropout

### Training Pipeline
- Focal Loss for handling class imbalance
- Adam optimizer with learning rate scheduling
- Early stopping with patience
- Checkpoint saving and management
- Comprehensive metrics tracking:
  - AUC-ROC
  - F1 Score
  - Precision/Recall
  - Accuracy

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
- Dynamic data loading with caching for improved performance
- Robust preprocessing:
  - Resizing to 224x224 pixels
  - Normalization using ImageNet statistics
  - Contrast enhancement for better feature visibility
- Data augmentation techniques:
  - Random horizontal flips
  - Random rotations (±10 degrees)
  - Random brightness and contrast adjustments
- Multi-threaded data loading for training efficiency
- Handling of multi-label cases (patients with multiple conditions)

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

### Data Setup
1. **Download the NIH Chest X-ray Dataset**:
   - Visit the [NIH Clinical Center website](https://nihcc.app.box.com/v/ChestXray-NIHCC)
   - Download the following files:
     - All image files (images_001.tar.gz to images_012.tar.gz)
     - Data_Entry_2017_v2020.csv (contains image labels)
     - BBox_List_2017.csv (if using localization features)

2. **Organize the Data**:
   ```bash
   # Create necessary directories
   mkdir -p data/raw
   mkdir -p data/metadata
   
   # Extract images to data/raw
   cd data/raw
   for f in images_*.tar.gz; do
     tar -xzf "$f"
   done
   
   # Move metadata files to data/metadata
   mv Data_Entry_2017_v2020.csv ../metadata/
   mv BBox_List_2017.csv ../metadata/  # if downloaded
   ```

3. **Verify Setup**:
   ```bash
   # Your directory structure should look like:
   chest_xray_project/
   ├── data/
   │   ├── raw/
   │   │   ├── images/
   │   │   │   ├── 00000001_000.png
   │   │   │   ├── 00000001_001.png
   │   │   │   └── ...
   │   └── metadata/
   │       └── Data_Entry_2017_v2020.csv
   │      
   ```

### Using the Model

#### 1. Basic Setup
```python
import torch
from pathlib import Path
from src.data.preprocessing import XRayPreprocessor
from src.data.dataset import ChestXrayDataset
from src.models.cnn import ChestXrayModel
from src.training.trainer import XRayTrainer
from torch.utils.data import DataLoader

# Define the condition mapping
findings_dict = {
    'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2,
    'Infiltration': 3, 'Mass': 4, 'Nodule': 5,
    'Pneumonia': 6, 'Pneumothorax': 7, 'Consolidation': 8,
    'Edema': 9, 'Emphysema': 10, 'Fibrosis': 11,
    'Pleural_Thickening': 12, 'Hernia': 13
}
```

#### 2. Training a New Model
```python
# Initialize preprocessor
preprocessor = XRayPreprocessor(
    target_size=(224, 224),
    normalize_method='standard',
    train_split=0.70,
    val_split=0.15,
    test_split=0.15
)

# Create datasets
train_dataset = ChestXrayDataset(
    folder_path=images_path,
    image_paths=image_splits['train'],
    labels=label_splits['train'],
    preprocessor=preprocessor
)
val_dataset = ChestXrayDataset(
    folder_path=images_path,
    image_paths=image_splits['val'],
    labels=label_splits['val'],
    preprocessor=preprocessor
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize and train model
model = ChestXrayModel(num_classes=14, pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch.device('mps' if torch.mps.is_available() else 'cpu') if Mac

trainer = XRayTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_classes=14,
    lr=5e-5,
    checkpoint_dir='checkpoints',
    device=device
)

# Train the model
best_metrics = trainer.train(
    num_epochs=10,
    early_stopping_patience=3
)
```

#### 3. Loading a Pretrained Model
```python
# Load model from checkpoint
model = ChestXrayModel(num_classes=14)
checkpoint_path = 'path/to/checkpoint.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()  # Set to evaluation mode
```

#### 4. Batch Evaluation
```python
# Evaluate on test set
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
test_trainer = XRayTrainer(
    model=model,
    train_loader=None,
    val_loader=test_loader,
    device=device
)
test_metrics = test_trainer._validate_epoch()
```

## Model Performance
The model achieves competitive performance on chest X-ray analysis:
- High sensitivity for pneumonia detection
- Robust performance across multiple conditions
- Interpretable predictions with attention visualization

## Visualization Tools (Not working)
- GradCAM heatmap generation
- Interactive Gradio interface for real-time analysis
- Attention map visualization
- Performance metric plotting

## Acknowledgments
- NIH Clinical Center for the Chest X-ray dataset
- PyTorch team for the deep learning framework
- Medical imaging community for research insights
