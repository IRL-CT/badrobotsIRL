# BADNet PyTorch Implementation

PyTorch implementation of the BADNet CNN for robot behavior classification with inter-participant cross-validation and Weights & Biases integration.

## Files

- **`badnet_pytorch.py`**: Main implementation with W&B sweep support
- **`train_badnet.py`**: Standalone training script (no W&B required)
- **`create_image_splits.py`**: Utilities for inter-participant data splits

## Requirements

```bash
pip install torch torchvision numpy pandas scikit-learn pillow wandb
```

## Data Structure

Expected data organization:
```
project/
├── preprocessing/
│   └── full_features/
│       └── all_participants_0_3.csv
├── data/
│   └── frames/
│       ├── p10nodbot/
│       │   ├── 1.jpg
│       │   ├── 2.jpg
│       │   └── ...
│       ├── p11nodbot/
│       │   └── ...
│       └── ...
└── code/badnet/
    ├── badnet_pytorch.py
    ├── train_badnet.py
    └── create_image_splits.py
```

The CSV file should have columns:
- `frame`: Frame number (matches jpg filename)
- `participant`: Participant ID (matches folder name)
- `binary_label`: Binary label (0 or 1)
- `multiclass_label`: Multiclass label (0, 1, 2, or 3)

## Usage

### 1. Run with W&B Sweep

```bash
python badnet_pytorch.py
```
or

```bash
python train_badnet.py
```

This will:
- Create a W&B sweep with various hyperparameter combinations
- Run 5-fold inter-participant cross-validation for each configuration
- Log all metrics to W&B

Sweep parameters include:
- `label_type`: 'binary' or 'multiclass'
- `activation`: 'relu' or 'sigmoid'
- `kernel_size`: 2, 4, 6, or 8
- `base_filters`: 16, 32, or 64
- `learning_rate`: 0.0001, 0.001, or 0.01
- `batch_size`: 16, 32, or 64

### 2. Standalone Training

```bash
# Binary classification, all folds
python train_badnet.py --label_type binary

# Multiclass classification, specific fold
python train_badnet.py --label_type multiclass --fold 0

# Custom hyperparameters
python train_badnet.py \
    --activation relu \
    --kernel_size 4 \
    --base_filters 32 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --epochs 100

# Exclude specific participants
python train_badnet.py --exclude_participants p10nodbot p11nodbot

# Custom paths
python train_badnet.py \
    --csv_path /path/to/labels.csv \
    --image_base_path /path/to/frames
```

### 3. Validate Data Splits

```bash
python create_image_splits.py \
    --csv_path ../../preprocessing/full_features/all_participants_0_3.csv \
    --num_folds 5 \
    --seed 42 \
    --image_path ../../data/frames
```

## Model Architecture

The CNN architecture matches the original BadNet:
1. Conv2D (base_filters) + Dropout(0.15)
2. Conv2D (base_filters*2) + Dropout(0.2)
3. Conv2D (base_filters*4) + Dropout(0.2) + BatchNorm
4. Global Average Pooling
5. Flatten + Dropout(0.6)
6. Dense(128)
7. Dense(num_classes)

All conv layers use:
- Stride 2
- Same padding
- He uniform initialization

## Inter-Participant Cross-Validation

The implementation uses 5-fold inter-participant CV:
- Participants are divided into 5 groups
- Each fold uses one group as test set
- Remaining participants are split 70-30 for train-val
- Ensures no data leakage between train and test

Approximate split ratios:
- Train: ~56% of data
- Val: ~24% of data  
- Test: ~20% of data

## Excluding Participants

To exclude specific participants (e.g., for debugging or if data is incomplete):

**With W&B sweep** (modify `badnet_pytorch.py`):
```python
"exclude_participants": {"values": [['p10nodbot', 'p11nodbot']]},
```

**Standalone training**:
```bash
python train_badnet.py --exclude_participants p10nodbot p11nodbot
```

## Metrics

The following metrics are computed and logged:
- Accuracy
- Precision (macro average)
- Recall (macro average)
- F1 Score (macro average)
- Cohen's Kappa
- Confusion Matrix

## Checkpoints

Models are saved in `./checkpoints/fold_{i}/`:
- `best_model.pth`: Best model based on validation accuracy
- `final_model.pth`: Final model after training (standalone script only)

## GPU Usage

The code automatically uses CUDA if available. To force CPU:
```bash
python train_badnet.py --no_cuda
```

## Customization

### Add Data Augmentation

Modify the transform in the main function:
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Change Fold Ratios

Modify `create_interparticipant_folds()` in `create_image_splits.py` or use the `create_interparticipant_folds_custom_ratio()` function.

### Add New Metrics

Modify `evaluate_model()` in `badnet_pytorch.py` to compute additional metrics.

## Troubleshooting

1. **Out of Memory**: Reduce batch_size or num_workers
2. **Missing Images**: Run `create_image_splits.py` with `--image_path` to validate
3. **Slow Training**: Increase num_workers, ensure GPU is being used
4. **Poor Performance**: Try different hyperparameters, check class balance

## W&B Dashboard

When running sweeps, you can monitor:
- Training/validation loss and accuracy per fold
- Test metrics per fold
- Average metrics across folds
- Hyperparameter comparisons

Access your dashboard at: https://wandb.ai/your-username/BADNet_PyTorch
