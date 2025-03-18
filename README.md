# Object Detection and Segmentation

A PyTorch implementation of YOLOv5 for object detection and segmentation, optimized for memory efficiency and ease of use.

## Features

- Custom YOLOv5 implementation with:
  - Backbone with CSP (Cross Stage Partial) blocks
  - Feature Pyramid Network (FPN)
  - Multi-scale detection heads
  - Memory-efficient gradient checkpointing
- Data augmentation pipeline using Albumentations:
  - Random resized crop
  - Horizontal flip
  - Color jittering
  - Adaptive padding
- Memory optimizations:
  - Gradient checkpointing
  - Configurable batch sizes and image sizes
  - Efficient data loading
  - CUDA memory management
- Training features:
  - Automatic mixed precision (AMP)
  - Cosine learning rate scheduling
  - TensorBoard logging
  - Checkpoint saving

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch >= 2.0
- Albumentations
- OpenCV
- TensorBoard
- PyYAML

## Dataset Structure

Organize your dataset in the following structure:
```
data/
├── train/
│   └── images/
├── train_labels/
│   └── labels/
├── val/
│   └── images/
├── val_labels/
│   └── labels/
└── dataset.yaml
```

The `dataset.yaml` should contain:
```yaml
train: data/train
train_labels: data/train_labels
val: data/val
val_labels: data/val_labels
names: ['class1', 'class2', ...]  # List of class names
```

## Training

Basic training command:
```bash
python train.py --data data/dataset.yaml --epochs 30 --batch-size 4 --img-size 416
```

### Training Arguments

- `--data`: Path to dataset.yaml file (required)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--img-size`: Input image size (default: 640)
- `--weights`: Path to pretrained weights (optional)
- `--device`: Training device ('cuda' or 'cpu', default: 'cuda')

### Memory Optimization

For systems with limited GPU memory:
1. Reduce batch size (e.g., --batch-size 2)
2. Reduce image size (e.g., --img-size 384)
3. Enable gradient checkpointing (enabled by default)
4. Adjust worker count in dataloader (default: 2)

### Model Architecture

The YOLOv5 implementation includes:
- Backbone: CSP-based feature extractor
- Neck: Feature Pyramid Network (FPN)
- Head: Multi-scale detection heads for different object sizes

Loss computation includes:
- Box regression loss (GIoU)
- Objectness loss (BCE)
- Classification loss (CrossEntropy)

## Monitoring

Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir runs
```

Metrics tracked:
- Training loss (per batch and epoch)
- Validation loss (when validation set is available)
- Learning rate
- Best model checkpoints

## Checkpoints

The training script saves:
- `weights/best.pt`: Best model based on validation loss (or training loss if no validation set)
- `weights/last.pt`: Latest model state

## License

This project is released under the MIT License.

## Acknowledgments

This implementation is inspired by the original YOLOv5 architecture while incorporating memory optimizations and modern PyTorch practices. 