# Object Detection and Segmentation

A PyTorch implementation of YOLOv5 for object detection and segmentation, optimized for memory efficiency and ease of use. Features both CPU and GPU support with Docker deployment.

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
- Deployment features:
  - Docker containerization
  - GPU acceleration support
  - React-based web interface
  - RESTful API endpoints

## Requirements

### Local Development
```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.9+
- Flask 2.3.3
- PyTorch >= 2.0
- Albumentations
- OpenCV
- TensorBoard
- PyYAML

### Docker Deployment
- Docker Engine
- Docker Compose
- NVIDIA Container Toolkit (for GPU support)
- NVIDIA GPU with CUDA support (optional)

## Docker Deployment

### Prerequisites for GPU Support

1. Install NVIDIA Container Toolkit:
```bash
# Add NVIDIA package repositories
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

2. Verify GPU support:
```bash
sudo docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### Running the Application

1. Build and start the containers:
```bash
docker-compose up --build
```

2. Access the application:
- Web interface: http://localhost
- API endpoint: http://localhost:5000

The application will automatically use GPU acceleration if available.

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

## API Endpoints

The server provides the following REST API endpoints:

- `POST /predict`
  - Accepts: Multipart form data with an image file
  - Returns: JSON with detection results including bounding boxes and class labels

## Web Interface

The React-based web interface provides:
- Image upload functionality
- Real-time object detection visualization
- Detection results display
- Responsive design for various screen sizes

## Checkpoints

The training script saves:
- `weights/best.pt`: Best model based on validation loss (or training loss if no validation set)
- `weights/last.pt`: Latest model state

## Server Configuration

The server uses a `config.yaml` file located in the `server` directory to manage various settings:

```yaml
model:
  weights_path: weights/best.pt  # Path to model weights
  img_size: 640                  # Input image size
  conf_thres: 0.25              # Confidence threshold
  iou_thres: 0.45               # NMS IoU threshold
  max_det: 1000                 # Maximum detections per image
  device: cuda                  # Device to run on (cuda or cpu)

classes:
  names: [
    'person', 'bicycle', 'car', 'motorcycle', 'bus',
    'truck', 'traffic light', 'stop sign', 'bench', 'bird'
  ]

server:
  host: 0.0.0.0                 # Server host
  port: 5000                    # Server port
  debug: false                  # Debug mode

preprocessing:
  mean: [0.485, 0.456, 0.406]  # Image normalization mean
  std: [0.229, 0.224, 0.225]   # Image normalization std
  resize_mode: letterbox        # Image resize mode (letterbox or stretch)
```

### Important Notes:
1. The number of classes in `config.yaml` must match the model checkpoint's number of classes
2. The default configuration is set up for a 10-class model trained on a subset of COCO dataset
3. If using a custom model, update the class names and ensure the model architecture matches the number of classes
4. The model's detection heads expect (num_classes + 5) × 3 output channels for each prediction:
   - num_classes: number of object classes to detect
   - 5: four box coordinates plus one objectness score
   - 3: number of anchors per detection head

## License

This project is released under the MIT License.

## Acknowledgments

This implementation is inspired by the original YOLOv5 architecture while incorporating memory optimizations and modern PyTorch practices. 