import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.yolo import create_model
from utils.datasets import create_dataloader

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    parser.add_argument('--weights', type=str, required=True, help='Path to weights file')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--source', type=str, required=True, help='Path to image or directory')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--save-dir', type=str, default='output', help='Directory to save results')
    return parser.parse_args()

def preprocess_image(image_path, img_size):
    # Read and preprocess image
    image = Image.open(image_path).convert('RGB')
    # Resize
    ratio = img_size / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    
    # Create canvas
    new_image = Image.new('RGB', (img_size, img_size), (114, 114, 114))
    new_image.paste(image, ((img_size - new_size[0]) // 2,
                           (img_size - new_size[1]) // 2))
    
    # Convert to numpy and normalize
    image = np.array(new_image) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float()

def plot_predictions(image, boxes, scores, labels, class_names, save_path):
    # Create figure and axes
    fig, ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image)
    
    # Plot each box
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                               edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        plt.text(x1, y1, f'{class_names[label]}: {score:.2f}',
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def run_inference(args):
    # Initialize
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load data config
    with open(args.data) as f:
        data_config = yaml.safe_load(f)
    num_classes = len(data_config['names'])
    
    # Create model and load weights
    model = create_model(num_classes)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Process source path
    source = Path(args.source)
    if source.is_file():
        paths = [source]
    else:
        paths = list(source.glob('*.jpg')) + list(source.glob('*.png'))
    
    # Run inference
    with torch.no_grad():
        for image_path in paths:
            # Preprocess image
            image = preprocess_image(str(image_path), args.img_size)
            image = image.unsqueeze(0).to(device)
            
            # Forward pass
            predictions = model(image)
            
            # Post-process predictions
            # This is a placeholder - implement actual post-processing based on your model's output format
            boxes = predictions[0]  # Assuming predictions returns [boxes, scores, labels]
            scores = predictions[1]
            labels = predictions[2]
            
            # Filter by confidence
            mask = scores > args.conf_thres
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            # Apply NMS
            # Implement Non-Maximum Suppression here
            
            # Save results
            orig_image = cv2.imread(str(image_path))
            orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            
            save_path = save_dir / f'{image_path.stem}_pred{image_path.suffix}'
            plot_predictions(orig_image, boxes.cpu().numpy(),
                           scores.cpu().numpy(),
                           labels.cpu().numpy(),
                           data_config['names'],
                           str(save_path))
            
            print(f'Processed {image_path}')

if __name__ == '__main__':
    args = parse_args()
    run_inference(args) 