import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import random

def plot_bbox(img, bbox, class_name, color=(255, 0, 0)):
    """
    Plot a bounding box on an image
    bbox format: [x_center, y_center, width, height] (normalized)
    """
    h, w = img.shape[:2]
    x_center, y_center, width, height = bbox
    
    # Convert normalized coordinates to pixel coordinates
    x_center *= w
    y_center *= h
    width *= w
    height *= h
    
    # Calculate corner points
    x1 = int(x_center - width/2)
    y1 = int(y_center - height/2)
    x2 = int(x_center + width/2)
    y2 = int(y_center + height/2)
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Add label
    cv2.putText(img, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def visualize_sample(img_path, label_path, class_names):
    """
    Visualize a single image with its annotations
    """
    # Read image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Read annotations
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            annotations = f.readlines()
        
        # Plot each bbox
        for ann in annotations:
            class_idx, *bbox = map(float, ann.strip().split())
            class_name = class_names[int(class_idx)]
            img = plot_bbox(img, bbox, class_name)
    
    return img

def visualize_dataset(data_yaml, num_samples=9):
    """
    Visualize random samples from the dataset
    """
    # Load dataset configuration
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get class names
    class_names = data_config['names']
    
    # Get image and label paths
    img_dir = Path(data_config['train'])
    label_dir = Path(data_config['train_labels'])
    
    # Get all image files
    img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    
    # Select random samples
    samples = random.sample(img_files, min(num_samples, len(img_files)))
    
    # Create subplot grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Dataset Samples with Annotations', fontsize=16)
    
    for i, img_path in enumerate(samples):
        # Get corresponding label path
        label_path = label_dir / f"{img_path.stem}.txt"
        
        # Visualize sample
        img = visualize_sample(img_path, label_path, class_names)
        
        # Plot
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.show()

def analyze_dataset(data_yaml):
    """
    Print dataset statistics
    """
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    splits = ['train', 'val', 'test']
    class_names = data_config['names']
    
    print("\nDataset Statistics:")
    print("-" * 50)
    
    for split in splits:
        img_dir = Path(data_config[split])
        label_dir = Path(data_config[f'{split}_labels'])
        
        # Count images and labels
        n_images = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
        n_labels = len(list(label_dir.glob('*.txt')))
        
        print(f"\n{split.upper()} Set:")
        print(f"  Images: {n_images}")
        print(f"  Labels: {n_labels}")
        
        # Count objects per class
        class_counts = {i: 0 for i in range(len(class_names))}
        for label_file in label_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_idx = int(line.strip().split()[0])
                    class_counts[class_idx] += 1
        
        print("\n  Objects per class:")
        for class_idx, count in class_counts.items():
            print(f"    {class_names[class_idx]}: {count}")

if __name__ == '__main__':
    # Visualize samples from the dataset
    visualize_dataset('data/dataset.yaml')
    
    # Print dataset statistics
    analyze_dataset('data/dataset.yaml') 