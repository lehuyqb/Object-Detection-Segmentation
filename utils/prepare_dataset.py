import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert bounding box from [x1, y1, x2, y2] to YOLO format [x_center, y_center, width, height]
    All values are normalized between 0 and 1
    """
    x1, y1, x2, y2 = bbox
    
    # Convert to center format
    width = (x2 - x1)
    height = (y2 - y1)
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return [x_center, y_center, width, height]

def convert_to_yolo_format(input_dir, output_dir, class_mapping, split_ratio={'train': 0.7, 'val': 0.2, 'test': 0.1}):
    """
    Convert a dataset to YOLO format and split into train/val/test sets
    
    Args:
        input_dir: Directory containing images and annotations
        output_dir: Directory to save the converted dataset
        class_mapping: Dictionary mapping class names to class indices
        split_ratio: Dictionary with train/val/test split ratios
    """
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(os.path.join(input_dir, 'images')) 
                  if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Calculate split indices
    n_files = len(image_files)
    n_train = int(n_files * split_ratio['train'])
    n_val = int(n_files * split_ratio['val'])
    
    # Split files
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # Process each split
    for split_name, files in splits.items():
        print(f"\nProcessing {split_name} split...")
        for img_file in tqdm(files):
            # Copy image
            src_img = os.path.join(input_dir, 'images', img_file)
            dst_img = os.path.join(output_dir, split_name, 'images', img_file)
            shutil.copy2(src_img, dst_img)
            
            # Read image dimensions
            img = cv2.imread(src_img)
            img_height, img_width = img.shape[:2]
            
            # Convert and save annotations
            ann_file = os.path.join(input_dir, 'annotations', 
                                  img_file.rsplit('.', 1)[0] + '.txt')
            if os.path.exists(ann_file):
                with open(ann_file, 'r') as f:
                    annotations = f.readlines()
                
                yolo_annotations = []
                for ann in annotations:
                    # Parse your annotation format and convert to YOLO format
                    # This is an example - modify according to your annotation format
                    class_name, x1, y1, x2, y2 = ann.strip().split()
                    class_idx = class_mapping[class_name]
                    bbox = [float(x1), float(y1), float(x2), float(y2)]
                    yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
                    
                    yolo_annotations.append(f"{class_idx} {' '.join(map(str, yolo_bbox))}")
                
                # Save YOLO format annotations
                dst_ann = os.path.join(output_dir, split_name, 'labels',
                                     img_file.rsplit('.', 1)[0] + '.txt')
                with open(dst_ann, 'w') as f:
                    f.write('\n'.join(yolo_annotations))

def create_dataset_yaml(output_dir, class_names):
    """
    Create dataset.yaml file with paths and class names
    """
    yaml_content = f"""# Dataset paths
train: {os.path.join(output_dir, 'train/images')}  # Path to training images
train_labels: {os.path.join(output_dir, 'train/labels')}  # Path to training labels
val: {os.path.join(output_dir, 'val/images')}  # Path to validation images
val_labels: {os.path.join(output_dir, 'val/labels')}  # Path to validation labels
test: {os.path.join(output_dir, 'test/images')}  # Path to test images
test_labels: {os.path.join(output_dir, 'test/labels')}  # Path to test labels

# Classes
names:
{chr(10).join(f'  {i}: {name}' for i, name in enumerate(class_names))}

# Training parameters
nc: {len(class_names)}  # number of classes
img_size: 640  # training image size"""

    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

def verify_dataset(dataset_dir):
    """
    Verify the dataset structure and print statistics
    """
    for split in ['train', 'val', 'test']:
        img_dir = Path(dataset_dir) / split / 'images'
        label_dir = Path(dataset_dir) / split / 'labels'
        
        n_images = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
        n_labels = len(list(label_dir.glob('*.txt')))
        
        print(f"\n{split} set:")
        print(f"  Images: {n_images}")
        print(f"  Labels: {n_labels}")
        
        if n_images != n_labels:
            print(f"  Warning: Number of images ({n_images}) does not match number of labels ({n_labels})")

if __name__ == '__main__':
    # Example usage
    class_mapping = {
        'person': 0,
        'car': 1,
        'motorcycle': 2,
        'bus': 3,
        'truck': 4,
        'traffic light': 5,
        'stop sign': 6,
        'bicycle': 7,
        'dog': 8,
        'cat': 9
    }
    
    # Convert dataset
    input_dir = 'path/to/your/original/dataset'
    output_dir = 'data'
    
    # Uncomment these lines when you have your input dataset ready
    # convert_to_yolo_format(input_dir, output_dir, class_mapping)
    # create_dataset_yaml(output_dir, list(class_mapping.keys()))
    # verify_dataset(output_dir) 