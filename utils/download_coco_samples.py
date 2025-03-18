import os
import json
import requests
from PIL import Image
from tqdm import tqdm
import numpy as np

def download_coco_samples(num_images=100):
    """
    Download sample images from COCO dataset
    """
    # COCO API URLs
    COCO_URL = "https://cocodataset.org/#download"
    COCO_API = "https://api.cocodataset.org/v1/coco"
    
    # Create directories
    os.makedirs('data/train/images', exist_ok=True)
    os.makedirs('data/train/labels', exist_ok=True)
    
    # COCO category IDs for our classes
    coco_categories = {
        1: 'person',
        3: 'car',
        4: 'motorcycle',
        6: 'bus',
        8: 'truck',
        10: 'traffic light',
        13: 'stop sign',
        2: 'bicycle',
        18: 'dog',
        17: 'cat'
    }
    
    # Our class mapping
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
    
    print("Downloading COCO sample images...")
    
    # Download and process images
    for i in tqdm(range(num_images)):
        try:
            # Get a random COCO image URL (this is a simplified example)
            # In practice, you would use the COCO API to get actual image URLs
            img_id = np.random.randint(1, 100000)
            img_url = f"http://images.cocodataset.org/train2017/{img_id:012d}.jpg"
            
            # Download image
            response = requests.get(img_url)
            if response.status_code == 200:
                img_path = f"data/train/images/{img_id:012d}.jpg"
                with open(img_path, 'wb') as f:
                    f.write(response.content)
                
                # Create dummy annotations (in practice, you would get these from COCO API)
                img = Image.open(img_path)
                width, height = img.size
                
                # Create YOLO format annotations
                annotations = []
                for _ in range(np.random.randint(1, 5)):  # Random number of objects
                    class_id = np.random.choice(list(class_mapping.values()))
                    
                    # Random bbox in YOLO format (x_center, y_center, width, height)
                    x_center = np.random.uniform(0.2, 0.8)
                    y_center = np.random.uniform(0.2, 0.8)
                    w = np.random.uniform(0.1, 0.3)
                    h = np.random.uniform(0.1, 0.3)
                    
                    annotations.append(f"{class_id} {x_center} {y_center} {w} {h}")
                
                # Save annotations
                with open(f"data/train/labels/{img_id:012d}.txt", 'w') as f:
                    f.write('\n'.join(annotations))
            
        except Exception as e:
            print(f"Error downloading image {img_id}: {e}")
            continue

if __name__ == '__main__':
    download_coco_samples() 