import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, augment=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        self.augment = augment
        
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                border_value=[114, 114, 114]
            ),
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.8, 1.0),
                ratio=(0.8, 1.2),
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        self.val_transform = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=0,
                border_value=[114, 114, 114]
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = np.array(Image.open(img_path).convert('RGB'))
        
        # Load labels
        label_path = os.path.join(self.label_dir, 
                                self.img_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        
        boxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    class_label = int(data[0])
                    # YOLO format: class_idx, x_center, y_center, width, height
                    box = [float(x) for x in data[1:]]
                    boxes.append(box)
                    class_labels.append(class_label)

        boxes = np.array(boxes)
        class_labels = np.array(class_labels)

        # Apply augmentations
        if self.augment:
            transformed = self.transform(image=img, bboxes=boxes, class_labels=class_labels)
        else:
            transformed = self.val_transform(image=img, bboxes=boxes, class_labels=class_labels)

        image = transformed['image']
        boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32) if len(transformed['bboxes']) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        class_labels = torch.tensor(transformed['class_labels'], dtype=torch.long) if len(transformed['class_labels']) > 0 else torch.zeros(0, dtype=torch.long)

        return {
            'image': image,
            'boxes': boxes,
            'labels': class_labels,
            'img_path': img_path
        }

def create_dataloader(img_dir, label_dir, batch_size=16, img_size=640, augment=True, shuffle=True):
    dataset = YOLODataset(img_dir, label_dir, img_size=img_size, augment=augment)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=False,
        persistent_workers=True
    )
    return dataloader

def collate_fn(batch):
    images = []
    boxes = []
    labels = []
    img_paths = []
    
    for item in batch:
        images.append(item['image'])
        boxes.append(item['boxes'])
        labels.append(item['labels'])
        img_paths.append(item['img_path'])
    
    images = torch.stack(images)
    
    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'img_paths': img_paths
    } 