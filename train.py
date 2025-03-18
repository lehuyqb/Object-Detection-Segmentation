import argparse
import os
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import Config

from models.yolo import create_model
from utils.datasets import create_dataloader

# Set CUDA memory optimization settings
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cudnn
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Enable memory efficient features
torch.set_float32_matmul_precision('medium')  # Reduce precision for better memory efficiency

# Empty CUDA cache before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--weights', type=str, default='', help='Initial weights path')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or cpu')
    return parser.parse_args()

def train(args):
    # Initialize device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load data configuration
    with open(args.data) as f:
        data_config = yaml.safe_load(f)
    
    # Verify data paths exist
    required_paths = ['train', 'train_labels', 'val', 'val_labels']
    for path in required_paths:
        if not os.path.exists(data_config[path]):
            raise FileNotFoundError(f"Path '{data_config[path]}' specified in {args.data} does not exist")
    
    # Create model
    num_classes = len(data_config['names'])
    model = create_model(num_classes, pretrained=bool(args.weights))
    model = model.to(device)
    
    # Load pretrained weights if specified
    if args.weights:
        if not os.path.exists(args.weights):
            raise FileNotFoundError(f"Weights file '{args.weights}' does not exist")
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt['model'])
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Create dataloaders
    train_loader = create_dataloader(
        data_config['train'], 
        data_config['train_labels'],
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=True
    )
    
    val_loader = create_dataloader(
        data_config['val'],
        data_config['val_labels'],
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=False,
        shuffle=False
    )
    
    # Verify dataloaders
    if len(train_loader) == 0:
        raise RuntimeError("Training dataset is empty")
    if len(val_loader) == 0:
        print("Warning: Validation dataset is empty. Training will proceed without validation.")
    
    # Initialize tensorboard
    writer = SummaryWriter()
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{args.epochs}')
        
        for batch_i, batch in enumerate(pbar):
            images = batch['images'].to(device)
            targets = {
                'boxes': [b.to(device) for b in batch['boxes']],
                'labels': [l.to(device) for l in batch['labels']]
            }
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            loss = model.compute_loss(predictions, targets)
            epoch_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + batch_i)
        
        # Log epoch metrics
        epoch_loss /= len(train_loader)
        writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
        
        # Validation phase
        if len(val_loader) > 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validation'):
                    images = batch['images'].to(device)
                    targets = {
                        'boxes': [b.to(device) for b in batch['boxes']],
                        'labels': [l.to(device) for l in batch['labels']]
                    }
                    
                    predictions = model(images)
                    loss = model.compute_loss(predictions, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            writer.add_scalar('val/loss', val_loss, epoch)
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss
                }, 'weights/best.pt')
        else:
            # If no validation set, save based on training loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss
                }, 'weights/best.pt')
        
        # Save last checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss
        }, 'weights/last.pt')
        
        # Update learning rate
        scheduler.step()

if __name__ == '__main__':
    args = parse_args()
    
    # Create output directories
    os.makedirs('weights', exist_ok=True)
    
    try:
        # Start training
        train(args)
    except KeyboardInterrupt:
        print('\nTraining interrupted by user')
    except Exception as e:
        print(f'\nError during training: {str(e)}') 