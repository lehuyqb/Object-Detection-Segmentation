import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.conv3 = ConvBlock(out_channels, out_channels)
        self.skip_conv = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        skip = self.skip_conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.cat([x, skip], dim=1)

class YOLOv5(nn.Module):
    def __init__(self, num_classes, img_size=640):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_checkpointing = True  # Enable gradient checkpointing by default
        
        # Backbone
        self.backbone = nn.ModuleList([
            ConvBlock(3, 32, 6, 2),      # P1/2
            ConvBlock(32, 64, 3, 2),     # P2/4
            CSPBlock(64, 64),            # 128 channels out
            ConvBlock(128, 128, 3, 2),   # P3/8
            CSPBlock(128, 128),          # 256 channels out
            ConvBlock(256, 256, 3, 2),   # P4/16
            CSPBlock(256, 256),          # 512 channels out
        ])

        # Neck (FPN)
        self.fpn_channels = [512, 256, 128]  # Input channels for each FPN level
        self.fpn = nn.ModuleList([
            nn.Sequential(
                ConvBlock(512, 256, 1),  # 512 -> 256
                ConvBlock(256, 256)      # 256 -> 256
            ),
            nn.Sequential(
                ConvBlock(256, 128, 1),  # 256 -> 128
                ConvBlock(128, 128)      # 128 -> 128
            ),
            nn.Sequential(
                ConvBlock(128, 64, 1),   # 128 -> 64
                ConvBlock(64, 64)        # 64 -> 64
            ),
        ])

        # Detection heads
        self.head_channels = [256, 128, 64]  # Input channels for each head
        self.heads = nn.ModuleList([
            nn.Conv2d(256, (num_classes + 5) * 3, 1),  # Large objects
            nn.Conv2d(128, (num_classes + 5) * 3, 1),  # Medium objects
            nn.Conv2d(64, (num_classes + 5) * 3, 1),   # Small objects
        ])

    def forward(self, x):
        # Backbone forward pass with gradient checkpointing
        features = []
        for i, layer in enumerate(self.backbone):
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
            if i in [2, 4, 6]:  # Save CSP outputs
                features.append(x)

        # FPN forward pass with gradient checkpointing
        outputs = []
        features = features[::-1]  # Reverse order for FPN (large to small)
        
        for i, (feature, fpn_layer) in enumerate(zip(features, self.fpn)):
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(fpn_layer, feature)
            else:
                x = fpn_layer(feature)
            outputs.append(self.heads[i](x))  # Apply corresponding detection head

        return outputs

    def compute_loss(self, predictions, targets):
        device = predictions[0].device
        total_loss = torch.tensor(0.0, device=device)
        
        # Loss coefficients
        lambda_box = 5.0  # box loss gain
        lambda_obj = 1.0  # objectness loss gain
        lambda_cls = 1.0  # class loss gain
        
        for i, pred in enumerate(predictions):
            # Reshape predictions [batch, anchors * (num_classes + 5), grid, grid] ->
            # [batch, grid, grid, anchors, (num_classes + 5)]
            batch_size = pred.shape[0]
            grid_size = pred.shape[-1]
            stride = self.img_size / grid_size  # detector stride
            
            pred = pred.view(batch_size, 3, self.num_classes + 5, grid_size, grid_size)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # Extract predictions
            pred_box = pred[..., :4]  # box predictions
            pred_obj = pred[..., 4]   # objectness predictions
            pred_cls = pred[..., 5:]  # class predictions
            
            # Match targets to anchors
            targets_for_layer = self._build_targets(targets, i, grid_size, stride)
            
            if targets_for_layer is not None:
                obj_mask = targets_for_layer['obj_mask']
                noobj_mask = ~obj_mask
                target_box = targets_for_layer['box']
                target_cls = targets_for_layer['cls']
                
                # Box loss (using GIoU loss)
                if obj_mask.any():
                    pred_box_selected = pred_box[obj_mask]
                    target_box_selected = target_box[obj_mask]
                    box_loss = self._box_loss(pred_box_selected, target_box_selected)
                    total_loss += lambda_box * box_loss
                
                # Objectness loss (BCE)
                obj_loss = nn.BCEWithLogitsLoss()(pred_obj, obj_mask.float())
                total_loss += lambda_obj * obj_loss
                
                # Classification loss
                if obj_mask.any():
                    pred_cls_selected = pred_cls[obj_mask]
                    target_cls_selected = target_cls[obj_mask]
                    cls_loss = nn.CrossEntropyLoss()(pred_cls_selected, target_cls_selected)
                    total_loss += lambda_cls * cls_loss
        
        return total_loss

    def _build_targets(self, targets, layer_idx, grid_size, stride):
        """Convert raw targets into format needed for training"""
        device = targets['boxes'][0].device
        batch_size = len(targets['boxes'])
        
        # Initialize masks and targets
        obj_mask = torch.zeros((batch_size, 3, grid_size, grid_size), dtype=torch.bool, device=device)
        target_box = torch.zeros((batch_size, 3, grid_size, grid_size, 4), dtype=torch.float32, device=device)
        target_cls = torch.zeros((batch_size, 3, grid_size, grid_size), dtype=torch.long, device=device)
        
        # Process each image in batch
        for batch_idx in range(batch_size):
            boxes = targets['boxes'][batch_idx]
            labels = targets['labels'][batch_idx]
            
            if len(boxes) == 0:
                continue
                
            # Convert YOLO format (x_center, y_center, w, h) to grid cells
            grid_x = (boxes[:, 0] * grid_size).long()
            grid_y = (boxes[:, 1] * grid_size).long()
            
            # Constrain to grid
            grid_x = torch.clamp(grid_x, 0, grid_size - 1)
            grid_y = torch.clamp(grid_y, 0, grid_size - 1)
            
            # Assign targets to best matching anchor
            for box_idx in range(len(boxes)):
                x, y = grid_x[box_idx], grid_y[box_idx]
                
                # Assign to all anchors at this grid cell
                obj_mask[batch_idx, :, y, x] = True
                target_box[batch_idx, :, y, x] = boxes[box_idx]
                target_cls[batch_idx, :, y, x] = labels[box_idx]
        
        return {
            'obj_mask': obj_mask,
            'box': target_box,
            'cls': target_cls
        }

    def _box_loss(self, pred_box, target_box):
        """Compute box loss using GIoU"""
        # Convert predictions to x1,y1,x2,y2 format
        pred_x1y1 = pred_box[..., :2] - pred_box[..., 2:] / 2
        pred_x2y2 = pred_box[..., :2] + pred_box[..., 2:] / 2
        
        # Convert targets to x1,y1,x2,y2 format
        target_x1y1 = target_box[..., :2] - target_box[..., 2:] / 2
        target_x2y2 = target_box[..., :2] + target_box[..., 2:] / 2
        
        # Calculate IoU
        intersect_x1y1 = torch.max(pred_x1y1, target_x1y1)
        intersect_x2y2 = torch.min(pred_x2y2, target_x2y2)
        intersect_wh = torch.clamp(intersect_x2y2 - intersect_x1y1, min=0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        pred_area = pred_box[..., 2] * pred_box[..., 3]
        target_area = target_box[..., 2] * target_box[..., 3]
        union_area = pred_area + target_area - intersect_area
        
        iou = intersect_area / (union_area + 1e-16)
        
        # Calculate GIoU
        enclosing_x1y1 = torch.min(pred_x1y1, target_x1y1)
        enclosing_x2y2 = torch.max(pred_x2y2, target_x2y2)
        enclosing_wh = enclosing_x2y2 - enclosing_x1y1
        enclosing_area = enclosing_wh[..., 0] * enclosing_wh[..., 1]
        
        giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-16)
        
        return (1 - giou).mean()

def create_model(num_classes, pretrained=False):
    model = YOLOv5(num_classes)
    if pretrained:
        # Load pretrained weights if available
        pass
    return model 