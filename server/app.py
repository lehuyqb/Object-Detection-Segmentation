from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
from models.yolo import create_model
import os
app = Flask(__name__)
CORS(app)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
class_names = []

def load_model(weights_path, num_classes):
    global model, class_names
    model = create_model(num_classes=num_classes)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

def process_image(image_data, img_size=416):
    # Decode base64 image
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    image = np.array(image)
    image = cv2.resize(image, (img_size, img_size))
    image = image / 255.0
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    return image

def post_process_predictions(predictions, confidence_threshold=0.5):
    results = []
    for pred in predictions:
        # Reshape predictions
        batch_size, _, height, width = pred.shape
        pred = pred.view(batch_size, 3, -1, height, width)
        pred = pred.permute(0, 1, 3, 4, 2).contiguous()
        
        # Extract box coordinates, objectness, and class scores
        boxes = pred[..., :4]
        obj_scores = torch.sigmoid(pred[..., 4])
        class_scores = torch.sigmoid(pred[..., 5:])
        
        # Get class with highest score
        class_scores, class_ids = class_scores.max(-1)
        
        # Combine objectness and class scores
        scores = obj_scores * class_scores
        
        # Filter by confidence threshold
        mask = scores > confidence_threshold
        
        for idx in range(batch_size):
            batch_boxes = boxes[idx][mask[idx]]
            batch_scores = scores[idx][mask[idx]]
            batch_classes = class_ids[idx][mask[idx]]
            
            # Convert to numpy for JSON serialization
            results.append({
                'boxes': batch_boxes.cpu().numpy().tolist(),
                'scores': batch_scores.cpu().numpy().tolist(),
                'classes': [class_names[i] for i in batch_classes.cpu().numpy().tolist()]
            })
    
    return results

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Process image
        image = process_image(request.json['image'])
        image = image.to(device)
        
        # Get predictions
        with torch.no_grad():
            predictions = model(image)
        
        # Post-process predictions
        results = post_process_predictions(predictions)
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Load model and class names from config
    import yaml
    with open(os.path.join(os.path.dirname(__file__), 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = config['classes']['names']
    load_model(config['model']['weights_path'], len(class_names))
    
    app.run(host='0.0.0.0', port=5000) 