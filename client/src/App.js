import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const canvasRef = useRef(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImage(e.target.result);
        setPredictions(null);
        setError(null);
        // Draw image on canvas
        const img = new Image();
        img.onload = () => {
          const canvas = canvasRef.current;
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0);
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  const drawBox = (ctx, box, label, score) => {
    const [x, y, w, h] = box;
    const canvas = canvasRef.current;
    const actualX = x * canvas.width;
    const actualY = y * canvas.height;
    const actualW = w * canvas.width;
    const actualH = h * canvas.height;

    // Draw box
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.strokeRect(actualX, actualY, actualW, actualH);

    // Draw label
    ctx.fillStyle = '#00ff00';
    ctx.font = '16px Arial';
    ctx.fillText(`${label} ${(score * 100).toFixed(1)}%`, actualX, actualY - 5);
  };

  const predict = async () => {
    if (!image) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image }),
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPredictions(data.predictions);

      // Draw predictions
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0);
        data.predictions[0].boxes.forEach((box, i) => {
          drawBox(
            ctx,
            box,
            data.predictions[0].classes[i],
            data.predictions[0].scores[i]
          );
        });
      };
      img.src = image;
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Object Detection Demo</h1>
      
      <div className="upload-section">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="file-input"
        />
        <button
          onClick={predict}
          disabled={!image || loading}
          className="predict-button"
        >
          {loading ? 'Processing...' : 'Detect Objects'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      <div className="result-section">
        <canvas ref={canvasRef} className="result-canvas" />
      </div>

      {predictions && (
        <div className="predictions">
          <h2>Detections:</h2>
          <ul>
            {predictions[0].classes.map((className, i) => (
              <li key={i}>
                {className}: {(predictions[0].scores[i] * 100).toFixed(1)}%
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App; 