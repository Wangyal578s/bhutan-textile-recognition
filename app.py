from flask import Flask, render_template, Response, request, jsonify
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import json
import numpy as np
import io
import threading
import time

# --- CONFIG ---
app = Flask(__name__)
MODEL_PATH = "bkt.pth"
DESC_PATH = "descriptions.json"
CLASS_NAMES = ['Hotra', 'Mathra', 'Mentha', 'Pangtsi', 'Satra', 'other_Kira_samples']

# --- LOAD DESCRIPTIONS ---
with open(DESC_PATH, "r") as f:
    descriptions = json.load(f)

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥 Using device: {device}")

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- LOAD MODEL ---
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.6),
    torch.nn.Linear(512, len(CLASS_NAMES))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# --- CAMERA CONTROL ---
camera = None
latest_frame = None
running = False


def classify_patch(patch):
    """Classify a single image patch."""
    img = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
        return CLASS_NAMES[preds.item()], conf.item()


def capture_frames():
    """Continuously capture frames in background."""
    global camera, latest_frame, running
    while running:
        success, frame = camera.read()
        if not success:
            continue
        latest_frame = frame
        time.sleep(0.05)  # ~20 FPS


def generate_frames():
    """Yield frames for streaming with balanced detection."""
    global camera, latest_frame, running
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    running = True
    threading.Thread(target=capture_frames, daemon=True).start()

    smooth_label = None
    smooth_conf = 0

    while running:
        if latest_frame is None:
            continue
        frame = latest_frame.copy()

        h, w, _ = frame.shape
        grid_size = 3
        box_h, box_w = h // grid_size, w // grid_size
        detections = []

        # Scan each grid cell
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * box_h, (i + 1) * box_h
                x1, x2 = j * box_w, (j + 1) * box_w
                patch = frame[y1:y2, x1:x2]
                label, confidence = classify_patch(patch)
                if confidence >= 0.65:
                    detections.append((x1, y1, x2, y2, label, confidence))

        if detections:
            # Aggregate detections — choose the most frequent label
            labels = [d[4] for d in detections]
            most_common = max(set(labels), key=labels.count)
            avg_conf = np.mean([d[5] for d in detections if d[4] == most_common])

            # Optional smoothing: avoid flicker
            if smooth_label is None or most_common == smooth_label:
                smooth_label = most_common
                smooth_conf = avg_conf
            else:
                smooth_conf = (smooth_conf * 0.7) + (avg_conf * 0.3)
                if smooth_conf < 0.5:
                    smooth_label = most_common

            # Draw all bounding boxes
            for (x1, y1, x2, y2, label, confidence) in detections:
                color = (0, 255, 0) if label == smooth_label else (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f"{label}: {confidence*100:.1f}%"
                cv2.putText(frame, text, (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Display summary text
            cv2.putText(frame, f"Detected: {smooth_label} ({smooth_conf*100:.1f}%)",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Textile Found", (w//2 - 150, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (w//3, h//3), (2*w//3, 2*h//3), (0, 0, 255), 2)
            smooth_label, smooth_conf = None, 0

        # Encode for web stream
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()


# --- ROUTES ---

@app.route('/')
@app.route('/home')
def home():
    """Home page"""
    return render_template('home.html')

@app.route('/detection')
def detection():
    """Main textile detection page"""
    return render_template('index.html')

@app.route('/significance')
def significance():
    """Significance of individual textiles"""
    return render_template('significance.html')

import os
from flask import url_for, render_template

@app.route('/gallery')
def gallery():
    gallery_folder = os.path.join(app.static_folder, 'gallery')
    textiles = []

    if os.path.exists(gallery_folder):
        for filename in sorted(os.listdir(gallery_folder), key=lambda s: s.lower()):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                name = os.path.splitext(filename)[0]
                display_name = name.replace('_', ' ').replace('-', ' ').title()
                textiles.append({
                    "name": display_name,
                    "image": f"gallery/{filename}"
                })

    return render_template("gallery.html", textiles=textiles)

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')


# --- Detection + Video APIs ---

@app.route('/video_feed')
def video_feed():
    """Start camera and stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_camera')
def stop_camera():
    """Stop camera safely."""
    global running
    running = False
    return jsonify({"status": "Camera stopped"})


@app.route('/predict', methods=['POST'])
def predict():
    """Predict textile type from uploaded image."""
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"})

    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "Empty filename"})

    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            probs = F.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            confidence = round(conf.item(), 2)
            label = CLASS_NAMES[preds.item()]

        # ✅ Lower threshold to 0.5
        if confidence < 0.5:
            return jsonify({
                "success": False,
                "label": "No Textile Detected",
                "confidence": confidence
            })

        # ✅ Load description if available
        info = descriptions.get(label, {})
        significance = info.get("significance", "No significance info available.")
        description = info.get("description", "No description available.")

        return jsonify({
            "success": True,
            "label": label,
            "confidence": confidence,
            "significance": significance,
            "description": description
        })

    except Exception as e:
        print("❌ Prediction error:", e)
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, threaded=True)
