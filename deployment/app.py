from ultralytics import YOLO  
import torch
import cv2
import numpy as np
import gradio as gr
from PIL import Image

# Load YOLOv8 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO("yolov8x.pt")  # Load a more powerful YOLOv8 model
model.to(device)
model.eval()

# Load COCO class labels
CLASS_NAMES = model.names  # YOLO's built-in class names

def preprocess_image(image):
    image = Image.fromarray(image)
    image = image.convert("RGB")
    return image

def detect_objects(image):
    image = preprocess_image(image)
    results = model.predict(image)  # Run YOLO inference

    # Convert results to bounding box format
    image = np.array(image)
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            class_name = CLASS_NAMES[int(cls)]  # Get class name
            confidence = conf.item() * 100  # Convert confidence to percentage

            # Draw a bolder bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Increased thickness

            # Larger text for class label
            label = f"{class_name} ({confidence:.1f}%)"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 3, cv2.LINE_AA)  # Larger text

    return image

# Gradio UI with Submit button
iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Image(type="numpy", label="Detected Objects"),
    title="Object Detection",
    description="Use webcam or Upload an image to detect objects.",
    allow_flagging="never"  # Disables unwanted flags
)

iface.launch()