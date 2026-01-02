import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

# ==================== LOCAL PATHS (Hardcoded - Using Path for proper methods) ====================
ROOT_DIR = Path(r"C:\Users\HEMANTH KUMAR\OneDrive\Desktop\maharshi_industries_pvt_ltd")

INPUT_DIR = ROOT_DIR / "Input"
OUTPUT_DIR = ROOT_DIR / "Output"

DETECTOR_PATH = ROOT_DIR / "models" / "faster_rcnn_object_detector_FAST.pth"
CLASSIFIER_PATH = ROOT_DIR / "models" / "animal_classifier.pth"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mov"}


print("🚀 Starting Part A: Object Detection + Human/Animal Classification")
print(f"   Input:  {INPUT_DIR}")
print(f"   Output: {OUTPUT_DIR}\n")

# ==================== GPU DETECTION ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print("   → Running with full GPU acceleration\n")
else:
    print("⚠️  No GPU found → Running on CPU\n")

# ==================== MODEL LOADING ====================
print("Loading trained models...")

try:
    # Object Detector
    detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, num_classes=2)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    detector.load_state_dict(torch.load(DETECTOR_PATH, map_location=device))
    detector.to(device)
    detector.eval()

    # Human/Animal Classifier
    classifier = resnet18(weights=None)
    classifier.fc = torch.nn.Linear(classifier.fc.in_features, 2)
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier.to(device)
    classifier.eval()

    print("Models loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("   Check if model files exist at:")
    print(f"      {DETECTOR_PATH}")
    print(f"      {CLASSIFIER_PATH}")
    exit(1)

# Transform for classification
classify_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== PROCESSING FUNCTION ====================
def process_file(input_path: Path):
    stem = input_path.stem
    ext = input_path.suffix.lower()
    output_path = OUTPUT_DIR / f"{stem}_annotated{ext}"

    print(f"   → Processing: {input_path.name}")

    if ext in {".jpg", ".jpeg", ".png"}:
        img_bgr = cv2.imread(str(input_path))
        if img_bgr is None:
            print(f"      ⚠️ Could not read image")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = detector(img_tensor)[0]

        boxes = prediction['boxes'][prediction['scores'] > 0.5].cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = img_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_tensor = classify_transform(Image.fromarray(crop)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_idx = classifier(crop_tensor).argmax(1).item()

            label = "Animal" if pred_idx == 0 else "Human"
            color = (0, 0, 255) if pred_idx == 0 else (0, 255, 0)

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        success = cv2.imwrite(str(output_path), img_bgr)
        if success:
            print(f"      ✅ Saved: {output_path.name}")
        else:
            print(f"      ❌ Failed to save output")

    else:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"      ⚠️ Could not open video")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transforms.ToTensor()(frame_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = detector(frame_tensor)[0]

            boxes = prediction['boxes'][prediction['scores'] > 0.5].cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_tensor = classify_transform(Image.fromarray(crop)).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_idx = classifier(crop_tensor).argmax(1).item()

                label = "Animal" if pred_idx == 0 else "Human"
                color = (0, 0, 255) if pred_idx == 0 else (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            writer.write(frame)
            frame_count += 1

        cap.release()
        writer.release()
        print(f"      ✅ Saved: {output_path.name} ({frame_count} frames)")

# ==================== MAIN EXECUTION ====================
def main():
    print("=== Part A Processing Started ===\n")

    if not INPUT_DIR.exists():
        print(f"❌ Input folder not found:\n   {INPUT_DIR}")
        print("   Please create 'Input' folder and add test files.")
        return

    files = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        print("⚠️  No supported files found in 'Input' folder.")
        print("   Supported formats: .jpg, .jpeg, .png, .mp4, .avi, .mov")
        return

    print(f"Found {len(files)} file(s) to process:\n")
    for file_path in files:
        process_file(file_path)

    print("\n🎉 Part A processing completed!")
    print(f"   All annotated results saved in:\n   {OUTPUT_DIR}")

if __name__ == "__main__":
    main()