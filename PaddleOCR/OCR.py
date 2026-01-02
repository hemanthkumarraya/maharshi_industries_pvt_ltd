import cv2
import json
import numpy as np
from pathlib import Path
from paddleocr import PaddleOCR

# ==================== PATHS ====================
ROOT_DIR = Path(__file__).parent.resolve()
INPUT_DIR = ROOT_DIR / "Input"
OUTPUT_DIR = ROOT_DIR / "Output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png"}

# ==================== LOAD PADDLE OCR ====================
ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True,
    text_det_thresh=0.3,
    text_det_box_thresh=0.6,
    text_det_unclip_ratio=1.5
)

# ==================== PREPROCESSING ====================
def preprocess_image(img):
    """
    Grayscale + CLAHE + color inversion
    Converted back to 3-channel for PaddleOCR
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Invert colors
    inverted = cv2.bitwise_not(enhanced)

    # PaddleOCR requires 3 channels
    inverted_bgr = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)
    return inverted_bgr

# ==================== OCR PROCESS ====================
for img_path in INPUT_DIR.iterdir():
    if img_path.suffix.lower() not in SUPPORTED_EXT:
        continue

    print(f"Processing: {img_path.name}")

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"⚠️ Could not read image: {img_path.name}")
        continue

    # ---- Preprocess image ----
    preprocessed = preprocess_image(image)

    # ---- SAVE PREPROCESSED IMAGE ----
    cv2.imwrite(
        str(OUTPUT_DIR / f"{img_path.stem}_preprocessed.jpg"),
        preprocessed
    )

    # ---- Run PaddleOCR ----
    result = ocr.predict(preprocessed)

    results = []
    output_img = image.copy()

    # Normalize output structure (version-safe)
    words = []
    if isinstance(result, list):
        for item in result:
            if isinstance(item, list):
                words.extend(item)
            elif isinstance(item, dict):
                words.append(item)

    for word in words:
        bbox = None
        text = None
        confidence = None

        # List-based format
        if isinstance(word, (list, tuple)) and len(word) == 2:
            if isinstance(word[0], (list, tuple)):
                bbox = np.array(word[0], dtype=int)
            if isinstance(word[1], (list, tuple)) and len(word[1]) == 2:
                text = word[1][0]
                confidence = float(word[1][1])

        # Dict-based format
        elif isinstance(word, dict):
            if isinstance(word.get("points"), list):
                bbox = np.array(word["points"], dtype=int)
            text = word.get("transcription")
            confidence = word.get("confidence", 0.0)

        # Validate detection
        if bbox is None or text is None:
            continue

        confidence = float(confidence)
        if confidence < 0.4:
            continue

        results.append({
            "text": text,
            "confidence": round(confidence, 3),
            "bbox": bbox.tolist(),
            "start": bbox[0],
            "end": bbox[2]
        })

        # Draw bounding box
        cv2.polylines(output_img, [bbox], True, (0, 255, 0), 2)

        # Draw text
        cv2.putText(
            output_img,
            f"{text} ({confidence:.2f})",
            (bbox[0][0], max(0, bbox[0][1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ==================== SAVE OUTPUT ====================
    json_output = {
        "input_file": img_path.name,
        "total_detections": len(results),
        "results": results
    }

    out_img = OUTPUT_DIR / f"{img_path.stem}_ocr.jpg"
    out_json = OUTPUT_DIR / f"{img_path.stem}_ocr.json"

    cv2.imwrite(str(out_img), output_img)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)

    print(f"✅ Saved: {out_img.name}, {out_json.name}")

print("\n PaddleOCR processing completed successfully.")
