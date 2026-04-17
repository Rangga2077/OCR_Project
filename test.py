import os
import cv2
import pandas as pd
import re
from paddleocr import PaddleOCR

# Initialize PaddleOCR (use_angle_cls helps with rotated text)
print("Loading PaddleOCR model...")
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
print("Model loaded. Starting processing...\n")


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Scale up small images for better OCR accuracy
    h, w = img.shape[:2]
    if w < 800:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive thresholding to sharpen text
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # PaddleOCR needs a 3-channel image
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def fix_ocr_text(text):
    """Fix common OCR misreads for digit-only plate numbers."""
    cleaned = text.strip().replace(" ", "").replace("-", "")
    cleaned = cleaned.replace("O", "0").replace("o", "0")
    cleaned = cleaned.replace("I", "1").replace("l", "1")
    cleaned = cleaned.replace("S", "5").replace("s", "5")
    cleaned = cleaned.replace("B", "8").replace("G", "6")
    return cleaned


def extract_plate_number(folder_path):
    """
    Scan ALL 施工後*.jpg images in the folder.
    Returns the best (highest confidence) 6-digit match.
    """
    try:
        all_files = os.listdir(folder_path)
    except PermissionError:
        return "Access Denied", 0.0, 0

    # Fix: use 'in' for substring check, endswith() for extension — NOT glob patterns
    files = [
        f for f in all_files
        if "施工後" in f and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not files:
        return "Not Detected", 0.0, 0

    best_text = "Not Detected"
    best_conf = 0.0

    for filename in files:
        image_path = os.path.join(folder_path, filename)
        processed = preprocess_image(image_path)

        if processed is None:
            continue

        # PaddleOCR on the preprocessed image (numpy array)
        results = ocr.ocr(processed, cls=True)

        if not results or results[0] is None:
            continue

        for line in results[0]:
            # PaddleOCR result format: [bbox, (text, confidence)]
            text, prob = line[1][0], line[1][1]

            cleaned = fix_ocr_text(text)

            # Match exactly 6 consecutive digits (plate format)
            match = re.search(r'\d{6}', cleaned)
            if match and prob > best_conf:
                best_text = match.group()
                best_conf = prob

    return best_text, round(best_conf, 4), len(files)


# --- Main Execution ---
data = []
base_path = r'Z:\HKLight'

all_folders = [
    f for f in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, f))
]

# Phase 1: Test first 50 folders (remove [:50] for full run)
sample_folders = all_folders[:50]
total = len(sample_folders)
print(f"Processing {total} folders...\n")

for i, folder_id in enumerate(sample_folders, 1):
    path = os.path.join(base_path, folder_id)
    plate, confidence, num_images = extract_plate_number(path)

    # Plate number == folder name, flag match/mismatch
    if plate == folder_id:
        result = "Match"
    elif plate in ("Not Detected", "Access Denied"):
        result = plate
    else:
        result = "Mismatch"

    data.append({
        "Folder ID": folder_id,
        "Expected Plate": folder_id,
        "Detected": plate,
        "Result": result,
        "Confidence": confidence,
        "Images Checked": num_images,
    })

    print(f"[{i}/{total}] {folder_id} -> {plate} ({result}) conf={confidence}")

# Save results — auto-increment filename (ins
df = pd.DataFrame(data)
counter = 1
while os.path.exists(f"inspection_{counter}.xlsx"):
    counter += 1
output_file = f"inspection_{counter}.xlsx"
df.to_excel(output_file, index=False)

# Summary
total_match = df[df["Result"] == "Match"].shape[0]
total_mismatch = df[df["Result"] == "Mismatch"].shape[0]
total_not_found = df[df["Result"] == "Not Detected"].shape[0]

print(f"\nDone! Results saved to {output_file}")
print(f"  Matched:    {total_match}/{total}")
print(f"  Mismatch:   {total_mismatch}/{total}")
print(f"  Not found:  {total_not_found}/{total}")