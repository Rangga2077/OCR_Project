import os
import re
from pathlib import Path

import cv2
import pandas as pd
from google.cloud import vision


# Change these values when you want to test another folder or file pattern.
BASE_PATH = Path(r"C:\Users\Rangga Saputra\Documents\Test_data")
IMAGE_NAME_CONTAINS = ""  # Empty string means "use all images".
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
SAMPLE_LIMIT = None  # Use None to process every folder.
OUTPUT_PREFIX = "inspection_test_data"
EXPECTED_NUMBER = "06039"  # Used when testing images directly in BASE_PATH.
USE_PREPROCESSING = True
MAX_PREPROCESSING_VARIANTS = 8  # Keep this low for large Google Vision batches.

def ocr_normalize(text):
    text = text.strip().replace(" ", "").replace("-", "")
    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        "G": "6",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

def find_five_digit_numbers(text):
    cleaned = ocr_normalize(text)
    return re.findall(r"\d{5}", cleaned)


def find_five_digit_number(text):
    numbers = find_five_digit_numbers(text)
    return numbers[0] if numbers else None


def detect_text_from_bytes(client, image_bytes):
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)

    if not response.text_annotations:
        return ""

    return response.text_annotations[0].description


def detect_text(client, image_path):
    return detect_text_from_bytes(client, image_path.read_bytes())


def encode_image(image):
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Failed to encode preprocessed image")

    return buffer.tobytes()


def resize_for_ocr(image, target_width=1400):
    height, width = image.shape[:2]
    if width >= target_width:
        return image

    scale = target_width / width
    new_size = (target_width, int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)


def ensure_bgr(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return image


def reduce_glare(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Very bright, low-saturation pixels are usually reflected light/glare.
    glare_mask = cv2.inRange(hsv, (0, 0, 220), (180, 65, 255))
    glare_area = cv2.countNonZero(glare_mask)
    image_area = image.shape[0] * image.shape[1]
    if glare_area == 0 or glare_area > image_area * 0.25:
        return image

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)
    return cv2.inpaint(image, glare_mask, 3, cv2.INPAINT_TELEA)


def suppress_green_obstruction(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green leaves/plants often create high-contrast shapes near the plate.
    green_mask = cv2.inRange(hsv, (35, 45, 35), (95, 255, 255))
    if cv2.countNonZero(green_mask) == 0:
        return image

    gray_bgr = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    reduced = image.copy()
    reduced[green_mask > 0] = gray_bgr[green_mask > 0]
    return reduced


def enhance_plate_contrast(image):
    gray = cv2.cvtColor(ensure_bgr(image), cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(enhanced, 1.6, blurred, -0.6, 0)
    return sharpened


def threshold_for_digits(image):
    gray = enhance_plate_contrast(image)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )


def rotate_image(image, angle):
    if angle == 0:
        return image

    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if angle == -90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def find_white_strip_crop(image):
    resized = resize_for_ocr(image)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # Low saturation + high brightness usually captures the white number strip.
    mask = cv2.inRange(hsv, (0, 0, 110), (180, 95, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = resized.shape[0] * resized.shape[1]
    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / max(h, 1)

        if area < image_area * 0.015:
            continue
        if aspect_ratio < 1.2:
            continue

        candidates.append((area * aspect_ratio, x, y, w, h))

    if not candidates:
        return None

    _, x, y, w, h = max(candidates)
    padding = 40
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, resized.shape[1])
    y2 = min(y + h + padding, resized.shape[0])
    return resized[y1:y2, x1:x2]


def append_variant(variants, seen_names, name, image):
    if image is None or name in seen_names:
        return

    variants.append((name, image))
    seen_names.add(name)


def opencv_preprocess_images(image_path):
    """
    Create plate-focused OCR image variants before sending to Google Vision.
    Each returned item is (variant_name, image_array).
    """
    original = cv2.imread(str(image_path))
    if original is None:
        return []

    variants = []
    seen_names = set()
    original = resize_for_ocr(original)

    append_variant(variants, seen_names, "opencv-original-resized", original)

    glare_reduced = reduce_glare(original)
    append_variant(variants, seen_names, "opencv-glare-reduced", glare_reduced)

    green_reduced = suppress_green_obstruction(glare_reduced)
    append_variant(variants, seen_names, "opencv-green-suppressed", green_reduced)

    enhanced = enhance_plate_contrast(green_reduced)
    append_variant(variants, seen_names, "opencv-enhanced-gray", enhanced)

    thresholded = threshold_for_digits(green_reduced)
    append_variant(variants, seen_names, "opencv-threshold", thresholded)

    strip_crop = find_white_strip_crop(green_reduced)
    if strip_crop is not None:
        append_variant(variants, seen_names, "opencv-plate-crop", strip_crop)
        append_variant(variants, seen_names, "opencv-plate-crop-threshold", threshold_for_digits(strip_crop))

    for angle in (-8, -4, 4, 8):
        if len(variants) >= MAX_PREPROCESSING_VARIANTS:
            break

        rotated = rotate_image(green_reduced, angle)
        append_variant(variants, seen_names, f"opencv-deskew-{angle}", rotated)

    return variants[:MAX_PREPROCESSING_VARIANTS]


def find_images(folder):
    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file()
            and (not IMAGE_NAME_CONTAINS or IMAGE_NAME_CONTAINS in file.name)
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

def scan_folder(client, folder, expected):
    try:
        images = find_images(folder)
    except PermissionError:
        return "Access Denied", 0, ""

    best_detected = "Not Detected"
    best_difference = None
    best_source_image = ""

    for image_path in images:
        image_results = scan_image(client, image_path, expected)
        if not image_results:
            print(f"  {image_path.name}: no 5-digit number found")
            continue

        for lamp_number, variant_name in image_results:
            print(f"  {image_path.name} [{variant_name}]: detected {lamp_number}")

            difference = count_digit_differences(expected, lamp_number)
            if best_difference is None or difference < best_difference:
                best_detected = lamp_number
                best_difference = difference
                best_source_image = f"{image_path.name} [{variant_name}]"

            if lamp_number == expected:
                return lamp_number, len(images), best_source_image

    return best_detected, len(images), best_source_image


def scan_image(client, image_path, expected):
    results = []
    seen = set()

    if USE_PREPROCESSING:
        variants = opencv_preprocess_images(image_path)
    else:
        variants = [("original", image_path.read_bytes())]

    for variant_name, variant in variants:
        try:
            if isinstance(variant, bytes):
                detected_text = detect_text_from_bytes(client, variant)
            else:
                detected_text = detect_text_from_bytes(client, encode_image(variant))
        except Exception as exc:
            print(f"  Warning: OCR failed for {image_path.name} [{variant_name}]: {exc}")
            continue

        for lamp_number in find_five_digit_numbers(detected_text):
            key = (lamp_number, variant_name)
            if key in seen:
                continue

            seen.add(key)
            results.append((lamp_number, variant_name))

            if lamp_number == expected:
                return results

    return results

def next_output_file():
    counter = 1
    while Path(f"{OUTPUT_PREFIX}_{counter}.xlsx").exists():
        counter += 1

    return Path(f"{OUTPUT_PREFIX}_{counter}.xlsx")


def get_result(expected, detected):
    if detected == expected:
        return "Match"
    if detected in ("Not Detected", "Access Denied"):
        return detected
    return "Mismatch"


def count_digit_differences(expected, detected):
    if len(expected) != len(detected):
        return max(len(expected), len(detected))

    return sum(left != right for left, right in zip(expected, detected))


def find_scan_targets():
    folders = [folder for folder in BASE_PATH.iterdir() if folder.is_dir()]
    if folders:
        return folders[:SAMPLE_LIMIT] if SAMPLE_LIMIT else folders

    if find_images(BASE_PATH):
        return [BASE_PATH]

    return []


def main():
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise SystemExit("Please set GOOGLE_APPLICATION_CREDENTIALS first.")

    if not BASE_PATH.exists():
        raise SystemExit(f"Base path not found: {BASE_PATH}")

    print("Loading Google Vision client...")
    client = vision.ImageAnnotatorClient()
    print(f"OpenCV preprocessing: {'on' if USE_PREPROCESSING else 'off'}")

    folders = find_scan_targets()
    if not folders:
        raise SystemExit(f"No images or folders found in: {BASE_PATH}")

    print(f"Processing {len(folders)} folders...\n")

    rows = []
    for index, folder in enumerate(folders, start=1):
        expected = EXPECTED_NUMBER if folder == BASE_PATH else folder.name
        detected, image_count, source_image = scan_folder(client, folder, expected)
        result = get_result(expected, detected)

        rows.append(
            {
                "Folder ID": folder.name,
                "Expected Lamp Number": expected,
                "Detected": detected,
                "Result": result,
                "Images Checked": image_count,
                "Source Image": source_image,
            }
        )

        print(f"[{index}/{len(folders)}] {folder.name} -> {detected} ({result})")

    df = pd.DataFrame(rows)
    output_file = next_output_file()
    df.to_excel(output_file, index=False)

    print(f"\nDone! Results saved to {output_file}")
    print(f"Matched: {len(df[df['Result'] == 'Match'])}/{len(df)}")
    print(f"Mismatch: {len(df[df['Result'] == 'Mismatch'])}/{len(df)}")
    print(f"Not found: {len(df[df['Result'] == 'Not Detected'])}/{len(df)}")


if __name__ == "__main__":
    main()
