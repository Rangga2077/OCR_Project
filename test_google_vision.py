import os
import re
from pathlib import Path

import pandas as pd
from google.cloud import vision


# Change these values when you want to test another folder or file pattern.
BASE_PATH = Path(r"Z:\HKLight")
IMAGE_NAME_CONTAINS = "施工後*"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
SAMPLE_LIMIT = 50  # Use None to process every folder.
OUTPUT_PREFIX = "inspection_google_vision"


def clean_ocr_text(text):
    """Normalize OCR text so common letter mistakes can become digits."""
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


def find_six_digit_number(text):
    cleaned = clean_ocr_text(text)
    match = re.search(r"\d{6}", cleaned)
    return match.group() if match else None


def detect_text(client, image_path):
    with open(image_path, "rb") as image_file:
        image = vision.Image(content=image_file.read())

    response = client.text_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)

    if not response.text_annotations:
        return ""

    return response.text_annotations[0].description


def find_images(folder):
    return [
        file
        for file in folder.iterdir()
        if file.is_file()
        and IMAGE_NAME_CONTAINS in file.name
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def scan_folder(client, folder):
    try:
        images = find_images(folder)
    except PermissionError:
        return "Access Denied", 0, ""

    for image_path in images:
        try:
            detected_text = detect_text(client, image_path)
        except Exception as exc:
            print(f"  Warning: OCR failed for {image_path.name}: {exc}")
            continue

        lamp_number = find_six_digit_number(detected_text)
        if lamp_number:
            return lamp_number, len(images), image_path.name

    return "Not Detected", len(images), ""


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


def main():
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise SystemExit("Please set GOOGLE_APPLICATION_CREDENTIALS first.")

    if not BASE_PATH.exists():
        raise SystemExit(f"Base path not found: {BASE_PATH}")

    print("Loading Google Vision client...")
    client = vision.ImageAnnotatorClient()

    folders = [folder for folder in BASE_PATH.iterdir() if folder.is_dir()]
    folders = folders[:SAMPLE_LIMIT] if SAMPLE_LIMIT else folders

    print(f"Processing {len(folders)} folders...\n")

    rows = []
    for index, folder in enumerate(folders, start=1):
        detected, image_count, source_image = scan_folder(client, folder)
        result = get_result(folder.name, detected)

        rows.append(
            {
                "Folder ID": folder.name,
                "Expected Lamp Number": folder.name,
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
