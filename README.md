# Taiwan Lamp Plate Auto Detection

This project experiments with automatic OCR detection for Taiwan lamp plate images. The goal is to detect the numeric lamp plate ID from field photos, including difficult images where the plate may be unclear, shiny, tilted, cropped, or partially covered by green plants.

The project currently contains two main OCR approaches:

- `test.py`: offline OCR using PaddleOCR.
- `test_API_local.py`: cloud OCR using Google Vision API with optional OpenCV preprocessing.

Based on testing so far, the Google Vision API approach is more accurate than the offline PaddleOCR approach for this use case.

## Project Files

| File | Purpose |
| --- | --- |
| `test.py` | Offline OCR experiment using PaddleOCR. This can run without Google Vision, but accuracy is currently not strong enough for difficult lamp plate images. |
| `test_API_local.py` | Main recommended script. Uses Google Vision API and OpenCV preprocessing to detect lamp plate numbers from local images. |
| `test_google_vision.py` | Earlier/simple Google Vision test script. Useful as a reference, but `test_API_local.py` is the better working version. |
| `debug_folders.py` | Helper script for checking folder/file names during earlier dataset exploration. |
| `requirements-google-vision.txt` | Python dependencies for the Google Vision workflow. |
| `vision-cred.json` | Local Google Cloud credential file. This file should not be committed or shared publicly. |

## OCR Strategy

For production or manager demo purposes, the safest OCR strategy is:

1. Send the raw/original image to Google Vision first.
2. If a valid lamp plate number is detected, stop immediately.
3. If raw OCR fails, try OpenCV preprocessing variants.
4. Use preprocessing as a fallback, not as the first mandatory step.
5. Save which image variant produced the detected result.

This is important because aggressive preprocessing can sometimes distort digits. For example, a cropped or thresholded image may cause similar digits such as `0`, `6`, `8`, and `9` to be misread. If the raw image is already readable, Google Vision may perform better on the original image.

## Current Google Vision Workflow

`test_API_local.py` scans all image files under `BASE_PATH`, including subfolders.

Supported image extensions:

```python
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
```

The script detects fixed-length numeric IDs using:

```python
OCR_DIGIT_COUNT = 5
```

Change this value if your lamp plate number has a different length:

```python
OCR_DIGIT_COUNT = 6
```

The output is saved as an Excel file:

```text
inspection_test_data_1.xlsx
inspection_test_data_2.xlsx
...
```

The Excel output includes:

- Image name
- Image path
- First detected number
- All detected numbers
- Detection status
- Source image variant

## OpenCV Preprocessing

`test_API_local.py` includes OpenCV preprocessing for difficult field images:

- Resize small images
- Reduce strong glare or reflection
- Suppress green obstruction from plants/leaves
- Improve contrast with CLAHE
- Sharpen plate text
- Adaptive thresholding
- Try plate crop detection
- Try small-angle deskew variants

Preprocessing is useful for bad images, but it should be controlled carefully because each preprocessed variant sent to Google Vision is another API request.

## Cloud Cost Safety

Google Vision API charges per OCR request. If one original image creates many preprocessed variants, the cost can multiply quickly.

Example:

```text
1 image x 1 raw request = 1 Vision request
1 image x 8 variants = 8 Vision requests
```

For cost safety:

- Keep `SAMPLE_LIMIT` low during testing.
- Keep `MAX_PREPROCESSING_VARIANTS` low.
- Prefer raw-first detection.
- Stop after the first valid result.
- Use Google Cloud billing alerts and quota limits.
- Do not run the full dataset until the pipeline is tested on a small sample.

Recommended demo settings:

```python
SAMPLE_LIMIT = 20
MAX_PREPROCESSING_VARIANTS = 3
USE_PREPROCESSING = True
```

## Installation

Create and activate a Python virtual environment:

```powershell
cd "C:\Users\Rangga Saputra\Documents\OCR_Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements-google-vision.txt
```

## Google Vision Credential Setup

Set the Google Vision credential environment variable before running the script:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\Rangga Saputra\Documents\OCR_Project\vision-cred.json"
```

The script checks this environment variable before starting.

## Running Google Vision OCR

Edit the configuration at the top of `test_API_local.py`:

```python
BASE_PATH = Path(r"C:\Users\Rangga Saputra\Documents\Test_data")
IMAGE_NAME_CONTAINS = ""
SAMPLE_LIMIT = None
OCR_DIGIT_COUNT = 5
USE_PREPROCESSING = True
MAX_PREPROCESSING_VARIANTS = 8
```

Then run:

```powershell
python test_API_local.py
```

## Running Offline PaddleOCR

`test.py` uses PaddleOCR locally. This avoids Google Vision API cost, but the accuracy was weaker in the current test images.

Run:

```powershell
python test.py
```

Use this script mainly as an offline experiment or backup approach.

## Recommended Next Improvements

- Make Google Vision raw-first, then preprocessing fallback.
- Stop OCR attempts after the first valid detected plate number.
- Add a confidence/ranking rule when multiple numbers are detected.
- Save failed images into a review list.
- Add resume support for large production batches.
- Add a daily request limit before processing millions of images.
- Compare accuracy between raw image, cropped image, and preprocessed image.

---

# 台灣路燈牌自動辨識專案

本專案用於測試台灣路燈牌照片的自動 OCR 辨識。目標是從現場拍攝的照片中，自動讀取路燈牌上的數字編號。實際照片可能會有模糊、反光、傾斜、裁切、或被植物遮住等問題。

目前專案包含兩種主要 OCR 方法：

- `test.py`：使用 PaddleOCR 的離線辨識版本。
- `test_API_local.py`：使用 Google Vision API，並搭配 OpenCV 前處理的版本。

目前測試結果顯示，針對這類路燈牌照片，Google Vision API 的辨識準確度比離線 PaddleOCR 更好。

## 專案檔案說明

| 檔案 | 用途 |
| --- | --- |
| `test.py` | PaddleOCR 離線辨識實驗。優點是不需要呼叫雲端 API，但目前對困難照片的準確度不夠穩定。 |
| `test_API_local.py` | 目前建議使用的主程式。使用 Google Vision API，並搭配 OpenCV 前處理來辨識本機圖片。 |
| `test_google_vision.py` | 較早期、較簡單的 Google Vision 測試版本。可作為參考，但主要建議使用 `test_API_local.py`。 |
| `debug_folders.py` | 用來檢查資料夾與檔名的輔助工具。 |
| `requirements-google-vision.txt` | Google Vision 流程需要安裝的 Python 套件。 |
| `vision-cred.json` | Google Cloud 認證檔。此檔案不應上傳或公開分享。 |

## OCR 辨識策略

若要用於正式流程或向主管展示，建議採用最安全的辨識策略：

1. 先將原始圖片送到 Google Vision 辨識。
2. 如果已經成功辨識出有效的路燈牌號碼，就立刻停止。
3. 如果原始圖片辨識失敗，再嘗試 OpenCV 前處理版本。
4. 前處理應作為備援方案，而不是每張圖片都強制先做。
5. 記錄是哪一種圖片版本成功辨識。

這點很重要，因為過度前處理可能會讓數字形狀失真。例如裁切、二值化或反光修正後，`0`、`6`、`8`、`9` 這些相近數字可能被誤判。如果原始圖片本身可讀，Google Vision 通常可能在原圖上表現更穩定。

## 目前 Google Vision 流程

`test_API_local.py` 會掃描 `BASE_PATH` 底下所有圖片，包含子資料夾。

支援圖片格式：

```python
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
```

目前程式用以下設定辨識固定長度的數字：

```python
OCR_DIGIT_COUNT = 5
```

如果你的路燈牌號碼是 6 碼，請改成：

```python
OCR_DIGIT_COUNT = 6
```

辨識結果會輸出成 Excel 檔：

```text
inspection_test_data_1.xlsx
inspection_test_data_2.xlsx
...
```

Excel 內容包含：

- 圖片檔名
- 圖片路徑
- 第一個辨識到的號碼
- 所有辨識到的號碼
- 辨識狀態
- 成功辨識的圖片版本

## OpenCV 圖片前處理

`test_API_local.py` 目前包含針對現場路燈牌照片的 OpenCV 前處理：

- 放大小圖
- 降低強烈反光
- 降低綠色植物遮擋的影響
- 使用 CLAHE 增強對比
- 銳化文字
- 自適應二值化
- 嘗試裁切亮色牌面區域
- 嘗試小角度校正傾斜

前處理對困難圖片有幫助，但必須小心控制，因為每一個送到 Google Vision 的前處理版本都會產生一次 API 請求。

## 雲端費用安全

Google Vision API 是依照 OCR 請求次數計費。如果一張原圖產生多個前處理版本，費用會快速增加。

範例：

```text
1 張圖片 x 1 次原圖請求 = 1 次 Vision 請求
1 張圖片 x 8 個版本 = 8 次 Vision 請求
```

為了控制費用：

- 測試時先把 `SAMPLE_LIMIT` 設小。
- 把 `MAX_PREPROCESSING_VARIANTS` 設小。
- 採用原圖優先的辨識方式。
- 找到第一個有效號碼後就停止。
- 在 Google Cloud 設定帳單警示與配額限制。
- 在小樣本測試完成前，不要直接跑完整大量資料。

建議展示用設定：

```python
SAMPLE_LIMIT = 20
MAX_PREPROCESSING_VARIANTS = 3
USE_PREPROCESSING = True
```

## 安裝方式

建立並啟用 Python 虛擬環境：

```powershell
cd "C:\Users\Rangga Saputra\Documents\OCR_Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安裝套件：

```powershell
pip install -r requirements-google-vision.txt
```

## Google Vision 認證設定

執行前需要先設定 Google Vision 認證檔路徑：

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\Users\Rangga Saputra\Documents\OCR_Project\vision-cred.json"
```

程式啟動時會檢查這個環境變數。

## 執行 Google Vision OCR

先修改 `test_API_local.py` 上方設定：

```python
BASE_PATH = Path(r"C:\Users\Rangga Saputra\Documents\Test_data")
IMAGE_NAME_CONTAINS = ""
SAMPLE_LIMIT = None
OCR_DIGIT_COUNT = 5
USE_PREPROCESSING = True
MAX_PREPROCESSING_VARIANTS = 8
```

然後執行：

```powershell
python test_API_local.py
```

## 執行離線 PaddleOCR

`test.py` 使用 PaddleOCR 進行本機離線辨識。此方法不會產生 Google Vision API 費用，但目前在測試照片中的準確度較弱。

執行：

```powershell
python test.py
```

此版本目前比較適合作為離線實驗或備援方案。

## 建議後續改善

- 將 Google Vision 流程調整為原圖優先，再用前處理作為備援。
- 找到第一個有效號碼後立刻停止，避免不必要的 API 請求。
- 當同一張圖片偵測到多個號碼時，加入排序或信心判斷規則。
- 將辨識失敗的圖片整理成待人工檢查清單。
- 對大量資料加入中斷後續跑的 resume 機制。
- 在處理大量圖片前，先設定每日請求上限。
- 比較原圖、裁切圖、前處理圖的準確度差異。
