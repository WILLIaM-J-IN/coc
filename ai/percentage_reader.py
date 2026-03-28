"""
percentage_reader.py
====================
Reads the destruction percentage from a CoC battle result screenshot using EasyOCR.
No external software installation required.

Requirements:
    pip install easyocr
    (first run will auto-download ~100MB model)
"""

import cv2
import os
import re
import time
import easyocr

RESULT_DIR = "result"

# Initialize EasyOCR reader once (English only, no GPU required)
# This takes a few seconds on first call and downloads the model on first run
_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        print("[Percentage] Initializing EasyOCR reader...")
        _reader = easyocr.Reader(['en'], gpu=False)
        print("[Percentage] EasyOCR ready")
    return _reader


def read_percentage(screenshot, save_debug=True):
    """
    Crop the center-top region of the screenshot and read the destruction
    percentage number (0-100) using EasyOCR.

    Works for both victory and defeat screens.

    Args:
        screenshot : numpy BGR image (from screenshot_window)
        save_debug : save debug images to result/ for inspection

    Returns:
        int (0-100) if recognised successfully, None otherwise
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    timestamp = int(time.time())

    h, w = screenshot.shape[:2]

    # ===== Step 1: crop the percentage region =====
    # "推毁率: XX%" always appears in the center star, top-center of screen
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)
    y1 = int(h * 0.05)
    y2 = int(h * 0.25)

    roi = screenshot[y1:y2, x1:x2]

    if save_debug:
        debug_full = screenshot.copy()
        cv2.rectangle(debug_full, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(RESULT_DIR, f"percentage_region_{timestamp}.jpg"), debug_full)
        cv2.imwrite(os.path.join(RESULT_DIR, f"percentage_roi_{timestamp}.jpg"), roi)

    # ===== Step 2: preprocess — isolate green text (percentage number is green) =====
    hsv        = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (40, 80, 80), (90, 255, 255))

    if cv2.countNonZero(green_mask) >= 50:
        # isolate green channel
        processed = cv2.bitwise_and(roi, roi, mask=green_mask)
        gray      = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        # fallback: plain grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # enlarge for better OCR accuracy
    resized   = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    ocr_input = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    if save_debug:
        cv2.imwrite(os.path.join(RESULT_DIR, f"percentage_preprocessed_{timestamp}.jpg"), ocr_input)

    # ===== Step 3: EasyOCR =====
    reader   = _get_reader()
    results  = reader.readtext(ocr_input, allowlist='0123456789%', detail=1)
    raw_text = " ".join([r[1] for r in results])
    print(f"[Percentage] OCR raw text: '{raw_text}'")

    # ===== Step 4: extract valid percentage =====

    # priority: number immediately before % sign
    pct_match = re.search(r'(\d+)\s*%', raw_text)
    if pct_match:
        value = min(int(pct_match.group(1)), 100)
        print(f"[Percentage] Result: {value}%")

        if save_debug:
            result_img = screenshot.copy()
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(result_img, f"Score: {value}%",
                        (x1, max(y1 - 15, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imwrite(
                os.path.join(RESULT_DIR, f"score_{value}pct_{timestamp}.jpg"),
                result_img
            )
        return value

    # fallback: take the largest number in 0-100 range
    numbers = re.findall(r'\d+', raw_text)
    candidates = [int(n) for n in numbers if 0 <= int(n) <= 100]
    if candidates:
        value = max(candidates)
        print(f"[Percentage] Result (fallback largest): {value}%")

        if save_debug:
            result_img = screenshot.copy()
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(result_img, f"Score: {value}%",
                        (x1, max(y1 - 15, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imwrite(
                os.path.join(RESULT_DIR, f"score_{value}pct_{timestamp}.jpg"),
                result_img
            )
        return value

    print(f"[Percentage] No valid number found. Raw: '{raw_text}'")
    return None


def evaluate_score(percentage):
    """
    Convert percentage to a rating label.

    Returns:
        'excellent'  >= 80%
        'average'    >= 50%
        'poor'       <  50%
        'unknown'    if None
    """
    if percentage is None:
        return "unknown"
    if percentage >= 80:
        return "excellent"
    if percentage >= 50:
        return "average"
    return "poor"