from pathlib import Path
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

EXPECTED_LEN = 4
WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# ถ้า Windows ปลด comment
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def clean_text(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()


def read_gray(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def preprocess_simple(gray):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def preprocess_strong(gray):
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


# -----------------------------
# WHOLE IMAGE OCR
# -----------------------------
def ocr_whole_image(img, psm=8):
    text = pytesseract.image_to_string(
        img,
        config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist={WHITELIST}"
    )
    return clean_text(text)[:EXPECTED_LEN]


def ocr_whole_image_multi(img):
    """
    ลองหลาย config แล้วเลือกผลที่ดูดีที่สุด
    """
    candidates = []

    for psm in [7, 8, 13]:
        text = pytesseract.image_to_string(
            img,
            config=f"--oem 3 --psm {psm} -c tessedit_char_whitelist={WHITELIST}"
        )
        cleaned = clean_text(text)

        score = 0
        if len(cleaned) == EXPECTED_LEN:
            score += 100
        else:
            score += max(0, 40 - abs(len(cleaned) - EXPECTED_LEN) * 10)

        if re.fullmatch(r"[A-Z0-9]+", cleaned or ""):
            score += 20

        candidates.append((score, cleaned[:EXPECTED_LEN], f"psm={psm}"))

    candidates.sort(key=lambda x: (-x[0], x[1]))
    best = candidates[0]
    return {
        "text": best[1],
        "meta": best[2],
        "all": candidates
    }


# -----------------------------
# SEGMENT OCR
# -----------------------------
def split_4(img):
    h, w = img.shape
    parts = []
    for i in range(4):
        x1 = int(i * w / 4)
        x2 = int((i + 1) * w / 4)
        x2 = max(x2, x1 + 1)
        parts.append(img[:, x1:x2])
    return parts


def ocr_single_char(img):
    pil = Image.fromarray(img)
    text = pytesseract.image_to_string(
        pil,
        config=f"--oem 3 --psm 10 -c tessedit_char_whitelist={WHITELIST}"
    )
    return clean_text(text)[:1]


def ocr_segmented(img):
    chars = []
    for part in split_4(img):
        chars.append(ocr_single_char(part))
    return "".join(chars)


# -----------------------------
# RUN COMPARE
# -----------------------------
def run_compare(image_path: Path):
    gray = read_gray(image_path)

    simple = preprocess_simple(gray)
    strong = preprocess_strong(gray)

    thai_raw = ocr_whole_image_multi(gray)
    thai_simple = ocr_whole_image_multi(simple)
    thai_strong = ocr_whole_image_multi(strong)

    result = {
        "file": image_path.name,
        "expected": image_path.stem.upper(),

        "thai_raw": thai_raw["text"],
        "thai_raw_meta": thai_raw["meta"],

        "thai_simple": thai_simple["text"],
        "thai_simple_meta": thai_simple["meta"],

        "thai_strong": thai_strong["text"],
        "thai_strong_meta": thai_strong["meta"],

        "seg_simple": ocr_segmented(simple),
        "seg_strong": ocr_segmented(strong),
    }

    return result


def evaluate(folder):
    folder = Path(folder)
    results = []

    print("\n===== RUN OCR LAB =====\n")

    for img in sorted(folder.iterdir()):
        if img.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
            continue

        # ตัดไฟล์ preprocess/debug ออก
        if img.name.startswith("processed_"):
            continue
        if img.name.startswith("debug_"):
            continue

        r = run_compare(img)
        results.append(r)

        print(f"\nFILE: {r['file']} (expected={r['expected']})")
        print(f"thai_raw      : {r['thai_raw']}   [{r['thai_raw_meta']}]")
        print(f"thai_simple   : {r['thai_simple']}   [{r['thai_simple_meta']}]")
        print(f"thai_strong   : {r['thai_strong']}   [{r['thai_strong_meta']}]")
        print(f"seg_simple    : {r['seg_simple']}")
        print(f"seg_strong    : {r['seg_strong']}")

    print("\n===== SUMMARY =====\n")

    keys = ["thai_raw", "thai_simple", "thai_strong", "seg_simple", "seg_strong"]

    for k in keys:
        correct = sum(1 for r in results if r[k] == r["expected"])
        total = len(results)
        acc = (correct / total * 100) if total else 0
        print(f"{k:15} => {correct}/{total} = {acc:.2f}%")

    return results


if __name__ == "__main__":
    evaluate("images/set1/5IHS.jpg")