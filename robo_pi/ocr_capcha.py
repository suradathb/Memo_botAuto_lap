from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
import re
import os

import cv2
import pytesseract
from PIL import Image

# =========================
# OCR ENGINE TYPES
# =========================
LocalEngine = Literal["tesseract"]
OnlineEngine = Literal["typhoon"]
OCREngine = Literal["tesseract", "typhoon"]


# =========================
# BASIC SETTINGS
# =========================
DEFAULT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# =========================
# CONFIG FUNCTIONS
# =========================
def configure_tesseract(tesseract_exe_path: Optional[str] = None) -> None:
    """
    ตั้งค่า path ของ Tesseract สำหรับ Windows ถ้าจำเป็น

    ตัวอย่าง:
        configure_tesseract(r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    """
    if tesseract_exe_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe_path


def get_engine_mode(engine: OCREngine) -> str:
    """
    บอกชัดว่า engine นี้เป็น LOCAL หรือ ONLINE
    """
    if engine == "tesseract":
        return "LOCAL"
    if engine == "typhoon":
        return "ONLINE"
    raise ValueError(f"Unsupported engine: {engine}")


def ensure_typhoon_import():
    """
    import typhoon_ocr เฉพาะตอนจะใช้
    """
    try:
        from typhoon_ocr import ocr_document
        return ocr_document
    except ImportError as e:
        raise ImportError(
            "Typhoon OCR is not installed. Install it first with pip install typhoon-ocr"
        ) from e


def check_typhoon_api_key() -> str:
    """
    เช็ก API key สำหรับ Typhoon OCR
    """
    api_key = (
        os.getenv("TYPHOON_OCR_API_KEY")
        or os.getenv("TYPHOON_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Typhoon OCR requires API key.\n"
            "Please set one of these environment variables:\n"
            "- TYPHOON_OCR_API_KEY\n"
            "- TYPHOON_API_KEY\n"
            "- OPENAI_API_KEY"
        )
    return api_key


# =========================
# TEXT / METRIC FUNCTIONS
# =========================
def clean_text(text: str) -> str:
    """
    ล้างข้อความให้เหลือ A-Z และ 0-9 เท่านั้น
    """
    return re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()


def char_accuracy(pred: str, truth: str) -> float:
    """
    วัดความแม่นแบบ character-level
    """
    if not truth:
        return 0.0
    matches = sum(1 for a, b in zip(pred, truth) if a == b)
    return matches / max(len(truth), len(pred), 1)


def exact_match(pred: str, truth: str) -> float:
    """
    ตรงเป๊ะทั้งคำ = 1.0 ไม่งั้น 0.0
    """
    return 1.0 if pred == truth else 0.0


# =========================
# IMAGE PROCESSING FUNCTIONS
# =========================
def preprocess_image(
    input_path: Path,
    save_processed: bool = True,
    output_prefix: str = "processed_",
) -> Path:
    """
    preprocess รูปแล้วคืน path ของรูปที่ผ่าน preprocess
    """
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Cannot read image: {input_path}")

    # resize ใหญ่ขึ้นเล็กน้อย
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur ลด noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # threshold
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    out_path = input_path.parent / f"{output_prefix}{input_path.name}"
    if save_processed:
        cv2.imwrite(str(out_path), thresh)

    return out_path


# =========================
# OCR ENGINE FUNCTIONS
# =========================
def run_tesseract_ocr(
    image_path: Path,
    save_processed: bool = True,
) -> str:
    """
    OCR แบบ LOCAL ด้วย Tesseract
    """
    processed_path = preprocess_image(image_path, save_processed=save_processed)

    # psm 7 = คาดว่ามี text บรรทัดเดียว
    # whitelist = บังคับ charset
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    text = pytesseract.image_to_string(Image.open(processed_path), config=config)
    return clean_text(text)


def run_typhoon_ocr(
    image_path: Path,
    save_processed: bool = True,
) -> str:
    """
    OCR แบบ ONLINE ด้วย Typhoon OCR API
    """
    check_typhoon_api_key()
    ocr_document = ensure_typhoon_import()

    processed_path = preprocess_image(image_path, save_processed=save_processed)
    text = ocr_document(str(processed_path))
    return clean_text(text)


def run_ocr(
    image_path: Path,
    engine: OCREngine = "tesseract",
    save_processed: bool = True,
) -> str:
    """
    จุดเรียกกลาง ใช้ engine ที่ต้องการ
    """
    if engine == "tesseract":
        return run_tesseract_ocr(image_path, save_processed=save_processed)
    if engine == "typhoon":
        return run_typhoon_ocr(image_path, save_processed=save_processed)

    raise ValueError(f"Unsupported engine: {engine}")


# =========================
# BENCHMARK FUNCTIONS
# =========================
def benchmark_folder(
    image_dir: str | Path,
    engine: OCREngine = "tesseract",
    extensions: Optional[set[str]] = None,
    save_processed: bool = True,
    skip_processed_files: bool = True,
) -> dict:
    """
    benchmark OCR ทั้งโฟลเดอร์
    ใช้ชื่อไฟล์เป็น ground truth เช่น AB12.jpg -> GT = AB12
    """
    image_dir = Path(image_dir)
    extensions = extensions or DEFAULT_EXTENSIONS

    if not image_dir.exists():
        raise FileNotFoundError(f"Folder not found: {image_dir}")

    print(f"\n[INFO] Engine      : {engine}")
    print(f"[INFO] Engine mode : {get_engine_mode(engine)}")
    print(f"[INFO] Folder      : {image_dir}\n")

    char_scores: list[float] = []
    exact_scores: list[float] = []
    total_files = 0

    for img in image_dir.iterdir():
        if img.suffix.lower() not in extensions:
            continue

        if skip_processed_files and img.name.startswith("processed_"):
            continue

        total_files += 1
        truth = clean_text(img.stem)
        pred = run_ocr(img, engine=engine, save_processed=save_processed)

        c_acc = char_accuracy(pred, truth)
        e_acc = exact_match(pred, truth)

        char_scores.append(c_acc)
        exact_scores.append(e_acc)

        print(
            f"{img.name} | GT={truth} | OCR={pred} | "
            f"CHAR_ACC={c_acc:.2f} | EXACT={e_acc:.0f}"
        )

    if total_files == 0:
        raise ValueError(f"No valid image files found in: {image_dir}")

    result = {
        "engine": engine,
        "engine_mode": get_engine_mode(engine),
        "total_files": total_files,
        "avg_char_accuracy": sum(char_scores) / len(char_scores),
        "avg_exact_match": sum(exact_scores) / len(exact_scores),
    }

    print("\n=== SUMMARY ===")
    print(f"Engine             : {result['engine']}")
    print(f"Engine Mode        : {result['engine_mode']}")
    print(f"Total Files        : {result['total_files']}")
    print(f"Avg Char Accuracy  : {result['avg_char_accuracy']:.2f}")
    print(f"Avg Exact Match    : {result['avg_exact_match']:.2f}")

    return result


def benchmark_compare(
    image_dir: str | Path,
    save_processed: bool = True,
) -> dict:
    """
    เทียบ LOCAL กับ ONLINE ในชุดเดียวกัน
    - tesseract = LOCAL
    - typhoon   = ONLINE
    """
    results = {}

    results["tesseract"] = benchmark_folder(
        image_dir=image_dir,
        engine="tesseract",
        save_processed=save_processed,
    )

    results["typhoon"] = benchmark_folder(
        image_dir=image_dir,
        engine="typhoon",
        save_processed=save_processed,
    )

    return results


def test_single_image(
    image_path: str | Path,
    engine: OCREngine = "tesseract",
    save_processed: bool = True,
) -> dict:
    """
    ทดสอบ OCR กับรูปเดียว
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    truth = clean_text(image_path.stem)
    pred = run_ocr(image_path, engine=engine, save_processed=save_processed)

    result = {
        "file": image_path.name,
        "engine": engine,
        "engine_mode": get_engine_mode(engine),
        "ground_truth": truth,
        "prediction": pred,
        "char_accuracy": char_accuracy(pred, truth),
        "exact_match": exact_match(pred, truth),
    }

    print("\n=== SINGLE IMAGE RESULT ===")
    print(f"File         : {result['file']}")
    print(f"Engine       : {result['engine']}")
    print(f"Engine Mode  : {result['engine_mode']}")
    print(f"GT           : {result['ground_truth']}")
    print(f"OCR          : {result['prediction']}")
    print(f"Char Acc     : {result['char_accuracy']:.2f}")
    print(f"Exact Match  : {result['exact_match']:.0f}")

    return result


# =========================
# EXAMPLE USAGE
# =========================
if __name__ == "__main__":
    # ถ้าใช้ Windows และลง Tesseract ไว้ path นี้
    # configure_tesseract(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    # ทดสอบรูปเดียว แบบ LOCAL
    # test_single_image("images/set1/5IHS.jpg", engine="tesseract")

    # ทดสอบทั้งโฟลเดอร์ แบบ LOCAL
    # benchmark_folder("images/set1", engine="tesseract")

    # ทดสอบทั้งโฟลเดอร์ แบบ ONLINE
    benchmark_folder("images/set1", engine="typhoon")

    # เทียบทั้งสองตัว
    # benchmark_compare("images/set1")