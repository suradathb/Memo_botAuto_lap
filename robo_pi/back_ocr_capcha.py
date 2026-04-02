from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal
import os
import re

import cv2
import numpy as np
import pytesseract
from PIL import Image


# ============================================
# OCR ENGINE TYPES
# ============================================
OCREngine = Literal["tesseract", "typhoon"]


# ============================================
# GLOBAL CONFIG
# ============================================
DEFAULT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEBUG_OUTPUT_DIRNAME = "debug_steps"


# ============================================
# BASIC HELPERS
# ============================================
def clean_text(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()


def char_accuracy(pred: str, truth: str) -> float:
    if not truth:
        return 0.0
    matches = sum(1 for a, b in zip(pred, truth) if a == b)
    return matches / max(len(truth), len(pred), 1)


def exact_match(pred: str, truth: str) -> float:
    return 1.0 if pred == truth else 0.0


def get_engine_mode(engine: OCREngine) -> str:
    if engine == "tesseract":
        return "LOCAL"
    if engine == "typhoon":
        return "ONLINE"
    raise ValueError(f"Unsupported engine: {engine}")


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def read_image(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return img


# ============================================
# TESSERACT / ONLINE CONFIG
# ============================================
def configure_tesseract(tesseract_exe_path: Optional[str] = None) -> None:
    """
    ใช้เฉพาะกรณี Windows แล้ว pytesseract หา tesseract.exe ไม่เจอ
    """
    if tesseract_exe_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe_path


def ensure_typhoon_import():
    try:
        from typhoon_ocr import ocr_document
        return ocr_document
    except ImportError as e:
        raise ImportError(
            "Typhoon OCR is not installed. Install it with: pip install typhoon-ocr"
        ) from e


def check_typhoon_api_key() -> str:
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


# ============================================
# IMAGE PROCESSING STEPS
# ============================================
def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def upscale_image(gray: np.ndarray, scale: float = 2.0) -> np.ndarray:
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def denoise_image(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, (3, 3), 0)


def threshold_image(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    if method == "adaptive":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return binary


def ensure_dark_text(binary: np.ndarray) -> np.ndarray:
    """
    บังคับให้พื้นหลังขาว ตัวอักษรดำ
    """
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)

    if black_pixels > white_pixels:
        return cv2.bitwise_not(binary)
    return binary


def close_text_gaps(binary: np.ndarray) -> np.ndarray:
    """
    ถมช่องว่างเล็ก ๆ ในตัวอักษร
    """
    inverted = cv2.bitwise_not(binary)
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.bitwise_not(closed)


def strengthen_text(binary: np.ndarray) -> np.ndarray:
    """
    ทำให้ตัวอักษรเข้ม/หนาขึ้นเล็กน้อย
    """
    inverted = cv2.bitwise_not(binary)
    kernel = np.ones((2, 2), np.uint8)
    thick = cv2.dilate(inverted, kernel, iterations=1)
    return cv2.bitwise_not(thick)


def remove_horizontal_lines(binary: np.ndarray, min_line_width: int = 40) -> np.ndarray:
    inverted = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_width, 1))
    detected = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.subtract(inverted, detected)
    return cv2.bitwise_not(cleaned)


def remove_vertical_lines(binary: np.ndarray, min_line_height: int = 40) -> np.ndarray:
    inverted = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_height))
    detected = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.subtract(inverted, detected)
    return cv2.bitwise_not(cleaned)


def remove_small_noise(binary: np.ndarray, min_area: int = 20) -> np.ndarray:
    inverted = cv2.bitwise_not(binary)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    cleaned = np.zeros_like(inverted)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cv2.bitwise_not(cleaned)


# ============================================
# PREPROCESS PIPELINE
# ============================================
def preprocess_for_ocr(
    image_path: Path,
    output_dir: Optional[Path] = None,
    threshold_method: str = "otsu",
    scale: float = 2.0,
    remove_h_lines: bool = True,
    remove_v_lines: bool = True,
    strengthen: bool = True,
    fill_gaps: bool = True,
    remove_noise: bool = True,
    debug: bool = False,
) -> Path:
    """
    preprocess หลัก
    คืน path ของไฟล์ processed สุดท้าย
    """
    output_dir = output_dir or image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / DEBUG_OUTPUT_DIRNAME if debug else None

    original = read_image(image_path)
    if debug:
        save_image(debug_dir / f"01_original_{image_path.name}", original)

    gray = to_grayscale(original)
    if debug:
        save_image(debug_dir / f"02_gray_{image_path.name}", gray)

    gray = upscale_image(gray, scale=scale)
    if debug:
        save_image(debug_dir / f"03_upscale_{image_path.name}", gray)

    gray = denoise_image(gray)
    if debug:
        save_image(debug_dir / f"04_denoise_{image_path.name}", gray)

    binary = threshold_image(gray, method=threshold_method)
    if debug:
        save_image(debug_dir / f"05_threshold_{image_path.name}", binary)

    binary = ensure_dark_text(binary)
    if debug:
        save_image(debug_dir / f"06_dark_text_{image_path.name}", binary)

    if fill_gaps:
        binary = close_text_gaps(binary)
        if debug:
            save_image(debug_dir / f"07_fill_gaps_{image_path.name}", binary)

    if strengthen:
        binary = strengthen_text(binary)
        if debug:
            save_image(debug_dir / f"08_strengthen_{image_path.name}", binary)

    if remove_h_lines:
        binary = remove_horizontal_lines(binary, min_line_width=40)
        if debug:
            save_image(debug_dir / f"09_remove_hlines_{image_path.name}", binary)

    if remove_v_lines:
        binary = remove_vertical_lines(binary, min_line_height=40)
        if debug:
            save_image(debug_dir / f"10_remove_vlines_{image_path.name}", binary)

    if remove_noise:
        binary = remove_small_noise(binary, min_area=20)
        if debug:
            save_image(debug_dir / f"11_remove_noise_{image_path.name}", binary)

    output_path = output_dir / f"processed_{image_path.name}"
    save_image(output_path, binary)

    if debug:
        print(f"[DEBUG] Saved step images to: {debug_dir}")

    return output_path


# ============================================
# OCR ENGINES
# ============================================
def run_tesseract_ocr(
    image_path: Path,
    debug: bool = False,
    output_dir: Optional[Path] = None,
) -> str:
    processed_path = preprocess_for_ocr(
        image_path=image_path,
        output_dir=output_dir,
        debug=debug,
    )

    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(Image.open(processed_path), config=config)
    return clean_text(text)


def run_typhoon_ocr(
    image_path: Path,
    debug: bool = False,
    output_dir: Optional[Path] = None,
) -> str:
    check_typhoon_api_key()
    ocr_document = ensure_typhoon_import()

    processed_path = preprocess_for_ocr(
        image_path=image_path,
        output_dir=output_dir,
        debug=debug,
    )

    text = ocr_document(str(processed_path))
    return clean_text(text)


def run_ocr(
    image_path: Path,
    engine: OCREngine = "tesseract",
    debug: bool = False,
    output_dir: Optional[Path] = None,
) -> str:
    if engine == "tesseract":
        return run_tesseract_ocr(image_path=image_path, debug=debug, output_dir=output_dir)

    if engine == "typhoon":
        return run_typhoon_ocr(image_path=image_path, debug=debug, output_dir=output_dir)

    raise ValueError(f"Unsupported engine: {engine}")


# ============================================
# TEST / BENCHMARK
# ============================================
def test_single_image(
    image_path: str | Path,
    engine: OCREngine = "tesseract",
    debug: bool = True,
    output_dir: Optional[str | Path] = None,
) -> dict:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir_path = Path(output_dir) if output_dir else image_path.parent

    truth = clean_text(image_path.stem)
    pred = run_ocr(
        image_path=image_path,
        engine=engine,
        debug=debug,
        output_dir=output_dir_path,
    )

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


def benchmark_folder(
    image_dir: str | Path,
    engine: OCREngine = "tesseract",
    debug: bool = False,
    output_dir: Optional[str | Path] = None,
    extensions: Optional[set[str]] = None,
) -> dict:
    image_dir = Path(image_dir)
    extensions = extensions or DEFAULT_EXTENSIONS

    if not image_dir.exists():
        raise FileNotFoundError(f"Folder not found: {image_dir}")

    output_dir_path = Path(output_dir) if output_dir else image_dir

    print(f"\n[INFO] Engine      : {engine}")
    print(f"[INFO] Engine mode : {get_engine_mode(engine)}")
    print(f"[INFO] Folder      : {image_dir}")
    print(f"[INFO] Debug       : {debug}\n")

    char_scores: list[float] = []
    exact_scores: list[float] = []
    total_files = 0

    for img in image_dir.iterdir():
        if img.suffix.lower() not in extensions:
            continue
        if img.name.startswith("processed_"):
            continue

        total_files += 1
        truth = clean_text(img.stem)

        pred = run_ocr(
            image_path=img,
            engine=engine,
            debug=debug,
            output_dir=output_dir_path,
        )

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
    debug: bool = False,
    output_dir: Optional[str | Path] = None,
) -> dict:
    results = {}
    results["tesseract"] = benchmark_folder(
        image_dir=image_dir,
        engine="tesseract",
        debug=debug,
        output_dir=output_dir,
    )

    results["typhoon"] = benchmark_folder(
        image_dir=image_dir,
        engine="typhoon",
        debug=debug,
        output_dir=output_dir,
    )

    return results


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    # ถ้า Windows หา tesseract ไม่เจอ ให้ปลดคอมเมนต์บรรทัดนี้
    # configure_tesseract(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    # ----------------------------------------
    # 1) ทดสอบรูปเดียว + เปิด debug
    # ----------------------------------------
    # test_single_image(
    #     image_path="images/set1/6UWG.jpg",
    #     engine="tesseract",
    #     debug=True,
    #     output_dir="images/set1"
    # )

    # ----------------------------------------
    # 2) ทดสอบทั้งโฟลเดอร์ + ปิด debug
    # ----------------------------------------
    benchmark_folder(
        image_dir="images/set2",
        engine="tesseract",
        debug=False,
        output_dir="images/set2",
    )

    # ----------------------------------------
    # 3) ถ้าจะลอง ONLINE
    # ----------------------------------------
    # benchmark_folder(
    #     image_dir="images/set1",
    #     engine="typhoon",
    #     debug=False,
    #     output_dir="images/set1",
    # )

    # ----------------------------------------
    # 4) เทียบ 2 engine
    # ----------------------------------------
    # benchmark_compare(
    #     image_dir="images/set1",
    #     debug=False,
    #     output_dir="images/set1",
    # )