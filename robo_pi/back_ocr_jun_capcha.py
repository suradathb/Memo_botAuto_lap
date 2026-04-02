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
# OCR ENGINE / PRESET TYPES
# ============================================
OCREngine = Literal["tesseract", "typhoon"]
PreprocessPreset = Literal[
    "basic",
    "faint",
    "aggressive",
    "white_bg_captcha",
    "white_bg_captcha_strong",
]


# ============================================
# GLOBAL CONFIG
# ============================================
DEFAULT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEBUG_OUTPUT_DIRNAME = "debug_steps"
EXPECTED_TEXT_LENGTH = 4
OCR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


# ============================================
# BASIC HELPERS
# ============================================
def clean_text(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()


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


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def add_white_border(image: np.ndarray, border: int = 20) -> np.ndarray:
    return cv2.copyMakeBorder(
        image,
        border,
        border,
        border,
        border,
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )


# ============================================
# TESSERACT / ONLINE CONFIG
# ============================================
def configure_tesseract(tesseract_exe_path: Optional[str] = None) -> None:
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
# BASIC GRAY PIPELINE HELPERS
# ============================================
def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def upscale_image(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def denoise_image(gray: np.ndarray, blur_ksize: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)


def enhance_contrast(gray: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def enhance_dark_text(gray: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    return blackhat


def threshold_image(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    if method == "adaptive":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return binary


def ensure_dark_text(binary: np.ndarray) -> np.ndarray:
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    if black_pixels > white_pixels:
        return cv2.bitwise_not(binary)
    return binary


def close_text_gaps(binary: np.ndarray, kernel_size: int = 2, iterations: int = 1) -> np.ndarray:
    inverted = cv2.bitwise_not(binary)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return cv2.bitwise_not(closed)


def strengthen_text(binary: np.ndarray, kernel_size: int = 2, iterations: int = 1) -> np.ndarray:
    inverted = cv2.bitwise_not(binary)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thick = cv2.dilate(inverted, kernel, iterations=iterations)
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
# WHITE-BACKGROUND CAPTCHA PIPELINE
# ============================================
def bgr_to_hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def mask_non_white_pixels(
    img_bgr: np.ndarray,
    s_min: int = 18,
    v_max: int = 245,
) -> np.ndarray:
    hsv = bgr_to_hsv(img_bgr)
    _, s, v = cv2.split(hsv)
    cond = (s >= s_min) | (v <= v_max)
    return np.where(cond, 255, 0).astype(np.uint8)


def suppress_near_gray_bright_noise(
    img_bgr: np.ndarray,
    base_mask: np.ndarray,
    gray_sat_max: int = 22,
    bright_val_min: int = 180,
) -> np.ndarray:
    hsv = bgr_to_hsv(img_bgr)
    _, s, v = cv2.split(hsv)
    near_gray_bright = ((s <= gray_sat_max) & (v >= bright_val_min)).astype(np.uint8) * 255
    return cv2.bitwise_and(base_mask, cv2.bitwise_not(near_gray_bright))


def clean_mask_open(mask: np.ndarray, kernel_size: int = 2, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)


def clean_mask_close(mask: np.ndarray, kernel_size: int = 2, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def filter_components_by_geometry(
    mask: np.ndarray,
    min_area: int = 12,
    min_height: int = 8,
    min_width: int = 2,
    max_area_ratio: float = 0.35,
) -> np.ndarray:
    h, w = mask.shape[:2]
    image_area = h * w

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros_like(mask)

    for i in range(1, num_labels):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue
        if cw < min_width or ch < min_height:
            continue
        if area > image_area * max_area_ratio:
            continue

        kept[labels == i] = 255

    return kept


def get_component_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[int, int, int, int, int]] = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        boxes.append((x, y, w, h, area))

    return boxes


def keep_top_components(mask: np.ndarray, keep_n: int = 12) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        components.append((i, x, y, w, h, area))

    components.sort(key=lambda item: item[5], reverse=True)
    selected_ids = {item[0] for item in components[:keep_n]}

    result = np.zeros_like(mask)
    for i in selected_ids:
        result[labels == i] = 255

    return result


def maybe_join_nearby_parts(mask: np.ndarray, kernel_w: int = 2, kernel_h: int = 2) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def invert_to_black_text_white_bg(mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(mask)


def normalize_binary(binary: np.ndarray) -> np.ndarray:
    _, out = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    return out


def draw_component_boxes_preview(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preview = img_bgr.copy()
    boxes = sorted(get_component_boxes(mask), key=lambda b: b[0])

    for (x, y, w, h, area) in boxes:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(
            preview,
            str(area),
            (x, max(10, y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return preview


def preprocess_white_bg_captcha(
    image_path: Path,
    output_dir: Optional[Path] = None,
    preset: PreprocessPreset = "white_bg_captcha",
    debug: bool = False,
) -> Path:
    output_dir = output_dir or image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = output_dir / DEBUG_OUTPUT_DIRNAME if debug else None

    original = read_image(image_path)
    if debug:
        save_image(debug_dir / f"01_original_{image_path.name}", original)

    scale = 3.0 if preset == "white_bg_captcha" else 3.5
    upscaled = upscale_image(original, scale=scale)
    if debug:
        save_image(debug_dir / f"02_upscaled_{image_path.name}", upscaled)

    blur_ksize = 3 if preset == "white_bg_captcha" else 5
    denoised = cv2.GaussianBlur(upscaled, (blur_ksize, blur_ksize), 0)
    if debug:
        save_image(debug_dir / f"03_denoised_color_{image_path.name}", denoised)

    if preset == "white_bg_captcha":
        raw_mask = mask_non_white_pixels(denoised, s_min=16, v_max=245)
        raw_mask = suppress_near_gray_bright_noise(
            denoised,
            raw_mask,
            gray_sat_max=22,
            bright_val_min=178,
        )
        open_k = 2
        close_k = 2
        min_area = 14
        min_height = 8
        keep_n = 14
        join_kw = 2
        join_kh = 2
    else:
        raw_mask = mask_non_white_pixels(denoised, s_min=12, v_max=248)
        raw_mask = suppress_near_gray_bright_noise(
            denoised,
            raw_mask,
            gray_sat_max=26,
            bright_val_min=172,
        )
        open_k = 2
        close_k = 3
        min_area = 10
        min_height = 7
        keep_n = 18
        join_kw = 3
        join_kh = 2

    if debug:
        save_image(debug_dir / f"04_non_white_mask_{image_path.name}", raw_mask)

    opened = clean_mask_open(raw_mask, kernel_size=open_k, iterations=1)
    if debug:
        save_image(debug_dir / f"05_open_clean_{image_path.name}", opened)

    closed = clean_mask_close(opened, kernel_size=close_k, iterations=1)
    if debug:
        save_image(debug_dir / f"06_close_clean_{image_path.name}", closed)

    filtered = filter_components_by_geometry(
        closed,
        min_area=min_area,
        min_height=min_height,
        min_width=2,
        max_area_ratio=0.35,
    )
    if debug:
        save_image(debug_dir / f"07_geometry_filtered_{image_path.name}", filtered)

    if debug:
        preview = draw_component_boxes_preview(upscaled, filtered)
        save_image(debug_dir / f"08_component_boxes_{image_path.name}", preview)

    top_components = keep_top_components(filtered, keep_n=keep_n)
    if debug:
        save_image(debug_dir / f"09_top_components_{image_path.name}", top_components)

    joined = maybe_join_nearby_parts(top_components, kernel_w=join_kw, kernel_h=join_kh)
    if debug:
        save_image(debug_dir / f"10_joined_parts_{image_path.name}", joined)

    final_mask = filter_components_by_geometry(
        joined,
        min_area=min_area,
        min_height=min_height,
        min_width=2,
        max_area_ratio=0.40,
    )
    if debug:
        save_image(debug_dir / f"11_final_mask_{image_path.name}", final_mask)

    result = invert_to_black_text_white_bg(final_mask)
    result = normalize_binary(result)
    if debug:
        save_image(debug_dir / f"12_final_binary_{image_path.name}", result)

    output_path = output_dir / f"processed_{image_path.name}"
    save_image(output_path, result)

    if debug:
        print(f"[DEBUG] Saved step images to: {debug_dir}")

    return output_path


# ============================================
# PRESET CONFIG
# ============================================
def get_preset_config(preset: PreprocessPreset) -> dict:
    if preset == "basic":
        return {
            "pipeline": "gray",
            "scale": 2.0,
            "blur_ksize": 3,
            "use_contrast": False,
            "use_blackhat": False,
            "threshold_method": "otsu",
            "fill_gaps": True,
            "fill_kernel": 2,
            "fill_iterations": 1,
            "strengthen": True,
            "strengthen_kernel": 2,
            "strengthen_iterations": 1,
            "remove_h_lines": True,
            "remove_v_lines": True,
            "remove_noise": True,
            "min_noise_area": 20,
            "min_h_line_width": 40,
            "min_v_line_height": 40,
        }

    if preset == "faint":
        return {
            "pipeline": "gray",
            "scale": 2.5,
            "blur_ksize": 3,
            "use_contrast": True,
            "contrast_clip_limit": 2.0,
            "contrast_tile_size": 8,
            "use_blackhat": True,
            "blackhat_kernel_size": 9,
            "threshold_method": "adaptive",
            "fill_gaps": True,
            "fill_kernel": 2,
            "fill_iterations": 1,
            "strengthen": True,
            "strengthen_kernel": 2,
            "strengthen_iterations": 1,
            "remove_h_lines": True,
            "remove_v_lines": True,
            "remove_noise": True,
            "min_noise_area": 20,
            "min_h_line_width": 40,
            "min_v_line_height": 40,
        }

    if preset == "aggressive":
        return {
            "pipeline": "gray",
            "scale": 3.0,
            "blur_ksize": 5,
            "use_contrast": True,
            "contrast_clip_limit": 3.0,
            "contrast_tile_size": 8,
            "use_blackhat": True,
            "blackhat_kernel_size": 11,
            "threshold_method": "adaptive",
            "fill_gaps": True,
            "fill_kernel": 3,
            "fill_iterations": 2,
            "strengthen": True,
            "strengthen_kernel": 2,
            "strengthen_iterations": 2,
            "remove_h_lines": True,
            "remove_v_lines": True,
            "remove_noise": True,
            "min_noise_area": 25,
            "min_h_line_width": 45,
            "min_v_line_height": 45,
        }

    if preset == "white_bg_captcha":
        return {"pipeline": "white_bg_captcha"}

    if preset == "white_bg_captcha_strong":
        return {"pipeline": "white_bg_captcha"}

    raise ValueError(f"Unsupported preset: {preset}")


# ============================================
# PREPROCESS PIPELINE
# ============================================
def preprocess_for_ocr(
    image_path: Path,
    output_dir: Optional[Path] = None,
    preset: PreprocessPreset = "basic",
    debug: bool = False,
) -> Path:
    output_dir = output_dir or image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_preset_config(preset)

    if cfg["pipeline"] == "white_bg_captcha":
        return preprocess_white_bg_captcha(
            image_path=image_path,
            output_dir=output_dir,
            preset=preset,
            debug=debug,
        )

    debug_dir = output_dir / DEBUG_OUTPUT_DIRNAME if debug else None

    original = read_image(image_path)
    if debug:
        save_image(debug_dir / f"01_original_{image_path.name}", original)

    gray = to_grayscale(original)
    if debug:
        save_image(debug_dir / f"02_gray_{image_path.name}", gray)

    gray = upscale_image(gray, scale=cfg["scale"])
    if debug:
        save_image(debug_dir / f"03_upscale_{image_path.name}", gray)

    gray = denoise_image(gray, blur_ksize=cfg["blur_ksize"])
    if debug:
        save_image(debug_dir / f"04_denoise_{image_path.name}", gray)

    if cfg.get("use_contrast", False):
        gray = enhance_contrast(
            gray,
            clip_limit=cfg.get("contrast_clip_limit", 2.0),
            tile_size=cfg.get("contrast_tile_size", 8),
        )
        if debug:
            save_image(debug_dir / f"05_contrast_{image_path.name}", gray)

    if cfg.get("use_blackhat", False):
        gray = enhance_dark_text(
            gray,
            kernel_size=cfg.get("blackhat_kernel_size", 9),
        )
        if debug:
            save_image(debug_dir / f"06_blackhat_{image_path.name}", gray)

    binary = threshold_image(gray, method=cfg["threshold_method"])
    if debug:
        save_image(debug_dir / f"07_threshold_{image_path.name}", binary)

    binary = ensure_dark_text(binary)
    if debug:
        save_image(debug_dir / f"08_dark_text_{image_path.name}", binary)

    if cfg.get("fill_gaps", False):
        binary = close_text_gaps(
            binary,
            kernel_size=cfg.get("fill_kernel", 2),
            iterations=cfg.get("fill_iterations", 1),
        )
        if debug:
            save_image(debug_dir / f"09_fill_gaps_{image_path.name}", binary)

    if cfg.get("strengthen", False):
        binary = strengthen_text(
            binary,
            kernel_size=cfg.get("strengthen_kernel", 2),
            iterations=cfg.get("strengthen_iterations", 1),
        )
        if debug:
            save_image(debug_dir / f"10_strengthen_{image_path.name}", binary)

    if cfg.get("remove_h_lines", False):
        binary = remove_horizontal_lines(
            binary,
            min_line_width=cfg.get("min_h_line_width", 40),
        )
        if debug:
            save_image(debug_dir / f"11_remove_hlines_{image_path.name}", binary)

    if cfg.get("remove_v_lines", False):
        binary = remove_vertical_lines(
            binary,
            min_line_height=cfg.get("min_v_line_height", 40),
        )
        if debug:
            save_image(debug_dir / f"12_remove_vlines_{image_path.name}", binary)

    if cfg.get("remove_noise", False):
        binary = remove_small_noise(
            binary,
            min_area=cfg.get("min_noise_area", 20),
        )
        if debug:
            save_image(debug_dir / f"13_remove_noise_{image_path.name}", binary)

    output_path = output_dir / f"processed_{image_path.name}"
    save_image(output_path, binary)

    if debug:
        print(f"[DEBUG] Saved step images to: {debug_dir}")

    return output_path


# ============================================
# SEGMENTATION HELPERS
# ============================================
def trim_binary_to_content(binary: np.ndarray, pad: int = 4) -> np.ndarray:
    ys, xs = np.where(binary == 0)
    if len(xs) == 0 or len(ys) == 0:
        return binary.copy()

    x1 = max(0, int(xs.min()) - pad)
    x2 = min(binary.shape[1], int(xs.max()) + pad + 1)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(binary.shape[0], int(ys.max()) + pad + 1)
    return binary[y1:y2, x1:x2]


def vertical_black_projection(binary: np.ndarray) -> np.ndarray:
    return np.sum(binary == 0, axis=0).astype(np.int32)


def smooth_projection(proj: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    kernel_size = max(3, kernel_size | 1)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(proj.astype(np.float32), kernel, mode="same")


def find_low_valley_cut_points(
    proj: np.ndarray,
    expected_parts: int = EXPECTED_TEXT_LENGTH,
    min_gap_ratio: float = 0.06,
) -> list[int]:
    """
    หา cut points จาก valley บนแนวตั้ง
    """
    width = len(proj)
    if width < expected_parts:
        return []

    smoothed = smooth_projection(proj, kernel_size=max(5, width // 25 | 1))
    max_val = float(np.max(smoothed)) if len(smoothed) else 0.0
    if max_val <= 0:
        return []

    min_distance = max(8, int(width * min_gap_ratio))
    ideal_positions = [int(width * i / expected_parts) for i in range(1, expected_parts)]

    cuts: list[int] = []
    used_ranges: list[tuple[int, int]] = []

    for ideal in ideal_positions:
        left = max(1, ideal - min_distance)
        right = min(width - 2, ideal + min_distance)
        if left >= right:
            continue

        local = smoothed[left:right + 1]
        local_idx = int(np.argmin(local))
        cut = left + local_idx

        conflict = False
        for a, b in used_ranges:
            if a <= cut <= b:
                conflict = True
                break
        if conflict:
            continue

        cuts.append(cut)
        used_ranges.append((cut - min_distance // 2, cut + min_distance // 2))

    cuts = sorted(set(cuts))
    if len(cuts) != expected_parts - 1:
        return []
    return cuts


def split_by_cut_points(binary: np.ndarray, cut_points: list[int]) -> list[np.ndarray]:
    segments = []
    start = 0
    width = binary.shape[1]

    for cut in cut_points:
        end = max(start + 1, cut)
        segments.append(binary[:, start:end])
        start = cut

    segments.append(binary[:, start:width])
    return segments


def equal_width_split(binary: np.ndarray, expected_parts: int = EXPECTED_TEXT_LENGTH) -> list[np.ndarray]:
    width = binary.shape[1]
    segments = []
    for i in range(expected_parts):
        x1 = int(round(i * width / expected_parts))
        x2 = int(round((i + 1) * width / expected_parts))
        x2 = max(x2, x1 + 1)
        segments.append(binary[:, x1:x2])
    return segments


def trim_segment(binary: np.ndarray, pad: int = 3) -> np.ndarray:
    return trim_binary_to_content(binary, pad=pad)


def ensure_segment_has_reasonable_size(binary: np.ndarray, min_w: int = 6, min_h: int = 10) -> np.ndarray:
    h, w = binary.shape[:2]
    if w >= min_w and h >= min_h:
        return binary

    target_w = max(w, min_w)
    target_h = max(h, min_h)
    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)

    y_off = (target_h - h) // 2
    x_off = (target_w - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = binary
    return canvas


def prepare_char_image_for_ocr(binary_char: np.ndarray) -> np.ndarray:
    char = trim_segment(binary_char, pad=3)
    char = ensure_segment_has_reasonable_size(char, min_w=10, min_h=16)

    h, w = char.shape[:2]
    scale = max(2.5, 48 / max(h, 1))
    char = upscale_image(char, scale=scale)

    # ทำให้ stroke เนียนขึ้นเล็กน้อย
    inv = cv2.bitwise_not(char)
    kernel = np.ones((2, 2), np.uint8)
    inv = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    char = cv2.bitwise_not(inv)

    char = add_white_border(char, border=18)
    char = normalize_binary(char)
    return char


def segment_captcha_into_4(
    binary_image: np.ndarray,
    debug_dir: Optional[Path] = None,
    file_stem: str = "captcha",
    save_debug: bool = False,
) -> list[np.ndarray]:
    """
    รับภาพขาวพื้นหลัง / ดำตัวอักษร แล้วแบ่งเป็น 4 ช่วงอัตโนมัติ
    """
    cropped = trim_binary_to_content(binary_image, pad=4)

    proj = vertical_black_projection(cropped)
    cuts = find_low_valley_cut_points(proj, expected_parts=EXPECTED_TEXT_LENGTH)

    if cuts:
        segments = split_by_cut_points(cropped, cuts)
    else:
        segments = equal_width_split(cropped, expected_parts=EXPECTED_TEXT_LENGTH)

    final_segments: list[np.ndarray] = []
    for idx, seg in enumerate(segments[:EXPECTED_TEXT_LENGTH], start=1):
        prepared = prepare_char_image_for_ocr(seg)
        final_segments.append(prepared)

        if save_debug and debug_dir is not None:
            save_image(debug_dir / f"20_segment_{idx}_{file_stem}.png", prepared)

    return final_segments


# ============================================
# OCR CONFIDENCE / SCORING
# ============================================
def extract_tesseract_confidences(data: dict) -> list[float]:
    confs: list[float] = []
    for raw in data.get("conf", []):
        try:
            value = float(raw)
            if value >= 0:
                confs.append(value)
        except (TypeError, ValueError):
            continue
    return confs


def compute_mean_confidence(confs: list[float]) -> float:
    if not confs:
        return 0.0
    return float(sum(confs) / len(confs))


def normalized_confidence(mean_conf: float) -> float:
    return clamp(mean_conf / 100.0)


def length_score(text: str, expected_len: int = EXPECTED_TEXT_LENGTH) -> float:
    if not text:
        return 0.0
    diff = abs(len(text) - expected_len)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.55
    if diff == 2:
        return 0.20
    return 0.0


def format_score(text: str) -> float:
    if not text:
        return 0.0
    return 1.0 if re.fullmatch(r"[A-Z0-9]+", text) else 0.0


def build_final_score(text: str, mean_conf: float) -> int:
    conf_part = normalized_confidence(mean_conf)
    len_part = length_score(text)
    fmt_part = format_score(text)

    score = (conf_part * 0.50) + (len_part * 0.35) + (fmt_part * 0.15)

    if len(text) != EXPECTED_TEXT_LENGTH:
        score *= 0.5

    return int(round(clamp(score) * 100))


def decide_accept_or_retry(score_pct: int, text: str) -> str:
    if len(text) != EXPECTED_TEXT_LENGTH:
        return "RETRY"
    if score_pct >= 75:
        return "ACCEPT"
    if score_pct >= 55:
        return "CHECK"
    return "RETRY"


# ============================================
# OCR ENGINES
# ============================================
def run_tesseract_single_char(image_gray: np.ndarray) -> dict:
    pil_img = Image.fromarray(image_gray)

    configs = [
        f"--oem 3 --psm 10 -c tessedit_char_whitelist={OCR_WHITELIST}",
        f"--oem 1 --psm 10 -c tessedit_char_whitelist={OCR_WHITELIST}",
        f"--oem 3 --psm 13 -c tessedit_char_whitelist={OCR_WHITELIST}",
    ]

    candidates = []

    for cfg in configs:
        data = pytesseract.image_to_data(
            pil_img,
            config=cfg,
            output_type=pytesseract.Output.DICT,
        )
        text = " ".join([str(x) for x in data.get("text", []) if str(x).strip()])
        cleaned = clean_text(text)[:1]

        confs = extract_tesseract_confidences(data)
        mean_conf = compute_mean_confidence(confs)

        if cleaned:
            candidates.append(
                {
                    "char": cleaned,
                    "conf": mean_conf,
                    "config": cfg,
                }
            )

    if not candidates:
        return {"char": "", "conf": 0.0, "config": None}

    candidates.sort(key=lambda x: (-x["conf"], x["char"]))
    return candidates[0]


def run_tesseract_segmented_4chars(
    image_path: Path,
    preset: PreprocessPreset = "white_bg_captcha",
    debug: bool = False,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    ฟังก์ชันหลักแบบใหม่:
    preprocess -> segment 4 chars -> OCR ทีละตัว
    """
    output_dir = output_dir or image_path.parent
    processed_path = preprocess_for_ocr(
        image_path=image_path,
        output_dir=output_dir,
        preset=preset,
        debug=debug,
    )

    binary = cv2.imread(str(processed_path), cv2.IMREAD_GRAYSCALE)
    if binary is None:
        raise ValueError(f"Cannot read processed image: {processed_path}")

    debug_dir = output_dir / DEBUG_OUTPUT_DIRNAME if debug else None
    file_stem = image_path.stem

    segments = segment_captcha_into_4(
        binary_image=binary,
        debug_dir=debug_dir,
        file_stem=file_stem,
        save_debug=debug,
    )

    chars = []
    per_char_results = []

    for idx, seg in enumerate(segments, start=1):
        result = run_tesseract_single_char(seg)
        chars.append(result["char"] if result["char"] else "")
        per_char_results.append(
            {
                "index": idx,
                "char": result["char"],
                "conf": result["conf"],
                "config": result["config"],
            }
        )

    prediction = "".join(chars)
    mean_conf = (
        sum(item["conf"] for item in per_char_results) / len(per_char_results)
        if per_char_results else 0.0
    )
    score_pct = build_final_score(prediction, mean_conf)
    decision = decide_accept_or_retry(score_pct, prediction)

    return {
        "prediction": prediction,
        "confidence_score": score_pct,
        "decision": decision,
        "preset": preset,
        "processed_path": str(processed_path),
        "mean_conf": mean_conf,
        "segments_count": len(segments),
        "per_char_results": per_char_results,
    }


def run_typhoon_ocr(
    image_path: Path,
    preset: PreprocessPreset = "white_bg_captcha",
    debug: bool = False,
    output_dir: Optional[Path] = None,
) -> str:
    check_typhoon_api_key()
    ocr_document = ensure_typhoon_import()

    processed_path = preprocess_for_ocr(
        image_path=image_path,
        output_dir=output_dir,
        preset=preset,
        debug=debug,
    )

    text = ocr_document(str(processed_path))
    return clean_text(text)


def run_ocr(
    image_path: Path,
    engine: OCREngine = "tesseract",
    preset: PreprocessPreset = "white_bg_captcha",
    debug: bool = False,
    output_dir: Optional[Path] = None,
) -> dict:
    if engine == "tesseract":
        return run_tesseract_segmented_4chars(
            image_path=image_path,
            preset=preset,
            debug=debug,
            output_dir=output_dir,
        )

    if engine == "typhoon":
        pred = run_typhoon_ocr(
            image_path=image_path,
            preset=preset,
            debug=debug,
            output_dir=output_dir,
        )
        score_pct = build_final_score(pred, 80.0 if len(pred) == EXPECTED_TEXT_LENGTH else 40.0)
        return {
            "prediction": pred,
            "confidence_score": score_pct,
            "decision": decide_accept_or_retry(score_pct, pred),
            "preset": preset,
            "processed_path": None,
            "mean_conf": 0.0,
            "segments_count": 0,
            "per_char_results": [],
        }

    raise ValueError(f"Unsupported engine: {engine}")


# ============================================
# DISPLAY / RUN HELPERS
# ============================================
def print_single_result(
    file_name: str,
    engine: OCREngine,
    result: dict,
) -> None:
    prediction = result.get("prediction", "")
    score = result.get("confidence_score", 0)
    decision = result.get("decision", "RETRY")

    print("\n=== SINGLE IMAGE RESULT ===")
    print(f"File         : {file_name}")
    print(f"Engine       : {engine}")
    print(f"Engine Mode  : {get_engine_mode(engine)}")
    print(f"OCR          : {prediction or '(empty)'}")
    print(f"Confidence   : {score}%")
    print(f"Decision     : {decision}")
    print(f"Preset       : {result.get('preset', '-')}")
    print(f"Mean Conf    : {result.get('mean_conf', 0.0):.2f}")
    print(f"Segments     : {result.get('segments_count', 0)}")


def print_segment_details(result: dict) -> None:
    rows = result.get("per_char_results", [])
    if not rows:
        return

    print("\n=== SEGMENT DETAILS ===")
    for row in rows:
        print(
            f"SEG={row['index']} | "
            f"CHAR={row['char'] or '(empty)'} | "
            f"CONF={row['conf']:.2f}"
        )


def test_single_image(
    image_path: str | Path,
    engine: OCREngine = "tesseract",
    preset: PreprocessPreset = "white_bg_captcha",
    debug: bool = True,
    output_dir: Optional[str | Path] = None,
    show_segment_details: bool = True,
) -> dict:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir_path = Path(output_dir) if output_dir else image_path.parent

    result = run_ocr(
        image_path=image_path,
        engine=engine,
        preset=preset,
        debug=debug,
        output_dir=output_dir_path,
    )

    final_result = {
        "file": image_path.name,
        "engine": engine,
        "engine_mode": get_engine_mode(engine),
        **result,
    }

    print_single_result(
        file_name=final_result["file"],
        engine=engine,
        result=final_result,
    )

    if show_segment_details:
        print_segment_details(final_result)

    return final_result


def run_folder_ocr(
    image_dir: str | Path,
    engine: OCREngine = "tesseract",
    preset: PreprocessPreset = "white_bg_captcha",
    debug: bool = False,
    output_dir: Optional[str | Path] = None,
    extensions: Optional[set[str]] = None,
) -> list[dict]:
    image_dir = Path(image_dir)
    extensions = extensions or DEFAULT_EXTENSIONS

    if not image_dir.exists():
        raise FileNotFoundError(f"Folder not found: {image_dir}")

    output_dir_path = Path(output_dir) if output_dir else image_dir

    print(f"\n[INFO] Engine      : {engine}")
    print(f"[INFO] Engine mode : {get_engine_mode(engine)}")
    print(f"[INFO] Folder      : {image_dir}")
    print(f"[INFO] Debug       : {debug}\n")

    results: list[dict] = []

    for img in image_dir.iterdir():
        if img.suffix.lower() not in extensions:
            continue
        if img.name.startswith("processed_"):
            continue

        result = run_ocr(
            image_path=img,
            engine=engine,
            preset=preset,
            debug=debug,
            output_dir=output_dir_path,
        )

        item = {
            "file": img.name,
            "engine": engine,
            "engine_mode": get_engine_mode(engine),
            **result,
        }
        results.append(item)

        print(
            f"{img.name} | "
            f"OCR={item['prediction'] or '(empty)'} | "
            f"CONF={item['confidence_score']}% | "
            f"DECISION={item['decision']}"
        )

    if not results:
        raise ValueError(f"No valid image files found in: {image_dir}")

    accepted = sum(1 for x in results if x["decision"] == "ACCEPT")
    check = sum(1 for x in results if x["decision"] == "CHECK")
    retry = sum(1 for x in results if x["decision"] == "RETRY")

    print("\n=== SUMMARY ===")
    print(f"Total Files        : {len(results)}")
    print(f"ACCEPT             : {accepted}")
    print(f"CHECK              : {check}")
    print(f"RETRY              : {retry}")
    print(f"Engine             : {engine}")
    print(f"Engine Mode        : {get_engine_mode(engine)}")

    return results


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    # ถ้า Windows หา tesseract ไม่เจอ ให้ปลดคอมเมนต์บรรทัดนี้
    # configure_tesseract(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    test_single_image(
        image_path="images/set1/5IHS.jpg",
        engine="tesseract",
        preset="white_bg_captcha_strong",
        debug=True,
        output_dir="images/set1",
        show_segment_details=True,
    )

    # run_folder_ocr(
    #     image_dir="images/set2",
    #     engine="tesseract",
    #     preset="white_bg_captcha_strong",
    #     debug=False,
    #     output_dir="images/set2",
    # )