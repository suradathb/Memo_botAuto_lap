from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

from playwright.sync_api import (
    Page,
    Locator,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

BASE_URL = "https://globaldms.aionauto.com/login"
USERNAME = "giit_maha"
PASSWORD = "Gold@rama311"

HEADLESS = False
SLOW_MO_MS = 200
MANUAL_CAPTCHA_TIMEOUT_SEC = 180

DEBUG_DIR = Path("playwright_debug")
DEBUG_DIR.mkdir(exist_ok=True)


def save_debug(page: Page, name: str) -> None:
    timestamp = int(time.time())
    png_path = DEBUG_DIR / f"{timestamp}_{name}.png"
    html_path = DEBUG_DIR / f"{timestamp}_{name}.html"

    try:
        page.screenshot(path=str(png_path), full_page=True)
    except Exception as e:
        print(f"[WARN] screenshot failed: {e}")

    try:
        html_path.write_text(page.content(), encoding="utf-8")
    except Exception as e:
        print(f"[WARN] save html failed: {e}")

    print(f"[DEBUG] Screenshot: {png_path}")
    print(f"[DEBUG] HTML: {html_path}")


def wait_for_page_ready(page: Page) -> None:
    page.wait_for_load_state("domcontentloaded")
    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except PlaywrightTimeoutError:
        print("[WARN] networkidle timeout, continue")


def click_when_ready(locator: Locator, timeout: int = 20000) -> None:
    locator.wait_for(state="visible", timeout=timeout)
    locator.scroll_into_view_if_needed()
    locator.click()


def open_login_page(page: Page) -> None:
    print("[STEP] Open login page")
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
    wait_for_page_ready(page)


def first_visible_locator(page: Page, selectors: list[str]) -> Locator:
    for selector in selectors:
        locator = page.locator(selector).first
        try:
            if locator.count() > 0 and locator.is_visible():
                print(f"[INFO] Found visible selector: {selector}")
                return locator
        except Exception:
            continue
    raise RuntimeError(f"No visible element found from selectors: {selectors}")


def get_username_input(page: Page) -> Locator:
    return first_visible_locator(
        page,
        [
            "input[name='username']",
            "input[id='username']",
            "input[placeholder*='User']",
            "input[placeholder*='user']",
            "input[placeholder*='Email']",
            "input[type='text']",
        ],
    )


def get_password_input(page: Page) -> Locator:
    return first_visible_locator(
        page,
        [
            "input[name='password']",
            "input[id='password']",
            "input[type='password']",
        ],
    )


def fill_login_form(page: Page, username: str, password: str) -> None:
    print("[STEP] Fill username/password")

    username_input = get_username_input(page)
    password_input = get_password_input(page)

    username_input.click()
    username_input.fill("")
    username_input.type(username, delay=60)

    password_input.click()
    password_input.fill("")
    password_input.type(password, delay=60)

    print("[INFO] Username/password filled")


def is_login_success(page: Page) -> bool:
    current_url = page.url.lower()

    if "/login" not in current_url:
        return True

    try:
        if page.locator("input[type='password']").count() == 0:
            return True
    except Exception:
        pass

    success_selectors = [
        "text=Dashboard",
        "text=Home",
        "text=Logout",
        "text=Sign out",
        "text=语言",
        "text=Language",
        "text=销售管理",
        "text=การจัดการการขาย",
        "[href*='dashboard']",
        "[class*='sidebar']",
        "[class*='menu']",
    ]

    for selector in success_selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0 and locator.is_visible():
                return True
        except Exception:
            continue

    return False


def wait_for_human_captcha_and_login(page: Page, timeout_sec: int = MANUAL_CAPTCHA_TIMEOUT_SEC) -> None:
    print("[STEP] Waiting for human CAPTCHA input")
    print("[ACTION] กรุณา:")
    print("         1) กรอก CAPTCHA")
    print("         2) กด Login บนหน้าเว็บ")
    print("         3) รอให้เข้าใช้งานสำเร็จ")

    deadline = time.time() + timeout_sec
    last_url: Optional[str] = None

    while time.time() < deadline:
        try:
            last_url = page.url
            if is_login_success(page):
                print("[PASS] Login success detected")
                return
        except Exception as e:
            print(f"[WARN] while waiting manual step: {e}")

        page.wait_for_timeout(2000)

    raise RuntimeError(f"Timeout waiting for human CAPTCHA/login. Last URL: {last_url}")


def click_text_if_exists(page: Page, texts: list[str], timeout: int = 15000) -> bool:
    deadline = time.time() + (timeout / 1000)

    while time.time() < deadline:
        for text in texts:
            try:
                locator = page.get_by_text(text, exact=True).first
                if locator.count() > 0 and locator.is_visible():
                    locator.scroll_into_view_if_needed()
                    locator.click()
                    print(f"[INFO] Clicked text: {text}")
                    return True
            except Exception:
                pass

            try:
                locator = page.locator(f"text={text}").first
                if locator.count() > 0 and locator.is_visible():
                    locator.scroll_into_view_if_needed()
                    locator.click()
                    print(f"[INFO] Clicked text: {text}")
                    return True
            except Exception:
                pass

        page.wait_for_timeout(500)

    return False


def switch_language_to_thai(page):
    print("[STEP] Switch language to Thai")

    lang_btn = page.locator(
        "div.translations-box.el-dropdown.is-dark span.icon-outer.el-dropdown-selfdefine"
    ).first
    lang_btn.wait_for(state="visible", timeout=20000)
    lang_btn.scroll_into_view_if_needed()
    lang_btn.click()

    page.wait_for_timeout(1000)

    thai_option = page.locator(
        "li.el-dropdown-menu__item:has-text('ภาษาไทย')"
    ).first
    thai_option.wait_for(state="visible", timeout=10000)
    thai_option.click()

    page.wait_for_timeout(2000)
    print("[INFO] Switched to Thai")


def open_sales_management(page):
    print("[STEP] Open menu: การจัดการการขาย")

    menu = page.locator("text=การจัดการการขาย").first
    menu.wait_for(state="visible", timeout=20000)
    menu.click()

    page.wait_for_timeout(1000)


def open_vehicle_stock_group(page):
    print("[STEP] Open group: สต็อครถยนต์")

    group_menu = page.locator("text=สต็อครถยนต์").first
    group_menu.wait_for(state="visible", timeout=20000)
    group_menu.click()

    page.wait_for_timeout(1000)

def open_vehicle_stock_check_page(page):
    print("[STEP] Open page: เช็คสต็อครถยนต์")

    target = page.locator("text=เช็คสต็อครถยนต์").first
    target.wait_for(state="visible", timeout=20000)
    target.click()

    page.wait_for_timeout(3000)

def wait_vehicle_stock_form(page):
    print("[STEP] Wait vehicle stock form")

    dropdown = page.locator("input[placeholder='โปรดเลือก']").first
    dropdown.wait_for(state="visible", timeout=30000)

    print("[INFO] Vehicle stock form loaded")


def select_warehouse_first_option(page):
    print("[STEP] Select first warehouse option")

    # จับเฉพาะ field 'ชื่อโกดัง'
    warehouse_field = page.locator(
        "xpath=//label[normalize-space()='ชื่อโกดัง']/following-sibling::div"
        "//div[contains(@class,'el-select')]"
    ).first

    warehouse_field.wait_for(state="visible", timeout=20000)
    warehouse_field.click()

    page.wait_for_timeout(1000)

    # dropdown ของ Element UI จะโผล่แยกออกมา ให้จับตัวที่ visible เท่านั้น
    visible_dropdown = page.locator(
        "div.el-select-dropdown.el-popper:visible"
    ).last

    visible_dropdown.wait_for(state="visible", timeout=10000)

    first_option = visible_dropdown.locator(
        "li.el-select-dropdown__item:not(.is-disabled)"
    ).first
    first_option.wait_for(state="visible", timeout=10000)
    first_option.click()

    page.wait_for_timeout(1000)
    print("[INFO] First warehouse option selected")


def click_search_button(page):
    print("[STEP] Click button: สอบถาม")

    btn = page.locator("button:has-text('สอบถาม')").first
    btn.wait_for(state="visible", timeout=20000)
    btn.click()

    page.wait_for_timeout(3000)


def wait_for_query_result(page):
    print("[STEP] Wait for result data")

    export_btn = page.locator("button:has-text('ส่งออก')").first
    export_btn.wait_for(state="visible", timeout=30000)

    print("[INFO] Result loaded")


def click_export_and_wait_download(page):
    print("[STEP] Click Export and wait download")

    # หา element ปุ่ม "ส่งออก"
    export_btn = page.locator("text=ส่งออก").first
    export_btn.wait_for(state="visible", timeout=20000)

    # 🔥 ตัวสำคัญ
    with page.expect_download(timeout=120000) as download_info:
        export_btn.click()

    download = download_info.value

    file_path = DEBUG_DIR / download.suggested_filename
    download.save_as(str(file_path))

    print(f"[SUCCESS] Downloaded: {file_path}")


def do_post_login_work(page):
    print("[STEP] Start post-login workflow")

    switch_language_to_thai(page)
    open_sales_management(page)
    open_vehicle_stock_group(page)
    open_vehicle_stock_check_page(page)
    wait_vehicle_stock_form(page)
    select_warehouse_first_option(page)
    click_search_button(page)
    wait_for_query_result(page)
    click_export_and_wait_download(page)

    print("[DONE] Post-login workflow completed")


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=HEADLESS,
            slow_mo=SLOW_MO_MS,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--start-maximized",
            ],
        )

        context = browser.new_context(
            viewport={"width": 1600, "height": 900},
            ignore_https_errors=True,
            accept_downloads=True,
        )

        page = context.new_page()

        try:
            open_login_page(page)
            save_debug(page, "01_opened_login")

            fill_login_form(page, USERNAME, PASSWORD)
            save_debug(page, "02_filled_credentials")

            wait_for_human_captcha_and_login(page)
            save_debug(page, "03_login_success")

            do_post_login_work(page)

            print("[DONE] All flow completed successfully")
            page.wait_for_timeout(5000)
            return 0

        except PlaywrightTimeoutError as e:
            print(f"[ERROR] Timeout: {e}")
            save_debug(page, "timeout_error")
            return 1

        except Exception as e:
            print(f"[ERROR] {e}")
            save_debug(page, "general_error")
            return 1

        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    sys.exit(main())