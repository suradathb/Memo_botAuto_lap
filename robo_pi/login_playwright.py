from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError, sync_playwright


BASE_URL = "https://devaion.com7tracking.com"
USERNAME = "GI10038"
PASSWORD = "GI10038"

SCREENSHOT_DIR = Path("playwright_debug")
SCREENSHOT_DIR.mkdir(exist_ok=True)


def save_debug(page: Page, name: str) -> None:
    """Save screenshot and html for debugging."""
    timestamp = int(time.time())
    png_path = SCREENSHOT_DIR / f"{timestamp}_{name}.png"
    html_path = SCREENSHOT_DIR / f"{timestamp}_{name}.html"

    page.screenshot(path=str(png_path), full_page=True)
    html_path.write_text(page.content(), encoding="utf-8")

    print(f"[DEBUG] Screenshot: {png_path}")
    print(f"[DEBUG] HTML: {html_path}")


def wait_for_page_ready(page: Page) -> None:
    page.wait_for_load_state("domcontentloaded")
    page.wait_for_load_state("networkidle")


def open_home(page: Page) -> None:
    print("[STEP] Open home page")
    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
    wait_for_page_ready(page)


def open_login_modal(page: Page) -> None:
    print("[STEP] Open login modal")

    login_nav = page.locator("#dropdownNavbarLink", has_text="เข้าสู่ระบบ")
    login_nav.wait_for(state="visible", timeout=20000)
    login_nav.click()

    page.locator("#login-username").wait_for(state="visible", timeout=20000)
    page.locator("#login-password").wait_for(state="visible", timeout=20000)

    print("[INFO] Login modal opened")


def fill_login_form(page: Page, username: str, password: str) -> None:
    print("[STEP] Fill login form")

    username_input = page.locator("#login-username")
    password_input = page.locator("#login-password")

    username_input.click()
    username_input.fill("")
    username_input.type(username, delay=80)

    password_input.click()
    password_input.fill("")
    password_input.type(password, delay=80)

    print("[INFO] Username and password filled")


def get_modal_submit_button(page: Page):
    """
    Get submit button inside the same modal/form section as password input.
    This avoids confusing it with the navbar button that has the same text.
    """
    return page.locator(
        "xpath=//input[@id='login-password']"
        "/ancestor::div[contains(@class,'space-y-6')]"
        "//button[normalize-space()='เข้าสู่ระบบ']"
    )


def submit_login(page: Page) -> None:
    print("[STEP] Submit login")

    submit_btn = get_modal_submit_button(page)
    submit_btn.wait_for(state="visible", timeout=20000)

    is_disabled = submit_btn.is_disabled()
    print(f"[INFO] Submit disabled before click: {is_disabled}")

    if is_disabled:
        page.locator("#login-password").press("Tab")
        page.wait_for_timeout(1000)
        is_disabled = submit_btn.is_disabled()
        print(f"[INFO] Submit disabled after TAB: {is_disabled}")

    if is_disabled:
        raise RuntimeError("Submit button is still disabled before login click.")

    submit_btn.click()
    page.wait_for_timeout(3000)


def wait_for_login_success(page: Page) -> None:
    print("[STEP] Wait for login success on same page")

    deadline = time.time() + 30
    last_state = None

    while time.time() < deadline:
        current_url = page.url

        stock_visible = page.locator("a[href='/stock']").count() > 0
        user_btn_changed = page.locator(
            "#dropdownNavbarLink:not(:has-text('เข้าสู่ระบบ'))"
        ).count() > 0

        last_state = {
            "url": current_url,
            "stock_visible": stock_visible,
            "user_btn_changed": user_btn_changed,
        }

        print(
            f"[INFO] url={current_url} | "
            f"stock_visible={stock_visible} | "
            f"user_btn_changed={user_btn_changed}"
        )

        if stock_visible or user_btn_changed:
            print("[PASS] Login success detected")
            return

        page.wait_for_timeout(2000)

    raise RuntimeError(f"Login success state not reached. Last state: {last_state}")


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            slow_mo=200,
            args=[
                "--disable-blink-features=AutomationControlled",
            ],
        )

        context = browser.new_context(
            viewport={"width": 1600, "height": 900},
            ignore_https_errors=True,
        )

        page = context.new_page()

        try:
            open_home(page)
            save_debug(page, "01_home")

            open_login_modal(page)
            save_debug(page, "02_modal_opened")

            fill_login_form(page, USERNAME, PASSWORD)
            save_debug(page, "03_filled")

            submit_login(page)
            save_debug(page, "04_after_submit")

            wait_for_login_success(page)
            save_debug(page, "05_login_success")

            print("[DONE] Login test passed")
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