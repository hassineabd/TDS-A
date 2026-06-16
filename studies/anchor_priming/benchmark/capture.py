"""Capture mobile screenshots + accessibility tree via BrowserStack App Automate.

Usage:
    python -m benchmark.capture --app com.android.settings --out data/settings_pixel7
    python -m benchmark.capture --app com.google.android.deskclock --out data/clock_pixel7

Requires BROWSERSTACK_USERNAME and BROWSERSTACK_ACCESS_KEY env vars.
We upload a lightweight sample app once (Wikipedia) so the session starts,
then `activate_app` to switch to the requested pre-installed app.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

HUB = "https://hub-cloud.browserstack.com/wd/hub"
UPLOAD_API = "https://api-cloud.browserstack.com/app-automate/upload"
# Public BrowserStack sample app (always uploadable)
SAMPLE_APP_URL = "https://www.browserstack.com/app-automate/sample-apps/android/WikipediaSample.apk"


def _auth() -> tuple[str, str]:
    load_dotenv()
    u = os.getenv("BROWSERSTACK_USERNAME")
    k = os.getenv("BROWSERSTACK_ACCESS_KEY")
    if not u or not k:
        print("ERROR: set BROWSERSTACK_USERNAME and BROWSERSTACK_ACCESS_KEY in .env",
              file=sys.stderr)
        sys.exit(1)
    return u, k


def upload_sample_app() -> str:
    """Upload the Wikipedia sample APK to BrowserStack and return the bs:// URL."""
    user, key = _auth()
    r = requests.post(UPLOAD_API, auth=(user, key),
                      data={"url": SAMPLE_APP_URL}, timeout=60)
    r.raise_for_status()
    app_url = r.json().get("app_url")
    if not app_url:
        raise RuntimeError(f"Upload failed: {r.json()}")
    return app_url


def create_session(app_url: str, device: str, os_version: str,
                   project: str = "TDS-A Mobile Grounding Benchmark") -> str:
    """Create an Appium session on BrowserStack. Returns session ID."""
    user, key = _auth()
    caps = {
        "capabilities": {
            "alwaysMatch": {
                "platformName": "Android",
                "appium:app": app_url,
                "appium:automationName": "UiAutomator2",
                "appium:noReset": True,
                "bstack:options": {
                    "projectName": project,
                    "buildName": "Grounding Capture",
                    "sessionName": f"Capture on {device}",
                    "deviceName": device,
                    "osVersion": os_version,
                },
            },
            "firstMatch": [{}],
        }
    }
    r = requests.post(f"{HUB}/session", auth=(user, key),
                      json=caps, timeout=120)
    r.raise_for_status()
    body = r.json()
    sid = body.get("value", {}).get("sessionId") or body.get("sessionId")
    if not sid:
        raise RuntimeError(f"Session creation failed: {body}")
    return sid


def activate_app(session_id: str, bundle_id: str) -> None:
    """Activate an already-installed app. Tries multiple payload keys for
    compatibility across Appium versions (W3C `appId` vs legacy `bundleId`)."""
    user, key = _auth()
    last_err: Exception | None = None
    for payload in ({"appId": bundle_id}, {"bundleId": bundle_id},
                    {"appId": bundle_id, "bundleId": bundle_id}):
        try:
            r = requests.post(
                f"{HUB}/session/{session_id}/appium/device/activate_app",
                auth=(user, key),
                json=payload,
                timeout=30,
            )
            if r.ok:
                return
            last_err = RuntimeError(f"{r.status_code} {r.text[:300]}")
        except Exception as exc:
            last_err = exc
    # Fallback: use mobile: shell to start the activity via am
    try:
        r = requests.post(
            f"{HUB}/session/{session_id}/execute/sync",
            auth=(user, key),
            json={"script": "mobile: shell",
                  "args": [{"command": "monkey",
                            "args": ["-p", bundle_id, "-c",
                                     "android.intent.category.LAUNCHER", "1"]}]},
            timeout=30,
        )
        if r.ok:
            return
        last_err = RuntimeError(f"shell fallback: {r.status_code} {r.text[:300]}")
    except Exception as exc:
        last_err = exc
    raise RuntimeError(f"activate_app failed: {last_err}")


def screenshot(session_id: str) -> bytes:
    user, key = _auth()
    r = requests.get(f"{HUB}/session/{session_id}/screenshot",
                     auth=(user, key), timeout=30)
    r.raise_for_status()
    return base64.b64decode(r.json()["value"])


def page_source(session_id: str) -> str:
    user, key = _auth()
    r = requests.get(f"{HUB}/session/{session_id}/source",
                     auth=(user, key), timeout=30)
    r.raise_for_status()
    return r.json()["value"]


def delete_session(session_id: str) -> None:
    user, key = _auth()
    try:
        requests.delete(f"{HUB}/session/{session_id}",
                        auth=(user, key), timeout=30)
    except Exception:
        pass


def capture(bundle_id: str, out_prefix: Path, device: str, os_version: str,
            wait_seconds: float = 3.0) -> None:
    """Run full capture flow: upload -> session -> activate -> screenshot + source."""
    print(f"[1/5] Uploading sample app to BrowserStack...")
    app_url = upload_sample_app()
    print(f"      app_url: {app_url}")

    print(f"[2/5] Creating session on {device} (Android {os_version})...")
    sid = create_session(app_url, device, os_version)
    print(f"      session: {sid}")

    try:
        print(f"[3/5] Activating app: {bundle_id}")
        activate_app(sid, bundle_id)
        time.sleep(wait_seconds)  # let the app settle

        print(f"[4/5] Capturing screenshot + page source...")
        img = screenshot(sid)
        src = page_source(sid)

        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        png_path = out_prefix.with_suffix(".png")
        xml_path = out_prefix.with_suffix(".xml")
        png_path.write_bytes(img)
        xml_path.write_text(src)
        print(f"      -> {png_path} ({len(img)} bytes)")
        print(f"      -> {xml_path} ({len(src)} chars)")

        print(f"[5/5] Cleaning up session...")
    finally:
        delete_session(sid)

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True,
                        help="Android bundle id, e.g. com.android.settings")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output prefix (without extension), e.g. data/settings_pixel7")
    parser.add_argument("--device", default="Google Pixel 7")
    parser.add_argument("--os-version", default="13.0")
    parser.add_argument("--wait", type=float, default=3.0)
    args = parser.parse_args()

    capture(args.app, args.out, args.device, args.os_version, args.wait)


if __name__ == "__main__":
    main()
