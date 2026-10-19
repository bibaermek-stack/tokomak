"""Screenshots of the SINTEZ web interface (original colours) for the article.   Playwright (Python) + system Chromium.

    python tools/screenshots.py            # needs `python -m http.server 8765` in the repository root and uvicorn on 8770 for the API page
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "screenshots"
OUT.mkdir(exist_ok=True)
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
BASE = "http://127.0.0.1:8765/index.html"


with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader",
                                                         "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--enable-webgl"])
    pg = b.new_page(viewport={"width": 1200, "height": 760}, device_scale_factor=1)
    logs = []
    pg.on("console", lambda m: logs.append(m.text))
    pg.goto(BASE, wait_until="load")
    time.sleep(4)
    pg.screenshot(path=str(OUT / "ui_landing.png"), timeout=170000)
    pg.click("#btn-start")
    time.sleep(3)
    try:
        pg.click("#btn-skip", timeout=5000)
    except Exception as e:
        print("skip:", e)
    for _ in range(40):
        if pg.evaluate("document.body.dataset.stage") not in ("flight", "landing"):
            break
        time.sleep(2)
    time.sleep(12)                                    # let the simulation run: the readouts need a few seconds of history
    pg.screenshot(path=str(OUT / "ui_reactor.png"), timeout=170000)
    try:
        pg.click("#b-heat", timeout=3000)
        time.sleep(3)
        pg.screenshot(path=str(OUT / "ui_heatmap.png"), timeout=170000)
    except Exception as e:
        print("heat map:", e)
    print("stage:", pg.evaluate("document.body.dataset.stage"))
    b.close()
print("done")
