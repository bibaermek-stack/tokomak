"""Screenshots of the SINTEZ HTTP API (FastAPI's Swagger UI served from local files), grayscale.   needs uvicorn on 8770."""
import json
from pathlib import Path

from PIL import Image, ImageOps
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "screenshots"
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
SW = HERE / "node_modules" / "swagger-ui-dist"
BASE = "http://127.0.0.1:8770"


def route(r):
    u = r.request.url
    if "swagger-ui-bundle.js" in u:
        r.fulfill(path=str(SW / "swagger-ui-bundle.js"), content_type="application/javascript")
    elif "swagger-ui.css" in u:
        r.fulfill(path=str(SW / "swagger-ui.css"), content_type="text/css")
    else:
        r.abort()


def operation(pg, method, path, body=None, name="api", parts=("summary", "body", "response")):
    """Open one operation, execute it, and save a compact picture: summary line, request body and server response."""
    head = pg.locator(f"span.opblock-summary-path[data-path='{path}']").first
    head.click()
    blk = pg.locator(f".opblock-{method}.is-open").first
    blk.locator(".try-out__btn").click()
    if body is not None:
        blk.locator("textarea.body-param__text").fill(json.dumps(body, indent=2))
    blk.locator(".execute").click()
    blk.locator(".responses-table.live-responses-table").wait_for(timeout=120000)
    pg.wait_for_timeout(1500)
    full = OUT / f"_{name}_full.png"
    blk.screenshot(path=str(full))
    box = blk.bounding_box()
    scale = 1.5
    crops = []

    def crop(sel, extra_top=0, extra_bottom=0):
        el = blk.locator(sel).first.bounding_box()
        y0 = max(0, int((el["y"] - box["y"] - extra_top) * scale))
        y1 = int((el["y"] - box["y"] + el["height"] + extra_bottom) * scale)
        crops.append((y0, y1))
    crop(".opblock-summary")
    if body is not None:
        crop(".body-param")
    crop(".live-responses-table .response-col_description pre", extra_top=34, extra_bottom=6)
    im = Image.open(full)
    pieces = [im.crop((0, y0, im.width, min(y1, im.height))) for y0, y1 in crops]
    h = sum(p_.height for p_ in pieces) + 12 * (len(pieces) - 1)
    out = Image.new("RGB", (im.width, h), "white")
    y = 0
    for p_ in pieces:
        out.paste(p_, (0, y))
        y += p_.height + 12
    out.save(OUT / f"{name}.png")
    full.unlink()
    head.click()


with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=1.5)
    pg.route("**/*", lambda r: route(r) if "127.0.0.1" not in r.request.url else r.continue_())
    pg.goto(f"{BASE}/docs", wait_until="load")
    pg.wait_for_selector(".opblock", timeout=60000)
    pg.add_style_tag(content="""
      *, *::before, *::after { color: #000 !important; text-shadow: none !important; }
      .swagger-ui .opblock, .swagger-ui .opblock .opblock-summary, .swagger-ui .opblock-body, .swagger-ui .opblock-section-header,
      .swagger-ui .responses-inner, .swagger-ui textarea, .swagger-ui .highlight-code, .swagger-ui .microlight, .swagger-ui pre,
      .swagger-ui .opblock-description-wrapper, .swagger-ui .opblock-body pre.microlight, .swagger-ui .responses-table {
        background: #fff !important; box-shadow: none !important; }
      .swagger-ui .opblock { border: 1px solid #000 !important; }
      .swagger-ui .opblock .opblock-summary-method { background: #fff !important; border: 1px solid #000 !important; }
      .swagger-ui .opblock-body pre.microlight, .swagger-ui .highlight-code > pre { border: 0.8px solid #000 !important; }
      .swagger-ui .btn { background: #fff !important; border-color: #000 !important; }
      .swagger-ui .btn.execute { background: #fff !important; border: 1.5px solid #000 !important; }
      svg, svg path { fill: #000 !important; }
    """)
    operation(pg, "post", "/ml/shrx", {"Ati": 7.0, "Ate": 7.0, "Ane": 2.0, "q": 1.5, "smag": 1.0, "x": 0.5, "Ti_Te": 1.0,
                                       "LogNuStar": -1.5, "normni": 0.9, "Ti_keV": 8.0, "B0": 5.3, "R0": 6.2, "a": 2.0, "a_mass": 2.5},
              name="api_shrx")
    operation(pg, "get", "/models", name="api_models")
    b.close()
for n in ("api_shrx", "api_models"):
    im = Image.open(OUT / f"{n}.png").convert("L")
    im.save(OUT / f"{n}.png", optimize=True)
print("done")
