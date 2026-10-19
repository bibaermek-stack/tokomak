"""Render a figure (svg/png) to a PNG preview with headless Chromium for visual checks.   python tools/preview.py name [width]"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
name = sys.argv[1]
width = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
src = next((HERE / "figures" / f"{name}.{e}" for e in ("svg", "png") if (HERE / "figures" / f"{name}.{e}").is_file()))
html = HERE / "build" / f"preview_{name}.html"
html.write_text(f'<html><body style="margin:0;background:#fff"><img src="{src}" style="width:{width}px"></body></html>')
out = HERE / "build" / f"preview_{name}.png"
subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", f"--window-size={width + 20},{int(width * 0.75)}",
                f"--screenshot={out}", f"file://{html}"], capture_output=True)
print(out)
