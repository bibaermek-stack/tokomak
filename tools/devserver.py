"""Static file server for the tokamak simulation, plus a small capture
endpoint so frames can be pulled out of the running page for inspection.

    POST /__save?name=foo   body = raw image bytes  ->  _capture/foo.jpg
"""
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURE_DIR = os.path.join(ROOT, "_capture")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/__save":
            self.send_error(404)
            return
        name = parse_qs(parsed.query).get("name", ["frame"])[0]
        name = "".join(c for c in name if c.isalnum() or c in "-_") or "frame"
        length = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(length)
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        path = os.path.join(CAPTURE_DIR, name + ".jpg")
        with open(path, "wb") as fh:
            fh.write(data)
        body = ("saved %s (%d bytes)" % (path, len(data))).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # never cache during development
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s\n" % (fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
