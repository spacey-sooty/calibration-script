#!/usr/bin/env python3
"""
Live web viewer for a V4L2 camera with sliders for its controls.

  ./camera_viewer.py --device /dev/video2 --size 1600x1304
  then open http://localhost:8080

* The stream is the camera's own MJPEG frames passed straight through by
  ffmpeg (no re-encoding), served as multipart/x-mixed-replace.
* Controls are read with `v4l2-ctl -L` and set with `v4l2-ctl --set-ctrl`;
  int -> slider, bool -> checkbox, menu -> dropdown. Inactive controls (e.g.
  exposure_time_absolute while auto exposure is on) are greyed out.
* The page shows a ready-to-paste --v4l2-ctl argument for calibrate.py with
  every control you've changed from its default.

Control changes persist on the camera after this exits (until it's unplugged).
Only one program can capture from the camera at a time, so stop this before
running calibrate.py on the same device.

The building blocks here (ControlsAPI, iter_jpegs, the controls UI snippets,
stream_mjpeg) are also used by calibration_viewer.py.

Requirements: ffmpeg, v4l2-ctl (sudo apt install ffmpeg v4l-utils)
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# --------------------------------------------------------------------------
# V4L2 controls
# --------------------------------------------------------------------------
CTRL_RE = re.compile(r"^\s*(\w+)\s+0x[0-9a-fA-F]+\s+\((\w+)\)\s*:\s*(.*)$")
MENU_RE = re.compile(r"^\s+(\d+):\s*(.*)$")
KV_RE = re.compile(r"(\w+)=(\S+)")
# Controls that switch other controls between auto/manual go first in the
# generated --v4l2-ctl string, so e.g. manual exposure is enabled before the
# exposure time is set.
AUTO_FIRST = ("auto_exposure", "white_balance_automatic", "focus_automatic_continuous",
              "exposure_dynamic_framerate")


def list_controls(device):
    out = subprocess.run(["v4l2-ctl", "-d", device, "-L"],
                         capture_output=True, text=True, check=True).stdout
    controls, current = [], None
    for line in out.splitlines():
        m = CTRL_RE.match(line)
        if m:
            name, ctype, rest = m.groups()
            kv = dict(KV_RE.findall(rest))
            flags = kv.get("flags", "").split(",") if "flags" in kv else []
            current = {
                "name": name,
                "type": ctype,
                "value": int(kv["value"]) if "value" in kv else None,
                "default": int(kv["default"]) if "default" in kv else None,
                "min": int(kv.get("min", 0)),
                "max": int(kv.get("max", 1)),
                "step": int(kv.get("step", 1)),
                "inactive": "inactive" in flags,
                "readonly": "read-only" in flags,
                "menu": [],
            }
            # Only control types we can render and set with a plain integer
            if ctype in ("int", "bool", "menu", "intmenu"):
                controls.append(current)
            continue
        m = MENU_RE.match(line)
        if m and current is not None and current["type"] in ("menu", "intmenu"):
            current["menu"].append({"value": int(m.group(1)), "label": m.group(2)})
    return controls


def set_control(device, name, value, known):
    if name not in known:
        raise ValueError(f"unknown control {name!r}")
    value = int(value)
    r = subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl", f"{name}={value}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or f"v4l2-ctl exited {r.returncode}")


def calibrate_arg(controls):
    changed = [c for c in controls
               if not c["readonly"] and not c["inactive"]
               and c["value"] is not None and c["value"] != c["default"]]
    changed.sort(key=lambda c: (c["name"] not in AUTO_FIRST, c["name"]))
    return ",".join(f"{c['name']}={c['value']}" for c in changed)


class ControlsAPI:
    """GET /api/controls, POST /api/control and POST /api/reset."""

    def __init__(self, device):
        self.device = device
        self.lock = threading.Lock()  # serialize v4l2-ctl calls

    def _state(self):
        controls = list_controls(self.device)
        return {"controls": controls, "calibrate_arg": calibrate_arg(controls)}

    def get(self):
        with self.lock:
            return self._state()

    def post(self, path, body):
        """Returns the new state, or None if path isn't a controls endpoint."""
        if path not in ("/api/control", "/api/reset"):
            return None
        with self.lock:
            controls = list_controls(self.device)
            known = {c["name"]: c for c in controls}
            if path == "/api/control":
                set_control(self.device, str(body["name"]), body["value"], known)
            else:
                # auto_* controls first so dependent controls become writable
                for c in sorted(controls, key=lambda c: (c["name"] not in AUTO_FIRST, c["name"])):
                    if c["readonly"] or c["default"] is None or c["value"] == c["default"]:
                        continue
                    try:
                        set_control(self.device, c["name"], c["default"], known)
                    except RuntimeError:
                        pass  # e.g. inactive control; skip
            return self._state()


# --------------------------------------------------------------------------
# MJPEG helpers
# --------------------------------------------------------------------------
def ffmpeg_input_args(args):
    return ["-f", "v4l2", "-input_format", args.input_format,
            "-video_size", args.size, "-framerate", str(args.framerate),
            "-i", args.device]


def iter_jpegs(stream):
    """Split a concatenated MJPEG byte stream into individual JPEG frames."""
    buf = bytearray()
    while True:
        chunk = stream.read(1 << 16)
        if not chunk:
            return
        buf += chunk
        while True:
            start = buf.find(b"\xff\xd8")
            if start < 0:
                buf.clear()
                break
            end = buf.find(b"\xff\xd9", start + 2)
            if end < 0:
                if start:
                    del buf[:start]
                break
            yield bytes(buf[start:end + 2])
            del buf[:end + 2]


class FrameBuffer:
    """Latest frame + condition variable so readers can wait for a new one."""

    def __init__(self):
        self.frame = None
        self.frame_id = 0
        self.cond = threading.Condition()
        self.closed = False

    def put(self, frame):
        with self.cond:
            self.frame = frame
            self.frame_id += 1
            self.cond.notify_all()

    def wait(self, last_id, timeout=5.0):
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id != last_id or self.closed, timeout=timeout)
            return self.frame_id, self.frame

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()


class FpsCounter:
    def __init__(self):
        self.fps, self._t0, self._n = 0.0, time.monotonic(), 0

    def tick(self):
        self._n += 1
        now = time.monotonic()
        if now - self._t0 >= 1.0:
            self.fps, self._t0, self._n = self._n / (now - self._t0), now, 0


def stream_mjpeg(handler, frames: FrameBuffer, max_fps):
    """Serve frames as multipart/x-mixed-replace until the client goes away."""
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True
    min_dt = 1.0 / max_fps if max_fps > 0 else 0
    last_id, last_t = -1, 0.0
    try:
        while not frames.closed:
            fid, frame = frames.wait(last_id)
            if frame is None or fid == last_id:
                continue
            now = time.monotonic()
            last_id = fid
            if now - last_t < min_dt:
                continue
            last_t = now
            handler.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                b"Content-Length: " + str(len(frame)).encode() +
                                b"\r\n\r\n" + frame + b"\r\n")
    except (BrokenPipeError, ConnectionResetError):
        pass


class JSONHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    verbose = False

    def log_message(self, fmt, *a):
        if self.verbose:
            super().log_message(fmt, *a)

    def send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, body, ctype):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > 10_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def handle_controls_post(self, controls: ControlsAPI, path):
        """Returns True if the request was a controls request (and was answered)."""
        try:
            state = controls.post(path, self.read_json())
        except (KeyError, ValueError, TypeError) as e:
            self.send_json({"error": f"bad request: {e}"}, HTTPStatus.BAD_REQUEST)
            return True
        except RuntimeError as e:
            self.send_json({"error": str(e)}, HTTPStatus.UNPROCESSABLE_ENTITY)
            return True
        if state is None:
            return False
        self.send_json(state)
        return True


# --------------------------------------------------------------------------
# Shared page pieces
# --------------------------------------------------------------------------
BASE_CSS = r"""
:root { --bg:#111418; --panel:#1b2027; --fg:#e6e8eb; --muted:#8a939e; --accent:#4c9aff; --border:#2c333d;
        --good:#3fb950; --warn:#d29922; --bad:#ff7b72; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }
header { padding:10px 16px; border-bottom:1px solid var(--border); display:flex; gap:16px; align-items:baseline; flex-wrap:wrap; }
header h1 { font-size:16px; margin:0; }
header span { color:var(--muted); font-variant-numeric: tabular-nums; }
main { display:flex; gap:16px; padding:16px; flex-wrap:wrap; }
#view { flex:1 1 640px; min-width:0; }
#view img { width:100%; height:auto; display:block; background:#000; border:1px solid var(--border); border-radius:6px; }
#err { color:var(--bad); margin-top:8px; white-space:pre-wrap; }
aside { flex:0 1 360px; display:flex; flex-direction:column; gap:16px; align-self:flex-start; width:100%; }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:6px; padding:12px; }
.ctrl { padding:8px 0; border-bottom:1px solid var(--border); }
.ctrl:last-of-type { border-bottom:none; }
.ctrl.inactive { opacity:.45; }
.row { display:flex; justify-content:space-between; align-items:center; gap:8px; }
.name { font-weight:600; }
input[type=range] { width:100%; accent-color:var(--accent); }
input[type=number] { width:7ch; background:var(--bg); color:var(--fg); border:1px solid var(--border); border-radius:4px; padding:2px 4px; }
select { background:var(--bg); color:var(--fg); border:1px solid var(--border); border-radius:4px; padding:2px 4px; }
button { background:var(--bg); color:var(--fg); border:1px solid var(--border); border-radius:4px; padding:3px 8px; cursor:pointer; }
button:hover { border-color:var(--accent); }
button:disabled { opacity:.5; cursor:default; }
.reset { font-size:12px; padding:1px 6px; }
#argbox { margin-top:12px; }
#argbox code { display:block; background:var(--bg); padding:8px; border-radius:4px; word-break:break-all; font-size:12px; margin:6px 0; min-height:2em; }
.hint { color:var(--muted); font-size:12px; }
"""

CONTROLS_HTML = r"""
<div class="panel">
  <div class="row"><strong>Camera controls</strong>
    <span><button id="resetall">Reset all</button> <button id="refresh">Refresh</button></span></div>
  <div id="controls"></div>
  <div id="argbox"><div class="hint">For calibrate.py (changed from default):</div>
    <code id="arg"></code><button id="copy">Copy</button></div>
</div>
"""

CONTROLS_JS = r"""
const $ = s => document.querySelector(s);
const pending = {};

async function api(path, body) {
  const r = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function renderControls(state) {
  const box = $('#controls');
  // Don't overwrite a control while it is being dragged/edited
  const active = document.activeElement && document.activeElement.dataset.name;
  for (const c of state.controls) {
    let el = box.querySelector(`[data-ctrl="${c.name}"]`);
    if (!el) { el = buildControl(c); box.appendChild(el); }
    el.classList.toggle('inactive', c.inactive);
    if (c.name === active) continue;
    const input = el.querySelector('[data-name]');
    if (c.type === 'bool') input.checked = !!c.value;
    else input.value = c.value;
    const num = el.querySelector('input[type=number]'); if (num) num.value = c.value;
  }
  $('#arg').textContent = state.calibrate_arg ? `--v4l2-ctl ${state.calibrate_arg}` : '(all defaults)';
}

function buildControl(c) {
  const el = document.createElement('div');
  el.className = 'ctrl'; el.dataset.ctrl = c.name;
  const head = document.createElement('div'); head.className = 'row';
  const label = document.createElement('span'); label.className = 'name'; label.textContent = c.name;
  const reset = document.createElement('button'); reset.className = 'reset'; reset.textContent = 'default';
  reset.title = `default: ${c.default}`;
  reset.onclick = () => sendControl(c.name, c.default);
  head.append(label);
  let input;
  if (c.type === 'bool') {
    input = document.createElement('input'); input.type = 'checkbox';
    input.onchange = () => sendControl(c.name, input.checked ? 1 : 0);
    head.append(input, reset); el.append(head);
  } else if (c.type === 'menu' || c.type === 'intmenu') {
    input = document.createElement('select');
    for (const m of c.menu) { const o = document.createElement('option'); o.value = m.value; o.textContent = `${m.value}: ${m.label}`; input.append(o); }
    input.onchange = () => sendControl(c.name, +input.value);
    head.append(input, reset); el.append(head);
  } else {
    const num = document.createElement('input'); num.type = 'number';
    num.min = c.min; num.max = c.max; num.step = c.step;
    num.onchange = () => { input.value = num.value; sendControl(c.name, +num.value); };
    head.append(num, reset);
    input = document.createElement('input'); input.type = 'range';
    input.min = c.min; input.max = c.max; input.step = c.step;
    input.oninput = () => { num.value = input.value; throttled(c.name, +input.value); };
    input.onchange = () => sendControl(c.name, +input.value);
    el.append(head, input);
    const lim = document.createElement('div'); lim.className = 'row hint';
    lim.innerHTML = `<span>${c.min}</span><span>${c.max}</span>`; el.append(lim);
  }
  input.dataset.name = c.name;
  if (c.readonly) input.disabled = true;
  return el;
}

// Rate-limit slider drags to ~10 updates/s per control
const timers = {};
function throttled(name, value) {
  pending[name] = value;
  if (timers[name]) return;
  timers[name] = setTimeout(() => { delete timers[name]; sendControl(name, pending[name]); }, 100);
}

async function sendControl(name, value) {
  try { renderControls(await api('/api/control', {name, value})); $('#err').textContent = ''; }
  catch (e) { $('#err').textContent = `${name}: ${e.message}`; refreshControls(); }
}

async function refreshControls() {
  try { renderControls(await api('/api/controls')); } catch (e) { $('#err').textContent = e.message; }
}

$('#refresh').onclick = refreshControls;
$('#resetall').onclick = async () => {
  try { renderControls(await api('/api/reset', {})); } catch (e) { $('#err').textContent = e.message; }
};
$('#copy').onclick = () => navigator.clipboard.writeText($('#arg').textContent).catch(() => {});

// Reconnect the stream if it drops (e.g. camera unplugged and replugged)
$('#stream').onerror = () => setTimeout(() => { $('#stream').src = '/stream?' + Date.now(); }, 1000);
refreshControls();
"""


# --------------------------------------------------------------------------
# Plain viewer
# --------------------------------------------------------------------------
class Camera:
    """Runs ffmpeg in a loop (restarting on errors) and keeps the latest JPEG."""

    def __init__(self, args):
        self.args = args
        self.frames = FrameBuffer()
        self.counter = FpsCounter()
        self.error = None
        self.proc = None
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        cmd = (["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
               + ffmpeg_input_args(self.args) + ["-c:v", "copy", "-f", "mjpeg", "-"])
        while self.running:
            try:
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, bufsize=0)
                for jpeg in iter_jpegs(self.proc.stdout):
                    self.frames.put(jpeg)
                    self.counter.tick()
                    self.error = None
                err = self.proc.stderr.read().decode(errors="replace").strip()
                self.proc.wait()
                self.error = err or f"ffmpeg exited with code {self.proc.returncode}"
            except Exception as e:  # keep the server alive; retry
                self.error = str(e)
            self.counter.fps = 0.0
            if self.running:
                print(f"capture stopped: {self.error}", file=sys.stderr)
                time.sleep(1)

    def stop(self):
        self.running = False
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.frames.close()


PAGE = (r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera Viewer</title>
<style>""" + BASE_CSS + r"""</style></head>
<body>
<header><h1>__DEVICE__</h1><span>__SIZE__ · __FORMAT__</span><span id="fps">– fps</span></header>
<main>
  <div id="view"><img id="stream" src="/stream" alt="camera stream"><div id="err"></div></div>
  <aside>""" + CONTROLS_HTML + r"""</aside>
</main>
<script>""" + CONTROLS_JS + r"""
setInterval(async () => {
  try {
    const s = await api('/api/status');
    $('#fps').textContent = `${s.fps.toFixed(1)} fps`;
    if (s.error) $('#err').textContent = s.error;
  } catch (e) {}
}, 1000);
</script>
</body></html>
""")


def make_handler(camera, args):
    controls = ControlsAPI(args.device)

    class Handler(JSONHandler):
        verbose = args.verbose

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                page = (PAGE.replace("__DEVICE__", args.device)
                            .replace("__SIZE__", args.size)
                            .replace("__FORMAT__", args.input_format))
                self.send_bytes(page.encode(), "text/html; charset=utf-8")
            elif path == "/stream":
                stream_mjpeg(self, camera.frames, args.stream_fps)
            elif path == "/snapshot.jpg":
                frame = camera.frames.frame
                if frame is None:
                    self.send_json({"error": "no frame yet"}, HTTPStatus.SERVICE_UNAVAILABLE)
                else:
                    self.send_bytes(frame, "image/jpeg")
            elif path == "/api/controls":
                try:
                    self.send_json(controls.get())
                except Exception as e:
                    self.send_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            elif path == "/api/status":
                self.send_json({"fps": camera.counter.fps, "error": camera.error})
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if not self.handle_controls_post(controls, path):
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="/dev/video2", help="V4L2 device (default: %(default)s)")
    p.add_argument("--size", default="1600x1304", help="Capture resolution WxH (default: %(default)s)")
    p.add_argument("--framerate", type=int, default=60, help="Capture framerate (default: %(default)s)")
    p.add_argument("--input-format", default="mjpeg",
                   help="V4L2 pixel format; must be mjpeg for passthrough streaming (default: %(default)s)")
    p.add_argument("--stream-fps", type=float, default=30,
                   help="Max frames/s sent to each browser, 0 = all (default: %(default)s)")
    p.add_argument("--host", default="127.0.0.1",
                   help="Address to listen on; use 0.0.0.0 to allow other machines (default: %(default)s)")
    p.add_argument("--port", type=int, default=8080, help="Port (default: %(default)s)")
    p.add_argument("--verbose", action="store_true", help="Log HTTP requests")
    args = p.parse_args()

    missing = [t for t in ("ffmpeg", "v4l2-ctl") if shutil.which(t) is None]
    if missing:
        sys.exit(f"Missing: {', '.join(missing)} (sudo apt install ffmpeg v4l-utils)")
    if args.input_format != "mjpeg":
        sys.exit("Only --input-format mjpeg is supported (frames are passed through unchanged)")

    camera = Camera(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(camera, args))
    server.daemon_threads = True
    host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"Streaming {args.device} at {args.size} -> http://{host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        server.server_close()


if __name__ == "__main__":
    main()
