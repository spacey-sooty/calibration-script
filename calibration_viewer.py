"""
Live web viewer used by calibrate.py while recording.

Shows the camera stream with the detected calibration-board corners drawn on
top, a coverage map of where in the image corners have been seen, live stats,
Start/Stop recording buttons and the camera-control sliders from
camera_viewer.py.

Flow:
  preview  -> (Start button / Enter in terminal) -> recording
  recording -> (Stop button / Enter / Ctrl+C / --duration reached) -> done
While recording, a single ffmpeg writes the lossless capture file AND a MJPEG
stream for the viewer, so what you see is exactly what's being recorded.
"""

import shutil
import signal
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

from camera_viewer import (BASE_CSS, CONTROLS_HTML, CONTROLS_JS, ControlsAPI, FpsCounter,
                           FrameBuffer, JSONHandler, ffmpeg_input_args, iter_jpegs,
                           stream_mjpeg)

GOOD = (80, 200, 80)      # BGR
PARTIAL = (0, 170, 255)
MARKER = (255, 140, 60)


# --------------------------------------------------------------------------
# Live corner detection
# --------------------------------------------------------------------------
class LiveDetector:
    def __init__(self, args, make_charuco_board):
        self.args = args
        if args.detector == "charuco":
            self.n_total = (args.squares_x - 1) * (args.squares_y - 1)
            self.detector = cv2.aruco.CharucoDetector(make_charuco_board(args))
        else:
            # mrgingham itself is a CLI tool that works on files; for the live
            # preview use OpenCV's chessboard detector on the same board.
            self.n_total = args.gridn * args.gridn
            self.detector = None

    def detect(self, gray):
        """Returns (corners Nx2 or None, ids or None, marker_corners, marker_ids)."""
        if self.detector is not None:
            corners, ids, mcorners, mids = self.detector.detectBoard(gray)
            if corners is not None and len(corners):
                corners = corners.reshape(-1, 2)
            else:
                corners = None
            return corners, ids, mcorners, mids
        # Chessboard: detect on a downscaled image to keep the preview responsive
        scale = 1.0
        small = gray
        if gray.shape[1] > 1000:
            scale = 1000 / gray.shape[1]
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        n = self.args.gridn
        ok, corners = cv2.findChessboardCornersSB(small, (n, n))
        if not ok:
            return None, None, None, None
        return corners.reshape(-1, 2) / scale, None, None, None


class Coverage:
    """Counts how many sampled board observations covered each image cell."""

    def __init__(self, cols=16):
        self.cols = cols
        self.grid = None

    def reset(self):
        self.grid = None

    def add(self, corners, shape):
        h, w = shape[:2]
        rows = max(1, round(self.cols * h / w))
        if self.grid is None or self.grid.shape != (rows, self.cols):
            self.grid = np.zeros((rows, self.cols), np.int32)
        cx = np.clip((corners[:, 0] / w * self.cols).astype(int), 0, self.cols - 1)
        cy = np.clip((corners[:, 1] / h * rows).astype(int), 0, rows - 1)
        hit = np.zeros_like(self.grid, bool)
        hit[cy, cx] = True
        self.grid += hit

    def fraction(self):
        return 0.0 if self.grid is None else float((self.grid > 0).mean())

    def draw(self, img):
        if self.grid is None:
            return
        h, w = img.shape[:2]
        rows, cols = self.grid.shape
        overlay = img.copy()
        for r in range(rows):
            for c in range(cols):
                n = self.grid[r, c]
                x0, x1 = c * w // cols, (c + 1) * w // cols
                y0, y1 = r * h // rows, (r + 1) * h // rows
                if n:
                    # green, stronger with more observations (saturates at 5)
                    g = 90 + int(120 * min(n, 5) / 5)
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, g, 0), -1)
                else:
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 120), -1)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, dst=img)


# --------------------------------------------------------------------------
# Capture + detection + web server
# --------------------------------------------------------------------------
class CalibrationViewer:
    def __init__(self, args, video_path, make_charuco_board):
        self.args = args
        self.video_path = video_path
        self.detector = LiveDetector(args, make_charuco_board)
        self.coverage = Coverage()
        self.show_coverage = True

        self.raw = FrameBuffer()
        self.annotated = FrameBuffer()
        self.cam_fps = FpsCounter()
        self.det_fps = FpsCounter()

        self.lock = threading.Lock()
        self.mode = "preview"           # preview | recording | finished
        self.phase = "Preview - adjust the camera, then press Start recording"
        self.error = None
        self.proc = None
        self.rec_start = None
        self.rec_end = None
        self.recording_done = threading.Event()
        self.samples = 0                # frames sampled at --fps while recording
        self.good_samples = 0           # ...of which had enough corners
        self.n_corners = 0
        self.result = None

        threading.Thread(target=self._capture_loop, daemon=True).start()
        threading.Thread(target=self._detect_loop, daemon=True).start()
        self._start_server()

    # ---------------- ffmpeg ----------------
    def _cmd(self, recording):
        a = self.args
        # MJPEG cameras: forward the camera's JPEGs untouched; otherwise encode.
        view = ["-c:v", "copy"] if a.input_format == "mjpeg" else ["-c:v", "mjpeg", "-q:v", "4"]
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + ffmpeg_input_args(a)
        if recording:
            t = ["-t", str(a.duration)] if a.duration else []
            # Lossless encode so compression artifacts don't hurt corner detection
            cmd += t + ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
                        "-pix_fmt", "yuv420p", str(self.video_path)]
            cmd += t
        return cmd + view + ["-f", "mjpeg", "-"]

    def _capture_loop(self):
        while True:
            with self.lock:
                mode = self.mode
            if mode == "finished":
                return
            recording = mode == "recording"
            try:
                # own session, so Ctrl+C in the terminal reaches only Python,
                # which then stops ffmpeg cleanly
                proc = subprocess.Popen(self._cmd(recording), stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        bufsize=0, start_new_session=True)
            except OSError as e:
                self.error = str(e)
                time.sleep(1)
                continue
            with self.lock:
                self.proc = proc
                if recording:
                    self.rec_start = time.monotonic()
            for jpeg in iter_jpegs(proc.stdout):
                self.raw.put(jpeg)
                self.cam_fps.tick()
            err = proc.stderr.read().decode(errors="replace").strip()
            proc.wait()
            self.cam_fps.fps = 0.0
            if recording:
                with self.lock:
                    self.rec_end = time.monotonic()
                    self.mode = "finished"
                if proc.returncode not in (0, 255, -signal.SIGINT) or err:
                    self.error = err or f"ffmpeg exited with code {proc.returncode}"
                self.phase = "Recording finished"
                self.recording_done.set()
                return
            with self.lock:
                switching = self.mode != "preview"
            if not switching:
                # preview died on its own (e.g. device busy/unplugged): retry
                self.error = err or f"ffmpeg exited with code {proc.returncode}"
                print(f"preview stopped: {self.error}", file=sys.stderr)
                time.sleep(1)
            else:
                self.error = None

    def start_recording(self):
        with self.lock:
            if self.mode != "preview":
                return False
            self.mode = "recording"
            self.coverage.reset()
            self.samples = self.good_samples = 0
            self.phase = "Recording"
            proc = self.proc
        print("Recording started" + (f" for {self.args.duration:g} s" if self.args.duration else ""))
        if proc and proc.poll() is None:
            proc.terminate()  # stop preview; capture loop restarts in recording mode
        return True

    def stop_recording(self):
        with self.lock:
            proc = self.proc if self.mode == "recording" else None
        if proc is None or proc.poll() is not None:
            return False
        print("Stopping recording...")
        proc.send_signal(signal.SIGINT)  # ffmpeg finalizes the file on SIGINT
        return True

    def toggle(self):
        if self.mode == "preview":
            self.start_recording()
        elif self.mode == "recording":
            self.stop_recording()

    # ---------------- detection ----------------
    def _detect_loop(self):
        a = self.args
        sample_dt = 1.0 / a.fps if a.fps > 0 else 0
        last_id, last_sample = -1, 0.0
        while True:
            fid, jpeg = self.raw.wait(last_id)
            if jpeg is None or fid == last_id:
                if self.raw.closed:
                    return
                continue
            last_id = fid
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            try:
                corners, ids, mcorners, mids = self.detector.detect(gray)
            except cv2.error as e:
                self.error = f"detector: {e}"
                corners = ids = mcorners = mids = None
            n = 0 if corners is None else len(corners)
            self.n_corners = n
            enough = n >= (a.min_corners if a.detector == "charuco" else self.detector.n_total)

            now = time.monotonic()
            if self.mode == "recording" and now - last_sample >= sample_dt:
                last_sample = now
                self.samples += 1
                if enough:
                    self.good_samples += 1
                    self.coverage.add(corners, img.shape)

            if self.show_coverage:
                self.coverage.draw(img)
            self._draw(img, corners, ids, mcorners, mids, enough)
            self._draw_banner(img, n)

            if img.shape[1] > a.viewer_width:
                s = a.viewer_width / img.shape[1]
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.annotated.put(enc.tobytes())
            self.det_fps.tick()

    def _draw(self, img, corners, ids, mcorners, mids, enough):
        thick = max(1, img.shape[1] // 800)
        if mids is not None and len(mids):
            cv2.aruco.drawDetectedMarkers(img, mcorners, mids, MARKER)
        if corners is None:
            return
        color = GOOD if enough else PARTIAL
        r = max(4, img.shape[1] // 200)
        if self.args.detector == "charuco":
            for (x, y), i in zip(corners, ids.ravel()):
                p = (int(round(x)), int(round(y)))
                cv2.circle(img, p, r, color, thick + 1, cv2.LINE_AA)
                cv2.putText(img, str(int(i)), (p[0] + r, p[1] - r), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4 * thick, color, thick, cv2.LINE_AA)
        else:
            n = self.args.gridn
            cv2.drawChessboardCorners(img, (n, n), corners.reshape(-1, 1, 2).astype(np.float32), True)

    def _draw_banner(self, img, n):
        thick = max(1, img.shape[1] // 800)
        if self.mode == "recording":
            cv2.circle(img, (30 * thick, 30 * thick), 10 * thick, (0, 0, 255), -1, cv2.LINE_AA)
            text = f"REC {self.elapsed():.0f}s"
        else:
            text = "PREVIEW" if self.mode == "preview" else "STOPPED"
        text += f"   corners {n}/{self.detector.n_total}"
        cv2.putText(img, text, (50 * thick, 38 * thick), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * thick, (0, 0, 0), 4 * thick, cv2.LINE_AA)
        cv2.putText(img, text, (50 * thick, 38 * thick), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * thick, (255, 255, 255), thick + 1, cv2.LINE_AA)

    def elapsed(self):
        if self.rec_start is None:
            return 0.0
        return (self.rec_end or time.monotonic()) - self.rec_start

    # ---------------- status / server ----------------
    def status(self):
        return {
            "mode": self.mode,
            "phase": self.phase,
            "error": self.error,
            "elapsed": self.elapsed(),
            "duration": self.args.duration,
            "cam_fps": self.cam_fps.fps,
            "det_fps": self.det_fps.fps,
            "corners": self.n_corners,
            "corners_total": self.detector.n_total,
            "samples": self.samples,
            "good_samples": self.good_samples,
            "coverage": self.coverage.fraction(),
            "show_coverage": self.show_coverage,
            "result": self.result,
        }

    def set_phase(self, text):
        self.phase = text
        print(f"[viewer] {text}")

    def _start_server(self):
        viewer, a = self, self.args
        controls = ControlsAPI(a.device)
        page = (PAGE.replace("__DEVICE__", a.device)
                    .replace("__SIZE__", a.size)
                    .replace("__FORMAT__", a.input_format)).encode()

        class Handler(JSONHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/":
                    self.send_bytes(page, "text/html; charset=utf-8")
                elif path == "/stream":
                    stream_mjpeg(self, viewer.annotated, a.stream_fps)
                elif path == "/api/status":
                    self.send_json(viewer.status())
                elif path == "/api/controls":
                    try:
                        self.send_json(controls.get())
                    except Exception as e:
                        self.send_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                else:
                    self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

            def do_POST(self):
                path = self.path.split("?", 1)[0]
                if path in ("/api/start", "/api/stop", "/api/coverage"):
                    try:
                        body = self.read_json()
                    except ValueError as e:
                        self.send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                        return
                    if path == "/api/start":
                        viewer.start_recording()
                    elif path == "/api/stop":
                        viewer.stop_recording()
                    else:
                        viewer.show_coverage = bool(body.get("show", True))
                    self.send_json(viewer.status())
                elif not self.handle_controls_post(controls, path):
                    self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        self.server = ThreadingHTTPServer((a.viewer_host, a.viewer_port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        host = "localhost" if a.viewer_host in ("127.0.0.1", "0.0.0.0") else a.viewer_host
        self.url = f"http://{host}:{a.viewer_port}"

    def _stdin_toggle(self):
        for _ in sys.stdin:
            if self.mode == "finished":
                return
            self.toggle()

    def record(self):
        """Block until recording is finished. Returns True if a video was recorded."""
        print(f"\nViewer: {self.url}")
        if self.args.autostart:
            self.start_recording()
        else:
            print("Press Start recording on the page (or Enter here) to begin.")
        if sys.stdin and sys.stdin.isatty():
            print("Press Enter to start/stop, Ctrl+C to stop recording (or abort during preview).")
            threading.Thread(target=self._stdin_toggle, daemon=True).start()
        try:
            while not self.recording_done.wait(0.2):
                pass
        except KeyboardInterrupt:
            if self.mode != "recording":
                self.close()
                sys.exit("Aborted before recording.")
            self.stop_recording()
            self.recording_done.wait()
        self.raw.close()
        if self.error:
            print(f"ffmpeg: {self.error}", file=sys.stderr)
        print(f"Recorded {self.elapsed():.1f} s: ~{self.good_samples} usable frames, "
              f"{self.coverage.fraction() * 100:.0f}% of the image covered")
        return self.video_path.exists() and self.video_path.stat().st_size > 0

    def close(self):
        with self.lock:
            proc, self.mode = self.proc, "finished"
        if proc and proc.poll() is None:
            proc.terminate()
        self.raw.close()
        self.annotated.close()
        self.server.shutdown()
        self.server.server_close()


PAGE = (r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calibration Viewer</title>
<style>""" + BASE_CSS + r"""
#rec button.big { font-size:15px; padding:8px 14px; width:100%; margin:8px 0; }
#start { border-color:var(--good); }
#stop { border-color:var(--bad); }
.stats { display:grid; grid-template-columns:auto 1fr; gap:2px 12px; font-variant-numeric:tabular-nums; }
.stats dt { color:var(--muted); }
.stats dd { margin:0; text-align:right; }
.bar { height:6px; background:var(--bg); border-radius:3px; overflow:hidden; margin-top:2px; }
.bar > div { height:100%; background:var(--good); width:0; transition:width .3s; }
#phase { margin-top:6px; }
#phase.rec { color:var(--bad); font-weight:600; }
ul.tips { margin:6px 0 0; padding-left:18px; color:var(--muted); font-size:12px; }
</style></head>
<body>
<header><h1>Calibration · __DEVICE__</h1><span>__SIZE__ · __FORMAT__</span>
  <span id="fps">– fps</span></header>
<main>
  <div id="view"><img id="stream" src="/stream" alt="camera stream with detected corners">
    <div id="err"></div></div>
  <aside>
    <div class="panel" id="rec">
      <strong>Recording</strong>
      <div id="phase" class="hint">–</div>
      <button class="big" id="start">● Start recording</button>
      <button class="big" id="stop" hidden>■ Stop recording</button>
      <dl class="stats">
        <dt>Time</dt><dd id="time">–</dd>
        <dt>Corners in view</dt><dd id="corners">–</dd>
        <dt>Usable frames</dt><dd id="good">–</dd>
        <dt>Image coverage</dt><dd id="cov">–</dd>
      </dl>
      <div class="bar"><div id="covbar"></div></div>
      <label class="row hint" style="margin-top:8px; justify-content:flex-start">
        <input type="checkbox" id="showcov" checked> Show coverage map</label>
      <ul class="tips">
        <li>Green dots: corners found. Orange: too few to use.</li>
        <li>Fill the red cells, especially the image edges and corners.</li>
        <li>Move slowly and tilt the board a lot.</li>
      </ul>
    </div>
""" + CONTROLS_HTML + r"""
  </aside>
</main>
<script>""" + CONTROLS_JS + r"""
function fmtTime(s) { const m = Math.floor(s / 60); return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`; }

function renderStatus(s) {
  $('#fps').textContent = `camera ${s.cam_fps.toFixed(1)} fps · detection ${s.det_fps.toFixed(1)} fps`;
  const ph = $('#phase');
  ph.textContent = s.phase;
  ph.className = s.mode === 'recording' ? 'rec' : 'hint';
  $('#start').hidden = s.mode !== 'preview';
  $('#stop').hidden = s.mode !== 'recording';
  $('#time').textContent = s.mode === 'preview' ? '–'
      : fmtTime(s.elapsed) + (s.duration ? ` / ${fmtTime(s.duration)}` : '');
  $('#corners').textContent = `${s.corners} / ${s.corners_total}`;
  $('#good').textContent = s.mode === 'preview' ? '–' : `${s.good_samples} / ${s.samples}`;
  $('#cov').textContent = `${Math.round(s.coverage * 100)}%`;
  $('#covbar').style.width = `${s.coverage * 100}%`;
  if (s.error) $('#err').textContent = s.error;
}

async function poll() {
  try { renderStatus(await api('/api/status')); } catch (e) { $('#phase').textContent = 'viewer offline (calibration finished?)'; }
}
$('#start').onclick = async () => renderStatus(await api('/api/start', {}));
$('#stop').onclick = async () => renderStatus(await api('/api/stop', {}));
$('#showcov').onchange = async () => renderStatus(await api('/api/coverage', {show: $('#showcov').checked}));
setInterval(poll, 500);
poll();
</script>
</body></html>
""")
