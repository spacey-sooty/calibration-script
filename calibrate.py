#!/usr/bin/env python3
"""
Webcam -> ffmpeg -> corner detection -> mrcal calibration pipeline (LENSMODEL_OPENCV8).

Steps:
  1. Record a video from a V4L2 webcam with ffmpeg (or use an existing video via --video)
  2. Split the video into still frames with ffmpeg
  3. Detect board corners in every frame:
       --detector charuco   (default) OpenCV ChArUco detector
       --detector mrgingham plain square chessboard via mrgingham
  4. Solve the calibration with mrcal-calibrate-cameras using LENSMODEL_OPENCV8
  5. Export the result as a .cameramodel plus OpenCV-style JSON and YAML

Requirements (Debian/Ubuntu):
  sudo apt install ffmpeg mrcal python3-mrcal python3-opencv   # + mrgingham if used

Default board (ChArUco):
  7 x 5 squares (7 across, 5 down, as OpenCV counts them), 36 mm squares, 27 mm markers, DICT_4X4_50
  -> 6 x 4 = 24 inner corners, 36 mm apart.
  ChArUco boards can be partially visible: frames where only some corners are
  seen are still used (missing corners are marked as skipped for mrcal).

Live viewer:
  When recording from the webcam, a web page (http://localhost:8080) shows the
  stream with detected corners, a coverage map, stats, camera-control sliders
  and Start/Stop recording buttons. Recording starts when you press Start (or
  Enter in the terminal; --autostart to start immediately) and stops at
  --duration, on Stop/Enter, or on Ctrl+C. Use --no-viewer for the plain
  terminal-only recording.

Examples:
  # Record 60 s from /dev/video2 at 1280x720 with the default ChArUco board
  ./calibrate.py --size 1280x720 --duration 60

  # Use a video you already recorded
  ./calibrate.py --video board.mkv

  # Plain 14x14-inner-corner chessboard with 20 mm squares, via mrgingham
  ./calibrate.py --detector mrgingham --gridn 14 --square-length 0.020

Units: --square-length / --marker-length are in metres; the extrinsics come
out in the same unit.

Capture tips (these matter more than anything else for a good result):
  * Move the board slowly to avoid motion blur and rolling-shutter artifacts.
  * Get the board close enough to fill much of the frame, and cover the
    corners/edges of the image -- that's where the distortion is.
  * Tilt the board a lot (±40°+) in different directions.
  * Lock focus and exposure if the camera allows it (see --v4l2-ctl).
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, **kwargs):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def require_tools(*tools):
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        sys.exit(
            f"Missing required tools: {', '.join(missing)}\n"
            "Install with: sudo apt install ffmpeg mrgingham mrcal python3-mrcal"
        )


# --------------------------------------------------------------------------
# 1. Record
# --------------------------------------------------------------------------
def record_video(args, out_path: Path):
    if args.v4l2_ctl:
        # e.g. "focus_automatic_continuous=0,focus_absolute=0,auto_exposure=1"
        require_tools("v4l2-ctl")
        run(["v4l2-ctl", "-d", args.device, "--set-ctrl", args.v4l2_ctl])

    cmd = ["ffmpeg", "-hide_banner", "-y",
           "-f", "v4l2",
           "-input_format", args.input_format,
           "-video_size", args.size,
           "-framerate", str(args.framerate),
           "-i", args.device]
    if args.duration:
        cmd += ["-t", str(args.duration)]
    # Near-lossless encode so compression artifacts don't hurt corner detection.
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
            "-pix_fmt", "yuv420p", str(out_path)]

    print("\nRecording from webcam. Wave the calibration board around in front of it.")
    if not args.duration:
        print("Press 'q' in this terminal to stop recording.\n")
    run(cmd)


# --------------------------------------------------------------------------
# 2. Split into frames
# --------------------------------------------------------------------------
def extract_frames(video: Path, frames_dir: Path, fps: float):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()
    run(["ffmpeg", "-hide_banner", "-y", "-i", video,
         "-vf", f"fps={fps}",
         # grayscale PNG: lossless and what mrgingham works on anyway
         "-pix_fmt", "gray",
         str(frames_dir / "frame_%05d.png")])
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        sys.exit("ffmpeg produced no frames")
    print(f"Extracted {len(frames)} frames to {frames_dir}")
    return frames


# --------------------------------------------------------------------------
# 3. Detect corners
# --------------------------------------------------------------------------
def make_charuco_board(args):
    import cv2
    if not hasattr(cv2.aruco, "CharucoDetector"):
        sys.exit("OpenCV >= 4.7 is required for cv2.aruco.CharucoDetector")
    try:
        dict_id = getattr(cv2.aruco, args.aruco_dict)
    except AttributeError:
        sys.exit(f"Unknown ArUco dictionary {args.aruco_dict}")
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y),
                                   args.square_length, args.marker_length, dictionary)
    # Boards made with OpenCV < 4.6 (and some online generators) use the
    # old layout, which differs when the number of rows is even.
    board.setLegacyPattern(args.legacy_pattern)
    return board


_charuco_detector = None


def _charuco_init(args):
    import cv2
    global _charuco_detector
    cv2.setNumThreads(1)  # parallelism comes from the process pool
    _charuco_detector = cv2.aruco.CharucoDetector(make_charuco_board(args))


def _charuco_detect(frame):
    import cv2
    img = cv2.imread(str(frame), cv2.IMREAD_GRAYSCALE)
    corners, ids, _, _ = _charuco_detector.detectBoard(img)
    if ids is None or len(ids) == 0:
        return frame, {}
    return frame, {int(i): (float(c[0]), float(c[1]))
                   for i, c in zip(ids.ravel(), corners.reshape(-1, 2))}


def detect_corners_charuco(frames, corners_path: Path, args):
    """Write a mrcal corners.vnl from OpenCV ChArUco detections.

    mrcal wants exactly Nw*Nh rows per image in grid order (ChArUco corner ids
    are already row-major with Nw = squares_x-1 per row). Corners that weren't
    seen get level '-', which mrcal treats as "skip this point". Images with
    no usable detection are written as 'FILENAME - - -'.
    """
    import multiprocessing

    nw, nh = args.squares_x - 1, args.squares_y - 1
    used = 0
    # "spawn", not the default "fork": the live viewer runs OpenCV in background
    # threads, and forking a process while those threads hold locks leaves the
    # workers deadlocked forever.
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.jobs, initializer=_charuco_init, initargs=(args,)) as pool, \
         open(corners_path, "w") as f:
        f.write("# filename x y level\n")
        for frame, found in pool.imap(_charuco_detect, frames, chunksize=4):
            if len(found) < args.min_corners:
                f.write(f"{frame} - - -\n")
                continue
            used += 1
            for cid in range(nw * nh):
                if cid in found:
                    x, y = found[cid]
                    f.write(f"{frame} {x:.4f} {y:.4f} 0\n")
                else:
                    f.write(f"{frame} -1 -1 -\n")
    print(f"ChArUco board ({nw}x{nh} inner corners) usable in {used} of "
          f"{len(frames)} frames (>= {args.min_corners} corners each)")
    if used < 10:
        sys.exit("Too few frames with a detected board (need >= 10, ideally 50+). "
                 "Check board parameters (--aruco-dict, --squares-x/y, "
                 "--legacy-pattern), lighting, focus and motion blur.")


def detect_corners_mrgingham(frames_dir: Path, corners_path: Path, args):
    cmd = ["mrgingham", "--jobs", str(args.jobs), "--gridn", str(args.gridn)]
    if args.clahe:
        cmd.append("--clahe")
    if args.blur_radius is not None:
        cmd += ["--blur", str(args.blur_radius)]
    cmd.append(str(frames_dir / "frame_*.png"))  # mrgingham expands the glob itself

    with open(corners_path, "w") as f:
        run(cmd, stdout=f)

    # mrgingham writes "filename x y level" per corner, or "filename - - -" when
    # no board was found in that image.
    found, missed = set(), set()
    with open(corners_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            (missed if fields[1] == "-" else found).add(fields[0])
    print(f"Chessboard found in {len(found)} frames, missed in {len(missed)}")
    if len(found) < 10:
        sys.exit("Too few frames with a detected board (need >= 10, ideally 50+). "
                 "Check --gridn, lighting, focus and motion blur, or try --clahe.")
    return found


# --------------------------------------------------------------------------
# 4. Calibrate
# --------------------------------------------------------------------------
def image_size(frame: Path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(frame)],
        check=True, capture_output=True, text=True).stdout.strip()
    w, h = out.split(",")
    return int(w), int(h)


def calibrate(frames_dir: Path, corners_path: Path, outdir: Path, width: int,
              object_w: int, object_h: int, args):
    focal = args.focal
    if focal is None:
        # Seed focal length (pixels) from the approximate horizontal FOV.
        focal = (width / 2) / math.tan(math.radians(args.hfov) / 2)
        print(f"Seeding focal length at {focal:.1f} px (from hfov={args.hfov}°)")

    cmd = ["mrcal-calibrate-cameras",
           "--corners-cache", corners_path,
           "--lensmodel", "LENSMODEL_OPENCV8",
           "--focal", f"{focal:.2f}",
           "--object-spacing", str(args.square_length),
           "--object-width-n", str(object_w),
           "--object-height-n", str(object_h),
           "--outdir", outdir]
    if args.outlier_threshold is not None:
        cmd += ["--outlier", str(args.outlier_threshold)]
    cmd.append(str(frames_dir / "frame_*.png"))
    run(cmd)

    model = outdir / "camera-0.cameramodel"
    if not model.exists():
        sys.exit(f"mrcal did not produce {model}")
    return model


# --------------------------------------------------------------------------
# 5. Export to OpenCV formats
# --------------------------------------------------------------------------
def export_opencv(model_path: Path, outdir: Path):
    import numpy as np
    import mrcal

    m = mrcal.cameramodel(str(model_path))
    lensmodel, intrinsics = m.intrinsics()
    assert lensmodel == "LENSMODEL_OPENCV8", lensmodel
    width, height = (int(x) for x in m.imagersize())

    fx, fy, cx, cy = intrinsics[:4]
    # mrcal's OPENCV8 distortion order is k1 k2 p1 p2 k3 k4 k5 k6, which is
    # exactly the order cv2.undistort/projectPoints expect.
    dist = intrinsics[4:]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    # Calibration quality numbers from the stored optimization inputs
    rms = None
    try:
        oi = m.optimization_inputs()
        x = mrcal.optimizer_callback(**oi)[1]
        # residuals for the board observations only, as (x,y) pairs
        n_board = mrcal.num_measurements_boards(**oi)
        x_board = x[:n_board].reshape(-1, 2)
        weights = oi["observations_board"][..., 2].ravel()
        inliers = weights > 0
        rms = float(np.sqrt(np.mean(np.sum(x_board[inliers] ** 2, axis=-1) / 2)))
    except Exception as e:  # stats are a nice-to-have
        print(f"(Could not compute residual stats: {e})")

    result = {
        "lensmodel": lensmodel,
        "image_width": width,
        "image_height": height,
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist.tolist(),  # k1 k2 p1 p2 k3 k4 k5 k6
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "rms_reprojection_error_px": rms,
    }
    json_path = outdir / "opencv_calibration.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n")

    def fmt(arr):
        return ", ".join(f"{v:.12g}" for v in np.ravel(arr))

    # Readable by cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    yaml_path = outdir / "opencv_calibration.yaml"
    yaml_path.write_text(
        "%YAML:1.0\n---\n"
        f"image_width: {width}\n"
        f"image_height: {height}\n"
        "camera_matrix: !!opencv-matrix\n"
        "   rows: 3\n   cols: 3\n   dt: d\n"
        f"   data: [ {fmt(K)} ]\n"
        "distortion_coefficients: !!opencv-matrix\n"
        f"   rows: 1\n   cols: {len(dist)}\n   dt: d\n"
        f"   data: [ {fmt(dist)} ]\n"
        + (f"rms_reprojection_error: {rms:.6f}\n" if rms is not None else "")
    )

    print("\n================ Calibration result (LENSMODEL_OPENCV8) ================")
    print(f"Image size : {width} x {height}")
    print(f"fx, fy     : {fx:.3f}, {fy:.3f}")
    print(f"cx, cy     : {cx:.3f}, {cy:.3f}")
    print(f"k1 k2 p1 p2 k3 k4 k5 k6:\n  {' '.join(f'{d:.6g}' for d in dist)}")
    if rms is not None:
        print(f"RMS reprojection error (weighted, inliers): {rms:.4f} px")
    print(f"\nmrcal model : {model_path}")
    print(f"OpenCV JSON : {json_path}")
    print(f"OpenCV YAML : {yaml_path}")
    print("\nTo sanity-check the result, try:")
    print(f"  mrcal-show-residuals-board-observation --from-worst --vectorscale 100 {model_path} 0")
    print(f"  mrcal-show-projection-uncertainty --cbmax 1 {model_path}")
    print(f"  mrcal-show-residuals --histogram {model_path}")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_argument_group("video source")
    src.add_argument("--video", type=Path,
                     help="Use an existing video instead of recording from the webcam")
    src.add_argument("--device", default="/dev/video2", help="V4L2 device (default: %(default)s)")
    src.add_argument("--size", default="1280x720", help="Capture resolution WxH (default: %(default)s)")
    src.add_argument("--framerate", type=int, default=30, help="Capture framerate (default: %(default)s)")
    src.add_argument("--input-format", default="mjpeg",
                     help="V4L2 pixel format, e.g. mjpeg or yuyv422 (default: %(default)s)")
    src.add_argument("--duration", type=float, default=60,
                     help="Seconds to record; 0 = until stopped (default: %(default)s)")
    src.add_argument("--v4l2-ctl", metavar="CTRLS",
                     help="Controls to set before recording, passed to "
                          "'v4l2-ctl --set-ctrl', e.g. 'focus_automatic_continuous=0,focus_absolute=0'")

    vw = p.add_argument_group("live viewer (while recording)")
    vw.add_argument("--no-viewer", action="store_true",
                    help="Record without the web viewer (press q in the terminal to stop)")
    vw.add_argument("--autostart", action="store_true",
                    help="Start recording immediately instead of waiting for Start")
    vw.add_argument("--viewer-host", default="127.0.0.1",
                    help="Address to listen on; 0.0.0.0 to allow other machines (default: %(default)s)")
    vw.add_argument("--viewer-port", type=int, default=8080, help="Port (default: %(default)s)")
    vw.add_argument("--viewer-width", type=int, default=1280,
                    help="Downscale the viewer stream to at most this width (default: %(default)s)")
    vw.add_argument("--stream-fps", type=float, default=30,
                    help="Max frames/s sent to the browser, 0 = all (default: %(default)s)")

    fr = p.add_argument_group("frame extraction")
    fr.add_argument("--fps", type=float, default=2,
                    help="Frames per second of video to keep for calibration (default: %(default)s)")

    bd = p.add_argument_group("calibration board")
    bd.add_argument("--detector", choices=["charuco", "mrgingham"], default="charuco",
                    help="Corner detector (default: %(default)s)")
    bd.add_argument("--square-length", type=float, default=0.036,
                    help="Checker square size = corner spacing, metres (default: %(default)s)")
    bd.add_argument("--jobs", type=int, default=8, help="Parallel detection jobs (default: %(default)s)")

    ch = p.add_argument_group("ChArUco (--detector charuco)")
    ch.add_argument("--squares-x", type=int, default=7,
                    help="Number of squares horizontally (default: %(default)s)")
    ch.add_argument("--squares-y", type=int, default=5,
                    help="Number of squares vertically (default: %(default)s)")
    ch.add_argument("--marker-length", type=float, default=0.027,
                    help="ArUco marker size, metres (default: %(default)s)")
    ch.add_argument("--aruco-dict", default="DICT_4X4_50",
                    help="cv2.aruco predefined dictionary (default: %(default)s)")
    ch.add_argument("--legacy-pattern", action="store_true",
                    help="Board uses the pre-OpenCV-4.6 ChArUco layout")
    ch.add_argument("--min-corners", type=int, default=8,
                    help="Skip frames with fewer detected ChArUco corners (default: %(default)s)")

    mg = p.add_argument_group("plain chessboard (--detector mrgingham)")
    mg.add_argument("--gridn", type=int, default=14,
                    help="Inner corners per side of the square chessboard (default: %(default)s)")
    mg.add_argument("--clahe", action="store_true", help="Pass --clahe to mrgingham (helps uneven lighting)")
    mg.add_argument("--blur-radius", type=int, help="Pass --blur N to mrgingham")

    cal = p.add_argument_group("mrcal")
    cal.add_argument("--focal", type=float, help="Seed focal length in pixels (overrides --hfov)")
    cal.add_argument("--hfov", type=float, default=70,
                     help="Approx. horizontal FOV in degrees, used to seed the focal length (default: %(default)s)")
    cal.add_argument("--outlier-threshold", type=float,
                     help="Pass --outlier to mrcal-calibrate-cameras")

    p.add_argument("--workdir", type=Path, default=Path("calib_out"),
                   help="Directory for video, frames and results (default: %(default)s)")
    args = p.parse_args()

    require_tools("ffmpeg", "ffprobe", "mrcal-calibrate-cameras")
    if args.detector == "mrgingham":
        require_tools("mrgingham")
    else:
        try:
            import cv2  # noqa: F401
        except ImportError:
            sys.exit("Python module 'cv2' not found. Install with: sudo apt install python3-opencv")
        if args.marker_length >= args.square_length:
            sys.exit("--marker-length must be smaller than --square-length")
    try:
        import mrcal  # noqa: F401
    except ImportError:
        sys.exit("Python module 'mrcal' not found. Install with: sudo apt install python3-mrcal")

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    frames_dir = workdir / "frames"
    corners_path = workdir / "corners.vnl"

    # 1. Record
    viewer = None
    if args.video:
        video = args.video.resolve()
        if not video.exists():
            sys.exit(f"No such video: {video}")
    else:
        video = workdir / "capture.mkv"
        if video.exists():
            video.unlink()
        if args.no_viewer:
            record_video(args, video)
        else:
            try:
                import cv2  # noqa: F401
            except ImportError:
                sys.exit("The viewer needs OpenCV: sudo apt install python3-opencv "
                         "(or use --no-viewer)")
            from calibration_viewer import CalibrationViewer
            if args.v4l2_ctl:
                require_tools("v4l2-ctl")
                run(["v4l2-ctl", "-d", args.device, "--set-ctrl", args.v4l2_ctl])
            viewer = CalibrationViewer(args, video, make_charuco_board)
            if not viewer.record():
                viewer.close()
                sys.exit("No video was recorded.")

    def phase(text):
        if viewer:
            viewer.set_phase(text)

    try:
        # 2. Split into frames
        phase("Extracting frames...")
        frames = extract_frames(video, frames_dir, args.fps)
        width, height = image_size(frames[0])
        print(f"Frame size: {width}x{height}")

        # 3. Detect board corners
        phase(f"Detecting corners in {len(frames)} frames...")
        if args.detector == "charuco":
            detect_corners_charuco(frames, corners_path, args)
            object_w, object_h = args.squares_x - 1, args.squares_y - 1
        else:
            detect_corners_mrgingham(frames_dir, corners_path, args)
            object_w = object_h = args.gridn

        # 4. Calibrate
        phase("Solving calibration with mrcal (this can take a while)...")
        model = calibrate(frames_dir, corners_path, workdir, width, object_w, object_h, args)

        # 5. Export
        export_opencv(model, workdir)
    finally:
        if viewer:
            viewer.close()


if __name__ == "__main__":
    main()
