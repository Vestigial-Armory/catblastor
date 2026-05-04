from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from ultralytics import YOLO
from adafruit_servokit import ServoKit
import cv2
import threading
import time
import json
import numpy as np
import RPi.GPIO as GPIO
import os
import subprocess
from datetime import datetime
from pathlib import Path

# ─── Constants ───────────────────────────────────────────────────────────────
FRAME_W, FRAME_H = 640, 480
INFER_W, INFER_H = 320, 240
PAN_CH, TILT_CH = 0, 1
PUMP_PIN, SOLENOID_PIN = 27, 17
CAT_CLASS = 15
RECORDINGS_DIR = Path("/home/wolfhard/catblastor/recordings")
CALIBRATION_FILE = Path("/home/wolfhard/catblastor/calibration.json")
FOV_FILE = Path("/home/wolfhard/catblastor/fov_calibration.json")
INFER_PIPE = "/tmp/catblastor_infer"
RECORDINGS_DIR.mkdir(exist_ok=True)
INFER_PIPE = "/tmp/catblastor_infer"

if not os.path.exists(INFER_PIPE):
    os.mkfifo(INFER_PIPE)

def start_streaming():
    """
    rpicam-vid outputs MJPEG to stdout.
    ffmpeg reads it and:
      1. Serves MJPEG over HTTP on port 8888 for browser
      2. Tees raw BGR frames at inference resolution to named pipe for Python
    """
    rpicam_cmd = [
        "rpicam-vid",
        "--width", "640",
        "--height", "480",
        "--framerate", "30",
        "--codec", "mjpeg",
        "--inline",
        "-o", "-",
        "-t", "0",
        "--nopreview",
        "--vflip",
        "--hflip",
    ]

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "mjpeg",
        "-i", "pipe:0",
        # Output 1 — HTTP MJPEG for browser
        "-map", "0:v",
        "-c:v", "copy",
        "-f", "mjpeg",
        "-listen", "1",
        "http://0.0.0.0:8888/stream",
        # Output 2 — raw BGR frames for inference
        "-map", "0:v",
        "-vf", f"scale={INFER_W}:{INFER_H},format=bgr24",
        "-r", "5",
        "-f", "rawvideo",
        INFER_PIPE,
    ]

    rpicam_proc = subprocess.Popen(
        rpicam_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    time.sleep(1.0)
    subprocess.Popen(
        ffmpeg_cmd,
        stdin=rpicam_proc.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print("Streaming started")

# Mutable FOV — updated by zone drag calibration
fov = {"h": 66.0, "v": 41.0}
if FOV_FILE.exists():
    fov.update(json.loads(FOV_FILE.read_text()))

def save_fov():
    FOV_FILE.write_text(json.dumps(fov))

# ─── GPIO Setup ──────────────────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM)
GPIO.setup(PUMP_PIN, GPIO.OUT)
GPIO.setup(SOLENOID_PIN, GPIO.OUT)
GPIO.output(PUMP_PIN, GPIO.LOW)
GPIO.output(SOLENOID_PIN, GPIO.LOW)

# ─── Hardware Init ───────────────────────────────────────────────────────────
kit = ServoKit(channels=16)
kit.servo[PAN_CH].angle = 90
kit.servo[TILT_CH].angle = 90

model = YOLO("yolov8n.pt")

# ─── State ───────────────────────────────────────────────────────────────────
state = {
    "armed": False,
    "mode": "live",           # live | zone_draw | calibration
    "firing_mode": "single",  # single | semi_auto
    "burst_length": 1.0,
    "reload_time": 10.0,
    "semi_auto_delay": 2.0,
    "confidence_threshold": 0.5,
    "on_target_tolerance": 20,
    "zone_points": [],
    "zone_closed": False,
}

calibration = {"offset_x": 0, "offset_y": 0}
if CALIBRATION_FILE.exists():
    calibration.update(json.loads(CALIBRATION_FILE.read_text()))

servo_angles = {"pan": 90.0, "tilt": 90.0}

# ─── Home Position & Activity ─────────────────────────────────────────────────
HOME_FILE = Path("/home/wolfhard/catblastor/home_position.json")
home_position = {"pan": 90.0, "tilt": 90.0}
if HOME_FILE.exists():
    home_position.update(json.loads(HOME_FILE.read_text()))
last_activity_time = time.time()
HOME_TIMEOUT = 60.0  # seconds of inactivity before returning home

def save_home_position():
    HOME_FILE.write_text(json.dumps(home_position))

# ─── Target Tracking State ───────────────────────────────────────────────────
tracking = {
    "primary_target_id": None,
    "target_entry_times": {},
    "last_seen": {},
    "locked": False,
}

firing = {
    "active": False,
    "last_fire_time": 0,
}

# ─── Shared Frame Data ───────────────────────────────────────────────────────
frame_lock = threading.Lock()
latest_frame = None
latest_detections = []
detection_lock = threading.Lock()

# ─── Recording State ─────────────────────────────────────────────────────────
recording = {
    "active": False,
    "writer": None,
    "last_detection_time": 0,
    "filename": None,
}

# ─── Calibration State ───────────────────────────────────────────────────────
calibration_state = {
    "active": False,
    "reticle_x": FRAME_W // 2,
    "reticle_y": FRAME_H // 2,
}

# ─── Helpers ─────────────────────────────────────────────────────────────────
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def point_in_polygon(cx, cy, polygon):
    if len(polygon) < 3:
        return False
    pts = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0

def pixel_to_angle(cx, cy):
    pan  = 90.0 - ((cx - FRAME_W / 2) / FRAME_W) * fov["h"]
    tilt = 90.0 - ((cy - FRAME_H / 2) / FRAME_H) * fov["v"]
    return clamp(pan, 45, 135), clamp(tilt, 45, 135)

def pixel_to_world_angle(cx, cy):
    pan  = servo_angles["pan"]  - ((cx - FRAME_W / 2) / FRAME_W) * fov["h"]
    tilt = servo_angles["tilt"] - ((cy - FRAME_H / 2) / FRAME_H) * fov["v"]
    return pan, tilt

def world_angle_to_pixel(pan, tilt):
    cx = int(-(pan  - servo_angles["pan"])  / fov["h"] * FRAME_W + FRAME_W / 2)
    cy = int(-(tilt - servo_angles["tilt"]) / fov["v"] * FRAME_H + FRAME_H / 2)
    return cx, cy

def point_in_angle_zone(cx, cy, zone_points):
    if len(zone_points) < 3:
        return False
    pan, tilt = pixel_to_world_angle(cx, cy)
    pts = np.array(zone_points, dtype=np.float32)
    return cv2.pointPolygonTest(pts, (float(pan), float(tilt)), False) >= 0

def move_servos(pan, tilt):
    global last_activity_time
    kit.servo[PAN_CH].angle  = clamp(pan,  45, 135)
    kit.servo[TILT_CH].angle = clamp(tilt, 45, 135)
    servo_angles["pan"]  = pan
    servo_angles["tilt"] = tilt
    last_activity_time = time.time()

def save_calibration():
    CALIBRATION_FILE.write_text(json.dumps(calibration))

def get_reticle_pos():
    return calibration_state["reticle_x"], calibration_state["reticle_y"]

# ─── Home Position Thread ────────────────────────────────────────────────────
def home_position_loop():
    while True:
        time.sleep(5)
        if not state["armed"]:
            continue
        with detection_lock:
            cats = len(latest_detections)
        if cats == 0 and not firing["active"]:
            if time.time() - last_activity_time > HOME_TIMEOUT:
                move_servos(home_position["pan"], home_position["tilt"])

# ─── Capture Thread — reads raw frames from ffmpeg inference pipe ─────────────
def capture_loop():
    global latest_frame
    frame_size = INFER_W * INFER_H * 3  # BGR24
    print("Waiting for inference pipe...")
    while True:
        try:
            pipe = open(INFER_PIPE, "rb")
            print("Inference pipe connected")
            break
        except Exception:
            time.sleep(0.5)
    while True:
        try:
            data = pipe.read(frame_size)
            if len(data) != frame_size:
                pipe.close()
                time.sleep(0.1)
                pipe = open(INFER_PIPE, "rb")
                continue
            frame = np.frombuffer(data, dtype=np.uint8).reshape((INFER_H, INFER_W, 3))
            with frame_lock:
                latest_frame = frame.copy()
        except Exception as e:
            print(f"Pipe read error: {e}")
            time.sleep(0.1)

# ─── Inference Thread ────────────────────────────────────────────────────────
def inference_loop():
    global latest_detections
    while True:
        if not state["armed"]:
            latest_detections = []
            time.sleep(0.1)
            continue
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
        results = model(frame, verbose=False)
        scale_x = FRAME_W / INFER_W
        scale_y = FRAME_H / INFER_H
        detections = []
        for r in results[0].boxes:
            cls  = int(r.cls)
            conf = float(r.conf)
            if cls == CAT_CLASS and conf >= state["confidence_threshold"]:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                in_zone = (
                    point_in_angle_zone(cx, cy, state["zone_points"])
                    if state["zone_closed"] and len(state["zone_points"]) >= 3
                    else False
                )
                detections.append({
                    "id": f"{cx}_{cy}",
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": cx, "cy": cy,
                    "conf": conf,
                    "in_zone": in_zone,
                })
        with detection_lock:
            latest_detections = detections

# ─── Target Priority ─────────────────────────────────────────────────────────
def select_target(detections):
    in_zone = [d for d in detections if d["in_zone"]]
    if not in_zone:
        tracking["primary_target_id"] = None
        tracking["locked"] = False
        return None

    now = time.time()
    rx, ry = get_reticle_pos()

    for d in in_zone:
        if d["id"] not in tracking["target_entry_times"]:
            tracking["target_entry_times"][d["id"]] = now
        tracking["last_seen"][d["id"]] = now

    current_ids = {d["id"] for d in in_zone}
    for tid in list(tracking["target_entry_times"].keys()):
        if tid not in current_ids:
            tracking["target_entry_times"].pop(tid, None)
            tracking["last_seen"].pop(tid, None)
            if tracking["primary_target_id"] == tid:
                tracking["primary_target_id"] = None
                tracking["locked"] = False
                firing["last_fire_time"] = 0  # immediate fire for new target

    if tracking["primary_target_id"]:
        for d in in_zone:
            if d["id"] == tracking["primary_target_id"]:
                return d

    def priority(d):
        entry_time = tracking["target_entry_times"].get(d["id"], now)
        dist = ((d["cx"] - rx) ** 2 + (d["cy"] - ry) ** 2) ** 0.5
        return (entry_time, dist)

    target = min(in_zone, key=priority)
    tracking["primary_target_id"] = target["id"]
    tracking["locked"] = True
    firing["last_fire_time"] = 0
    return target

# ─── Servo Tracking Thread ───────────────────────────────────────────────────
def servo_tracking_loop():
    while True:
        if not state["armed"] or state["mode"] == "calibration":
            time.sleep(0.05)
            continue
        with detection_lock:
            detections = list(latest_detections)
        target = select_target(detections)
        if target:
            aim_x = target["cx"] + calibration["offset_x"]
            aim_y = target["cy"] + calibration["offset_y"]
            pan, tilt = pixel_to_angle(aim_x, aim_y)
            move_servos(pan, tilt)
        time.sleep(0.05)

# ─── Firing Thread ───────────────────────────────────────────────────────────
def firing_loop():
    while True:
        if not state["armed"] or state["mode"] == "calibration":
            GPIO.output(PUMP_PIN, GPIO.LOW)
            GPIO.output(SOLENOID_PIN, GPIO.LOW)
            firing["active"] = False
            time.sleep(0.1)
            continue

        with detection_lock:
            detections = list(latest_detections)

        target = select_target(detections)
        now = time.time()

        if not target:
            GPIO.output(PUMP_PIN, GPIO.LOW)
            GPIO.output(SOLENOID_PIN, GPIO.LOW)
            firing["active"] = False
            time.sleep(0.1)
            continue

        rx, ry = get_reticle_pos()
        dist = ((target["cx"] - rx) ** 2 + (target["cy"] - ry) ** 2) ** 0.5
        on_target = dist <= state["on_target_tolerance"]

        if not on_target:
            time.sleep(0.05)
            continue

        if state["firing_mode"] == "single":
            if now - firing["last_fire_time"] >= state["reload_time"]:
                _fire_burst()
                firing["last_fire_time"] = time.time()

        elif state["firing_mode"] == "semi_auto":
            if now - firing["last_fire_time"] >= state["semi_auto_delay"]:
                _fire_burst()
                firing["last_fire_time"] = time.time()

        time.sleep(0.05)

def _fire_burst():
    firing["active"] = True
    GPIO.output(PUMP_PIN, GPIO.HIGH)
    GPIO.output(SOLENOID_PIN, GPIO.HIGH)
    time.sleep(state["burst_length"])
    GPIO.output(SOLENOID_PIN, GPIO.LOW)
    GPIO.output(PUMP_PIN, GPIO.LOW)
    firing["active"] = False

# ─── Calibration Firing Thread ───────────────────────────────────────────────
def calibration_loop():
    while True:
        if calibration_state["active"]:
            # Pump runs, solenoid pulses for stream calibration
            GPIO.output(PUMP_PIN, GPIO.HIGH)
            time.sleep(0.5)  # let pump prime
            GPIO.output(SOLENOID_PIN, GPIO.HIGH)
            time.sleep(1.0)
            GPIO.output(SOLENOID_PIN, GPIO.LOW)
            time.sleep(1.0)
            GPIO.output(PUMP_PIN, GPIO.LOW)
            time.sleep(0.5)
        else:
            GPIO.output(PUMP_PIN, GPIO.LOW)
            GPIO.output(SOLENOID_PIN, GPIO.LOW)
            time.sleep(0.1)

# ─── Recording Thread ────────────────────────────────────────────────────────
def recording_loop():
    global latest_frame
    while True:
        with detection_lock:
            cats_detected = len(latest_detections) > 0

        now = time.time()

        if cats_detected:
            recording["last_detection_time"] = now
            if not recording["active"]:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = RECORDINGS_DIR / f"catblastor_{timestamp}.mp4"
                recording["filename"] = str(filename)
                recording["writer"] = cv2.VideoWriter(
                    str(filename),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    15, (FRAME_W, FRAME_H)
                )
                recording["active"] = True

        if recording["active"]:
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None
            if frame is not None:
                recording["writer"].write(frame)
            if not cats_detected and (now - recording["last_detection_time"]) > 10:
                recording["writer"].release()
                recording["writer"] = None
                recording["active"] = False
                recording["filename"] = None

        time.sleep(1/15)

# ─── Start Threads ───────────────────────────────────────────────────────────
threading.Thread(target=capture_loop,        daemon=True).start()
threading.Thread(target=inference_loop,      daemon=True).start()
threading.Thread(target=servo_tracking_loop, daemon=True).start()
threading.Thread(target=firing_loop,         daemon=True).start()
threading.Thread(target=calibration_loop,    daemon=True).start()
threading.Thread(target=recording_loop,      daemon=True).start()
threading.Thread(target=home_position_loop,  daemon=True).start()
threading.Thread(target=start_streaming,     daemon=True).start()

# ─── FastAPI ─────────────────────────────────────────────────────────────────
app = FastAPI()

@app.get("/arm")
def arm():
    state["armed"] = True
    return {"status": "armed"}

@app.get("/disarm")
def disarm():
    state["armed"] = False
    calibration_state["active"] = False
    GPIO.output(PUMP_PIN, GPIO.LOW)
    GPIO.output(SOLENOID_PIN, GPIO.LOW)
    return {"status": "disarmed"}

@app.get("/status")
def status():
    with detection_lock:
        cats = len(latest_detections)
        in_zone_count = len([d for d in latest_detections if d["in_zone"]])
        dets = [
            {
                "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                "conf": round(d["conf"], 2),
                "in_zone": d["in_zone"],
                "is_primary": d["id"] == tracking["primary_target_id"]
            }
            for d in latest_detections
        ]
    # Convert zone angle points to pixel coords for overlay rendering
    zone_px = [world_angle_to_pixel(p[0], p[1]) for p in state["zone_points"]] if state["zone_points"] else []
    return {
        "armed": state["armed"],
        "mode": state["mode"],
        "cats_detected": cats,
        "cats_in_zone": in_zone_count,
        "firing": firing["active"],
        "recording": recording["active"],
        "calibration_active": calibration_state["active"],
        "reticle_x": calibration_state["reticle_x"],
        "reticle_y": calibration_state["reticle_y"],
        "pan": servo_angles["pan"],
        "tilt": servo_angles["tilt"],
        "fov_h": fov["h"],
        "fov_v": fov["v"],
        "detections": dets,
        "zone_px": zone_px,
        "zone_closed": state["zone_closed"],
    }

@app.post("/settings")
async def update_settings(request: Request):
    data = await request.json()
    for key in ["firing_mode", "burst_length", "reload_time",
                "semi_auto_delay", "confidence_threshold", "on_target_tolerance"]:
        if key in data:
            state[key] = data[key]
    return {"status": "updated"}

@app.post("/zone")
async def set_zone(request: Request):
    data = await request.json()
    raw_points = data.get("points", [])
    # Convert pixel coordinates to world angle space using current servo position
    angle_points = []
    for p in raw_points:
        pan, tilt = pixel_to_world_angle(p[0], p[1])
        angle_points.append([pan, tilt])
    state["zone_points"] = angle_points
    state["zone_closed"] = data.get("closed", False)
    return {"status": "zone updated"}

@app.post("/zone/drag")
async def drag_zone(request: Request):
    data = await request.json()
    dx = data.get("dx", 0)
    dy = data.get("dy", 0)
    pan_delta  = data.get("pan_delta",  0)
    tilt_delta = data.get("tilt_delta", 0)

    print(f"Zone drag: dx={dx} dy={dy} pan_delta={pan_delta:.2f} tilt_delta={tilt_delta:.2f} fov={fov}")

    if len(state["zone_points"]) < 3 or not state["zone_closed"]:
        return {"status": "no closed zone"}

    if abs(pan_delta) > 0.5 and abs(dx) > 2:
        new_h = abs(pan_delta) / (abs(dx) / FRAME_W)
        fov["h"] = round(clamp(new_h, 20.0, 120.0), 2)
        print(f"Updated H_FOV to {fov['h']}")

    if abs(tilt_delta) > 0.5 and abs(dy) > 2:
        new_v = abs(tilt_delta) / (abs(dy) / FRAME_H)
        fov["v"] = round(clamp(new_v, 10.0, 80.0), 2)
        print(f"Updated V_FOV to {fov['v']}")

    save_fov()

    pan_shift  = (dx / FRAME_W) * fov["h"]
    tilt_shift = (dy / FRAME_H) * fov["v"]
    state["zone_points"] = [
        [p[0] - pan_shift, p[1] - tilt_shift]
        for p in state["zone_points"]
    ]

    return {"status": "zone dragged", "fov_h": fov["h"], "fov_v": fov["v"]}

@app.get("/fov")
def get_fov():
    return {"h": fov["h"], "v": fov["v"]}

@app.post("/fov")
async def set_fov(request: Request):
    data = await request.json()
    if "h" in data:
        fov["h"] = round(clamp(float(data["h"]), 20.0, 120.0), 2)
    if "v" in data:
        fov["v"] = round(clamp(float(data["v"]), 10.0, 80.0), 2)
    save_fov()
    return {"h": fov["h"], "v": fov["v"]}

@app.get("/zone/clear")
def clear_zone():
    state["zone_points"] = []
    state["zone_closed"] = False
    return {"status": "cleared"}

@app.post("/calibration/reticle")
async def set_reticle(request: Request):
    data = await request.json()
    calibration_state["reticle_x"] = data.get("x", FRAME_W // 2)
    calibration_state["reticle_y"] = data.get("y", FRAME_H // 2)
    calibration["offset_x"] = calibration_state["reticle_x"] - FRAME_W // 2
    calibration["offset_y"] = calibration_state["reticle_y"] - FRAME_H // 2
    save_calibration()
    return {"status": "reticle updated"}

@app.get("/calibration/start")
def calibration_start():
    state["mode"] = "calibration"
    calibration_state["active"] = True
    return {"status": "calibration started"}

@app.get("/calibration/stop")
def calibration_stop():
    state["mode"] = "live"
    calibration_state["active"] = False
    GPIO.output(PUMP_PIN, GPIO.LOW)
    GPIO.output(SOLENOID_PIN, GPIO.LOW)
    return {"status": "calibration stopped"}

@app.get("/mode/{mode}")
def set_mode(mode: str):
    if mode in ["live", "zone_draw", "calibration"]:
        state["mode"] = mode
        if mode != "calibration":
            calibration_state["active"] = False
    return {"mode": state["mode"]}

@app.get("/servos/center")
def center_servos():
    move_servos(90, 90)
    return {"status": "centered"}

@app.post("/servos/move")
async def move_servos_endpoint(request: Request):
    data = await request.json()
    pan  = clamp(servo_angles["pan"]  - data.get("pan_delta",  0), 45, 135)  # inverted
    tilt = clamp(servo_angles["tilt"] + data.get("tilt_delta", 0), 45, 135)
    move_servos(pan, tilt)
    return {"pan": pan, "tilt": tilt}

@app.get("/servos/home/set")
def set_home():
    home_position["pan"]  = servo_angles["pan"]
    home_position["tilt"] = servo_angles["tilt"]
    save_home_position()
    return {"home_pan": home_position["pan"], "home_tilt": home_position["tilt"]}

@app.get("/servos/home/go")
def go_home():
    move_servos(home_position["pan"], home_position["tilt"])
    return {"status": "going home"}

@app.get("/recordings")
def list_recordings():
    files = sorted(RECORDINGS_DIR.glob("*.mp4"), reverse=True)
    return {"recordings": [f.name for f in files]}

@app.get("/recordings/{filename}")
def get_recording(filename: str):
    path = RECORDINGS_DIR / filename
    if path.exists():
        return FileResponse(str(path), media_type="video/mp4")
    return {"error": "not found"}

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML

# ─── Frontend ────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>CatBlastor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #111; color: #eee; font-family: sans-serif; }
  header { background: #1a1a1a; padding: 10px 20px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #333; }
  h1 { color: #ff4444; font-size: 1.4em; }
  nav button { background: none; border: none; color: #aaa; font-size: 1em; padding: 8px 16px; cursor: pointer; border-bottom: 2px solid transparent; }
  nav button.active { color: #fff; border-bottom: 2px solid #ff4444; }
  .page { display: none; padding: 16px; }
  .page.active { display: block; }
  #video-container { position: relative; display: inline-block; }
  #overlay { position: absolute; top: 0; left: 0; cursor: crosshair; }
  .controls { margin: 12px 0; display: flex; flex-wrap: wrap; gap: 8px; }
  button.btn { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9em; }
  .btn-green { background: #2d7a2d; color: white; }
  .btn-red { background: #7a2d2d; color: white; }
  .btn-blue { background: #2d4a7a; color: white; }
  .btn-gray { background: #444; color: white; }
  .btn-yellow { background: #7a6a2d; color: white; }
  .btn.active-mode { outline: 2px solid #ff4444; }
  #status-bar { background: #1a1a1a; border-radius: 4px; padding: 8px 12px; margin: 8px 0; font-size: 0.85em; display: flex; gap: 16px; flex-wrap: wrap; }
  .status-item { display: flex; align-items: center; gap: 6px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #555; }
  .dot.on { background: #44ff44; }
  .dot.warn { background: #ff4444; }
  .dot.rec { background: #ff4444; animation: blink 1s infinite; }
  @keyframes blink { 50% { opacity: 0; } }
  .settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; max-width: 600px; }
  .setting { display: flex; flex-direction: column; gap: 4px; }
  label { font-size: 0.85em; color: #aaa; }
  input[type=range] { width: 100%; }
  select { background: #222; color: #eee; border: 1px solid #444; padding: 4px 8px; border-radius: 4px; }
  .recordings-list { display: flex; flex-direction: column; gap: 8px; max-width: 600px; }
  .recording-item { background: #1a1a1a; border-radius: 4px; padding: 10px; display: flex; justify-content: space-between; align-items: center; }
  .value-label { color: #fff; font-size: 0.9em; }
</style>
</head>
<body>
<header>
  <h1>🐱 CatBlastor</h1>
  <nav>
    <button class="active" onclick="showPage('live', this)">Live</button>
    <button onclick="showPage('calibration', this)">Calibrate</button>
    <button onclick="showPage('settings', this)">Settings</button>
    <button onclick="showPage('recordings', this)">Recordings</button>
  </nav>
</header>

<!-- LIVE PAGE -->
<div id="live" class="page active">
  <div id="status-bar">
    <div class="status-item"><div class="dot" id="dot-armed"></div><span id="txt-armed">Disarmed</span></div>
    <div class="status-item"><div class="dot" id="dot-cats"></div><span id="txt-cats">No cats</span></div>
    <div class="status-item"><div class="dot" id="dot-zone"></div><span id="txt-zone">Zone clear</span></div>
    <div class="status-item"><div class="dot" id="dot-firing"></div><span id="txt-firing">Idle</span></div>
    <div class="status-item"><div class="dot" id="dot-rec"></div><span id="txt-rec">Not recording</span></div>
  </div>

  <div id="video-container">
    <div style="position:relative;display:inline-block">
      <img src="http://catblastor.local:8888/stream" width="640" height="480" style="display:block;background:#000"/>
      <canvas id="overlay" width="640" height="480" style="position:absolute;top:0;left:0;cursor:crosshair"></canvas>
    </div>
  </div>

  <div class="controls">
    <button class="btn btn-green" onclick="fetch('/arm')">ARM</button>
    <button class="btn btn-red" onclick="fetch('/disarm')">DISARM</button>
    <button class="btn btn-blue" id="btn-zone" onclick="toggleMode('zone_draw')">Draw Zone</button>
    <button class="btn btn-gray" onclick="clearZoneLocal()">Clear Zone</button>
    <button class="btn btn-blue" id="btn-close-zone" onclick="closeZone()" style="display:none">Close Zone</button>
    <button class="btn btn-gray" onclick="fetch('/servos/center')">Center Servos</button>
  </div>

  <div class="controls" style="align-items:center">
    <span style="color:#aaa;font-size:0.85em">Pan/Tilt:</span>
    <button class="btn btn-gray" onclick="moveServo(0,-5)">▲ Tilt Up</button>
    <button class="btn btn-gray" onclick="moveServo(0, 5)">▼ Tilt Down</button>
    <button class="btn btn-gray" onclick="moveServo(-5,0)">◀ Pan Left</button>
    <button class="btn btn-gray" onclick="moveServo( 5,0)">▶ Pan Right</button>
    <button class="btn btn-gray" onclick="fetch('/servos/home/go')">🏠 Go Home</button>
    <button class="btn btn-yellow" onclick="fetch('/servos/home/set')">📌 Set Home</button>
  </div>
</div>

<!-- CALIBRATION PAGE -->
<div id="calibration" class="page">
  <div id="video-container-cal">
    <div style="position:relative;display:inline-block">
      <img src="http://catblastor.local:8888/stream" width="640" height="480" style="display:block;background:#000"/>
      <canvas id="overlay-cal" width="640" height="480" style="position:absolute;top:0;left:0;cursor:crosshair"></canvas>
    </div>
  </div>

  <div class="controls">
    <button class="btn btn-blue" id="btn-zone-cal" onclick="toggleMode('zone_draw')">Draw Zone</button>
    <button class="btn btn-gray" onclick="clearZoneLocal()">Clear Zone</button>
    <button class="btn btn-blue" id="btn-close-zone-cal" onclick="closeZone()" style="display:none">Close Zone</button>
    <button class="btn btn-yellow" id="btn-targeting" onclick="toggleTargeting()">▶ Start Targeting</button>
  </div>

  <div class="controls" style="align-items:center">
    <span style="color:#aaa;font-size:0.85em">Pan/Tilt:</span>
    <button class="btn btn-gray" onclick="moveServo(0,-5)">▲ Tilt Up</button>
    <button class="btn btn-gray" onclick="moveServo(0, 5)">▼ Tilt Down</button>
    <button class="btn btn-gray" onclick="moveServo(-5,0)">◀ Pan Left</button>
    <button class="btn btn-gray" onclick="moveServo( 5,0)">▶ Pan Right</button>
    <button class="btn btn-gray" onclick="fetch('/servos/home/go')">🏠 Go Home</button>
    <button class="btn btn-yellow" onclick="fetch('/servos/home/set')">📌 Set Home</button>
  </div>

  <div style="margin-top:12px;padding:12px;background:#1a1a1a;border-radius:4px;max-width:640px">
    <strong style="color:#fff">FOV Calibration</strong>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px">
      <div>
        <label style="color:#aaa;font-size:0.85em">H-FOV (°): <span id="val-hfov" style="color:#fff">66.0</span></label><br>
        <button class="btn btn-gray" style="padding:4px 10px" onclick="adjustFov('h',-1)">−</button>
        <button class="btn btn-gray" style="padding:4px 10px" onclick="adjustFov('h',-0.1)">−0.1</button>
        <button class="btn btn-gray" style="padding:4px 10px" onclick="adjustFov('h',0.1)">+0.1</button>
        <button class="btn btn-gray" style="padding:4px 10px" onclick="adjustFov('h',1)">+</button>
      </div>
      <div>
        <label style="color:#aaa;font-size:0.85em">V-FOV (°): <span id="val-vfov" style="color:#fff">41.0</span></label><br>
        <button class="btn btn-gray" style="padding:4px 10px" onclick="adjustFov('v',-1)">−</button>
        <button class="btn btn-gray" style="padding:4px 10px" onclick="adjustFov('v',-0.1)">−0.1</button>
        <button class="btn btn-gray" style="padding:4px 10px" onclick="adjustFov('v',0.1)">+0.1</button>
        <button class="btn btn-gray" style="padding:4px 10px" onclick="adjustFov('v',1)">+</button>
      </div>
    </div>
    <div style="color:#aaa;font-size:0.75em;margin-top:8px">
      Drag the closed zone to correct its position — adjusts ratio automatically.<br>
      Or use +/− buttons to tune manually.
    </div>
  </div>
</div>

<!-- SETTINGS PAGE -->
<div id="settings" class="page">
  <h2 style="margin-bottom:16px">Settings</h2>
  <div class="settings-grid">
    <div class="setting">
      <label>Firing Mode</label>
      <select id="firing_mode" onchange="updateSettings()">
        <option value="single">Single Fire</option>
        <option value="semi_auto">Semi-Auto</option>
      </select>
    </div>
    <div class="setting">
      <label>Confidence Threshold: <span class="value-label" id="val-conf">0.5</span></label>
      <input type="range" min="0.1" max="1.0" step="0.05" value="0.5" id="confidence_threshold"
             oninput="document.getElementById('val-conf').textContent=this.value" onchange="updateSettings()">
    </div>
    <div class="setting">
      <label>Burst Length (s): <span class="value-label" id="val-burst">1.0</span></label>
      <input type="range" min="0.1" max="5.0" step="0.1" value="1.0" id="burst_length"
             oninput="document.getElementById('val-burst').textContent=this.value" onchange="updateSettings()">
    </div>
    <div class="setting">
      <label>Reload Time (s): <span class="value-label" id="val-reload">10</span></label>
      <input type="range" min="1" max="60" step="1" value="10" id="reload_time"
             oninput="document.getElementById('val-reload').textContent=this.value" onchange="updateSettings()">
    </div>
    <div class="setting">
      <label>Semi-Auto Delay (s): <span class="value-label" id="val-semi">2.0</span></label>
      <input type="range" min="0.5" max="10" step="0.5" value="2.0" id="semi_auto_delay"
             oninput="document.getElementById('val-semi').textContent=this.value" onchange="updateSettings()">
    </div>
    <div class="setting">
      <label>On-Target Tolerance (px): <span class="value-label" id="val-tol">20</span></label>
      <input type="range" min="5" max="100" step="5" value="20" id="on_target_tolerance"
             oninput="document.getElementById('val-tol').textContent=this.value" onchange="updateSettings()">
    </div>
  </div>
</div>

<!-- RECORDINGS PAGE -->
<div id="recordings" class="page">
  <h2 style="margin-bottom:16px">Recordings</h2>
  <button class="btn btn-gray" onclick="loadRecordings()" style="margin-bottom:12px">Refresh</button>
  <div class="recordings-list" id="recordings-list">Loading...</div>
</div>

<script>
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
let zonePoints = [];
let zoneClosed = false;
let currentMode = 'live';
let calibrationMode = false;
let targetingActive = false;
let dragState = null;
let currentPan = 90, currentTilt = 90;

function updateServoAngles(pan, tilt) {
  currentPan = pan;
  currentTilt = tilt;
}

function moveServo(pan_delta, tilt_delta) {
  fetch('/servos/move', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({pan_delta, tilt_delta})
  });
}

function showPage(page, btn) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById(page).classList.add('active');
  btn.classList.add('active');
  calibrationMode = (page === 'calibration');
  if (calibrationMode) {
    fetch('/fov').then(r => r.json()).then(d => {
      document.getElementById('val-hfov').textContent = d.h;
      document.getElementById('val-vfov').textContent = d.v;
    });
  }
  if (!calibrationMode && targetingActive) {
    targetingActive = false;
    fetch('/calibration/stop');
    document.getElementById('btn-targeting').textContent = '▶ Start Targeting';
    document.getElementById('btn-targeting').classList.remove('active-mode');
  }
  fetch('/mode/' + (calibrationMode ? 'calibration' : 'live'));
  if (page === 'recordings') loadRecordings();
}

function toggleMode(mode) {
  const baseMode = calibrationMode ? 'calibration' : 'live';
  if (currentMode === mode) {
    currentMode = baseMode;
    fetch('/mode/' + baseMode);
    ['btn-zone','btn-zone-cal'].forEach(id => { const el = document.getElementById(id); if(el) el.classList.remove('active-mode'); });
    ['btn-close-zone','btn-close-zone-cal'].forEach(id => { const el = document.getElementById(id); if(el) el.style.display = 'none'; });
  } else {
    currentMode = mode;
    fetch('/mode/' + mode);
    ['btn-zone','btn-zone-cal'].forEach(id => { const el = document.getElementById(id); if(el) el.classList.add('active-mode'); });
    ['btn-close-zone','btn-close-zone-cal'].forEach(id => { const el = document.getElementById(id); if(el) el.style.display = 'inline-block'; });
  }
}

function closeZone() {
  if (zonePoints.length < 3) return;
  zoneClosed = true;
  const baseMode = calibrationMode ? 'calibration' : 'live';
  currentMode = baseMode;
  fetch('/mode/' + baseMode);
  ['btn-zone','btn-zone-cal'].forEach(id => { const el = document.getElementById(id); if(el) el.classList.remove('active-mode'); });
  ['btn-close-zone','btn-close-zone-cal'].forEach(id => { const el = document.getElementById(id); if(el) el.style.display = 'none'; });
  // Clear preview canvas — backend will draw the zone from now on
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const calCvs = document.getElementById('overlay-cal');
  if (calCvs) calCvs.getContext('2d').clearRect(0, 0, calCvs.width, calCvs.height);
  sendZone();
}

// ── Calibration canvas drag ──────────────────────────────────────────────────
const calCanvas = document.getElementById('overlay-cal');
if (calCanvas) {
  const calCtx = calCanvas.getContext('2d');

  calCanvas.addEventListener('mousedown', (e) => {
    if (!zoneClosed) return;
    const rect = calCanvas.getBoundingClientRect();
    dragState = {
      startX: Math.round(e.clientX - rect.left),
      startY: Math.round(e.clientY - rect.top),
      panAtStart: currentPan,
      tiltAtStart: currentTilt
    };
  });

  calCanvas.addEventListener('mouseup', (e) => {
    if (!dragState) return;
    const rect = calCanvas.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);
    const dx = x - dragState.startX;
    const dy = y - dragState.startY;
    if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
      fetch('/zone/drag', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          dx, dy,
          pan_delta:  currentPan  - dragState.panAtStart,
          tilt_delta: currentTilt - dragState.tiltAtStart,
        })
      });
    }
    dragState = null;
  });

  calCanvas.addEventListener('click', (e) => {
    if (zoneClosed) return;
    if (currentMode !== 'zone_draw') return;
    const rect = calCanvas.getBoundingClientRect();
    zonePoints.push([
      Math.round(e.clientX - rect.left),
      Math.round(e.clientY - rect.top)
    ]);
  });
}

canvas.addEventListener('click', (e) => {
  if (currentMode !== 'zone_draw' || zoneClosed) return;
  const rect = canvas.getBoundingClientRect();
  zonePoints.push([
    Math.round(e.clientX - rect.left),
    Math.round(e.clientY - rect.top)
  ]);
});

function clearZoneLocal() {
  zonePoints = [];
  zoneClosed = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const calCvs = document.getElementById('overlay-cal');
  if (calCvs) calCvs.getContext('2d').clearRect(0, 0, calCvs.width, calCvs.height);
  fetch('/zone/clear');
}

function sendZone() {
  fetch('/zone', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({points: zonePoints, closed: zoneClosed})
  });
}

function adjustFov(axis, delta) {
  fetch('/fov').then(r => r.json()).then(d => {
    const newVal = Math.round((d[axis] + delta) * 10) / 10;
    const body = {};
    body[axis] = newVal;
    fetch('/fov', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    }).then(r => r.json()).then(d => {
      document.getElementById('val-hfov').textContent = d.h;
      document.getElementById('val-vfov').textContent = d.v;
    });
  });
}

function toggleTargeting() {
  targetingActive = !targetingActive;
  const btn = document.getElementById('btn-targeting');
  if (targetingActive) {
    fetch('/calibration/start');
    btn.textContent = '⏹ Stop Targeting';
    btn.classList.add('active-mode');
  } else {
    fetch('/calibration/stop');
    btn.textContent = '▶ Start Targeting';
    btn.classList.remove('active-mode');
  }
}

function updateSettings() {
  fetch('/settings', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      firing_mode: document.getElementById('firing_mode').value,
      burst_length: parseFloat(document.getElementById('burst_length').value),
      reload_time: parseFloat(document.getElementById('reload_time').value),
      semi_auto_delay: parseFloat(document.getElementById('semi_auto_delay').value),
      confidence_threshold: parseFloat(document.getElementById('confidence_threshold').value),
      on_target_tolerance: parseInt(document.getElementById('on_target_tolerance').value),
    })
  });
}

function drawOverlays(d) {
  // Draw on both canvases
  [canvas, document.getElementById('overlay-cal')].forEach(cvs => {
    if (!cvs) return;
    const c = cvs.getContext('2d');
    c.clearRect(0, 0, cvs.width, cvs.height);

    // Draw zone
    if (d.zone_px && d.zone_px.length >= 2) {
      c.strokeStyle = d.zone_closed ? '#00ff00' : '#00ffff';
      c.lineWidth = 2;
      c.beginPath();
      c.moveTo(d.zone_px[0][0], d.zone_px[0][1]);
      for (let i = 1; i < d.zone_px.length; i++) c.lineTo(d.zone_px[i][0], d.zone_px[i][1]);
      if (d.zone_closed) c.closePath();
      c.stroke();
      d.zone_px.forEach(p => {
        c.fillStyle = d.zone_closed ? '#00ff00' : '#00ffff';
        c.beginPath();
        c.arc(p[0], p[1], 5, 0, 2*Math.PI);
        c.fill();
      });
    }

    // Draw zone preview (in-progress clicks before close)
    if (!zoneClosed && zonePoints.length > 0) {
      c.strokeStyle = '#00ffff';
      c.lineWidth = 2;
      c.beginPath();
      c.moveTo(zonePoints[0][0], zonePoints[0][1]);
      for (let i = 1; i < zonePoints.length; i++) c.lineTo(zonePoints[i][0], zonePoints[i][1]);
      c.stroke();
      zonePoints.forEach(p => {
        c.fillStyle = '#00ffff';
        c.beginPath();
        c.arc(p[0], p[1], 5, 0, 2*Math.PI);
        c.fill();
      });
    }

    // Draw detections
    if (d.detections) {
      d.detections.forEach(det => {
        const color = det.in_zone ? '#ff0000' : '#ff8800';
        c.strokeStyle = color;
        c.lineWidth = det.is_primary ? 3 : 1;
        c.strokeRect(det.x1, det.y1, det.x2-det.x1, det.y2-det.y1);
        c.fillStyle = color;
        c.font = '12px sans-serif';
        c.fillText(`CAT${det.is_primary ? ' [TARGET]' : ''} ${det.conf}`, det.x1, det.y1 - 4);
      });
    }

    // Draw reticle
    const rx = d.reticle_x, ry = d.reticle_y;
    c.strokeStyle = '#ffff00';
    c.lineWidth = 2;
    c.beginPath();
    c.arc(rx, ry, 20, 0, 2*Math.PI);
    c.stroke();
    c.beginPath();
    c.moveTo(rx-30, ry); c.lineTo(rx+30, ry);
    c.moveTo(rx, ry-30); c.lineTo(rx, ry+30);
    c.stroke();

    // REC indicator
    if (d.recording) {
      c.fillStyle = '#ff0000';
      c.beginPath();
      c.arc(20, 20, 8, 0, 2*Math.PI);
      c.fill();
      c.fillStyle = '#ff0000';
      c.font = 'bold 12px sans-serif';
      c.fillText('REC', 32, 25);
    }
  });
}

setInterval(() => {
  fetch('/status').then(r => r.json()).then(d => {
    updateServoAngles(d.pan || 90, d.tilt || 90);
    document.getElementById('dot-armed').className = 'dot ' + (d.armed ? 'on' : '');
    document.getElementById('txt-armed').textContent = d.armed ? 'Armed' : 'Disarmed';
    document.getElementById('dot-cats').className = 'dot ' + (d.cats_detected > 0 ? 'warn' : '');
    document.getElementById('txt-cats').textContent = d.cats_detected > 0 ? d.cats_detected + ' cat(s)' : 'No cats';
    document.getElementById('dot-zone').className = 'dot ' + (d.cats_in_zone > 0 ? 'warn' : '');
    document.getElementById('txt-zone').textContent = d.cats_in_zone > 0 ? d.cats_in_zone + ' in zone' : 'Zone clear';
    document.getElementById('dot-firing').className = 'dot ' + (d.firing ? 'warn' : '');
    document.getElementById('txt-firing').textContent = d.firing ? '💦 FIRING' : 'Idle';
    document.getElementById('dot-rec').className = 'dot ' + (d.recording ? 'rec' : '');
    document.getElementById('txt-rec').textContent = d.recording ? '⏺ Recording' : 'Not recording';
    if (d.fov_h) {
      const hEl = document.getElementById('val-hfov');
      const vEl = document.getElementById('val-vfov');
      if (hEl) hEl.textContent = d.fov_h;
      if (vEl) vEl.textContent = d.fov_v;
    }
    drawOverlays(d);
  });
}, 200);

function loadRecordings() {
  fetch('/recordings').then(r => r.json()).then(d => {
    const list = document.getElementById('recordings-list');
    if (d.recordings.length === 0) {
      list.innerHTML = '<p style="color:#666">No recordings yet</p>';
      return;
    }
    list.innerHTML = d.recordings.map(f => `
      <div class="recording-item">
        <span>${f}</span>
        <a href="/recordings/${f}" target="_blank">
          <button class="btn btn-blue">▶ Play</button>
        </a>
      </div>
    `).join('');
  });
}
</script>
</body>
</html>
"""
