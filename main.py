from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from picamera2 import Picamera2
from ultralytics import YOLO
from adafruit_servokit import ServoKit
import cv2
import threading
import time
import json
import numpy as np
import RPi.GPIO as GPIO
import os
from datetime import datetime
from pathlib import Path

# ─── Constants ───────────────────────────────────────────────────────────────
FRAME_W, FRAME_H = 640, 480
H_FOV, V_FOV = 66.0, 41.0
PAN_CH, TILT_CH = 0, 1
PUMP_PIN, SOLENOID_PIN = 27, 17
CAT_CLASS = 15
RECORDINGS_DIR = Path("/home/wolfhard/catblastor/recordings")
CALIBRATION_FILE = Path("/home/wolfhard/catblastor/calibration.json")
RECORDINGS_DIR.mkdir(exist_ok=True)

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

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "XBGR8888", "size": (FRAME_W, FRAME_H)}
))
picam2.start()

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
    pan  = 90.0 + ((cx - FRAME_W / 2) / FRAME_W) * H_FOV
    tilt = 90.0 + ((cy - FRAME_H / 2) / FRAME_H) * V_FOV
    return clamp(pan, 45, 135), clamp(tilt, 45, 135)

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
def capture_loop():
    global latest_frame
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.rotate(frame, cv2.ROTATE_180)  # camera mounted upside down
        with frame_lock:
            latest_frame = frame.copy()

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
        detections = []
        for r in results[0].boxes:
            cls  = int(r.cls)
            conf = float(r.conf)
            if cls == CAT_CLASS and conf >= state["confidence_threshold"]:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                in_zone = (
                    point_in_polygon(cx, cy, state["zone_points"])
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
        if not state["armed"]:
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
        if not state["armed"]:
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
            GPIO.output(PUMP_PIN, GPIO.HIGH)
            GPIO.output(SOLENOID_PIN, GPIO.HIGH)
            time.sleep(1.0)
            GPIO.output(SOLENOID_PIN, GPIO.LOW)
            time.sleep(1.0)
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

# ─── FastAPI ─────────────────────────────────────────────────────────────────
app = FastAPI()

def generate_frames():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        if len(state["zone_points"]) >= 2:
            pts = np.array(state["zone_points"], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=state["zone_closed"],
                         color=(0, 255, 0), thickness=2)
            for p in state["zone_points"]:
                cv2.circle(frame, tuple(p), 5, (0, 255, 0), -1)

        with detection_lock:
            detections = list(latest_detections)
        for d in detections:
            color = (0, 0, 255) if d["in_zone"] else (0, 165, 255)
            is_primary = d["id"] == tracking["primary_target_id"]
            thickness = 3 if is_primary else 1
            cv2.rectangle(frame, (d["x1"], d["y1"]), (d["x2"], d["y2"]), color, thickness)
            label = f'CAT {"[TARGET]" if is_primary else ""} {d["conf"]:.2f}'
            cv2.putText(frame, label, (d["x1"], d["y1"]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        rx, ry = calibration_state["reticle_x"], calibration_state["reticle_y"]
        cv2.circle(frame, (rx, ry), 20, (0, 255, 255), 2)
        cv2.line(frame, (rx-30, ry), (rx+30, ry), (0, 255, 255), 1)
        cv2.line(frame, (rx, ry-30), (rx, ry+30), (0, 255, 255), 1)

        if recording["active"]:
            cv2.circle(frame, (20, 20), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (32, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.get("/stream")
def stream():
    return StreamingResponse(generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")

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
        in_zone = len([d for d in latest_detections if d["in_zone"]])
    return {
        "armed": state["armed"],
        "mode": state["mode"],
        "cats_detected": cats,
        "cats_in_zone": in_zone,
        "primary_target": tracking["primary_target_id"],
        "firing": firing["active"],
        "recording": recording["active"],
        "calibration_active": calibration_state["active"],
        "reticle_x": calibration_state["reticle_x"],
        "reticle_y": calibration_state["reticle_y"],
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
    state["zone_points"] = data.get("points", [])
    state["zone_closed"] = data.get("closed", False)
    return {"status": "zone updated"}

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
    pan  = clamp(servo_angles["pan"]  + data.get("pan_delta",  0), 45, 135)
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
    <img src="/stream" width="640" height="480"/>
    <canvas id="overlay" width="640" height="480"></canvas>
  </div>

  <div class="controls">
    <button class="btn btn-green" onclick="fetch('/arm')">ARM</button>
    <button class="btn btn-red" onclick="fetch('/disarm')">DISARM</button>
    <button class="btn btn-blue" id="btn-zone" onclick="toggleMode('zone_draw')">Draw Zone</button>
    <button class="btn btn-gray" onclick="clearZoneLocal()">Clear Zone</button>
    <button class="btn btn-blue" id="btn-close-zone" onclick="closeZone()" style="display:none">Close Zone</button>
    <button class="btn btn-yellow" id="btn-cal" onclick="toggleCalibration()">Calibrate</button>
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
let calibrationActive = false;

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
  if (page === 'recordings') loadRecordings();
}

function toggleMode(mode) {
  if (currentMode === mode) {
    currentMode = 'live';
    fetch('/mode/live');
    document.getElementById('btn-zone').classList.remove('active-mode');
    document.getElementById('btn-close-zone').style.display = 'none';
  } else {
    currentMode = mode;
    fetch('/mode/' + mode);
    document.getElementById('btn-zone').classList.add('active-mode');
    document.getElementById('btn-close-zone').style.display = 'inline-block';
  }
}

canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const x = Math.round(e.clientX - rect.left);
  const y = Math.round(e.clientY - rect.top);

  if (calibrationActive) {
    fetch('/calibration/reticle', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({x, y})
    });
    return;
  }

  if (currentMode !== 'zone_draw' || zoneClosed) return;
  zonePoints.push([x, y]);
  drawOverlay();
  sendZone();
});

function closeZone() {
  if (zonePoints.length < 3) return;
  zoneClosed = true;
  currentMode = 'live';
  fetch('/mode/live');
  document.getElementById('btn-zone').classList.remove('active-mode');
  document.getElementById('btn-close-zone').style.display = 'none';
  drawOverlay();
  sendZone();
}

function clearZoneLocal() {
  zonePoints = [];
  zoneClosed = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  fetch('/zone/clear');
}

function sendZone() {
  fetch('/zone', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({points: zonePoints, closed: zoneClosed})
  });
}

function drawOverlay() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (zonePoints.length >= 2) {
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(zonePoints[0][0], zonePoints[0][1]);
    for (let i = 1; i < zonePoints.length; i++) ctx.lineTo(zonePoints[i][0], zonePoints[i][1]);
    if (zoneClosed) ctx.closePath();
    ctx.stroke();
    zonePoints.forEach(p => {
      ctx.fillStyle = '#00ff00';
      ctx.beginPath();
      ctx.arc(p[0], p[1], 5, 0, 2*Math.PI);
      ctx.fill();
    });
  }
}

function toggleCalibration() {
  calibrationActive = !calibrationActive;
  const btn = document.getElementById('btn-cal');
  if (calibrationActive) {
    fetch('/calibration/start');
    btn.classList.add('active-mode');
    btn.textContent = 'Stop Calibration';
    canvas.style.cursor = 'crosshair';
  } else {
    fetch('/calibration/stop');
    btn.classList.remove('active-mode');
    btn.textContent = 'Calibrate';
    canvas.style.cursor = 'crosshair';
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

setInterval(() => {
  fetch('/status').then(r => r.json()).then(d => {
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
  });
}, 500);

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
