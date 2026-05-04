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
PAN_CH, TILT_CH  = 0, 1
PUMP_PIN, SOLENOID_PIN = 27, 17
CAT_CLASS  = 15
PAN_MIN,  PAN_MAX  = 45.0, 135.0
TILT_MIN, TILT_MAX = 45.0, 135.0
PAN_CENTER  = (PAN_MIN  + PAN_MAX)  / 2
TILT_CENTER = (TILT_MIN + TILT_MAX) / 2
PAN_RANGE   = PAN_MAX  - PAN_CENTER
TILT_RANGE  = TILT_MAX - TILT_CENTER

BASE_DIR       = Path("/home/wolfhard/catblastor")
RECORDINGS_DIR = BASE_DIR / "recordings"
ZONE_CAL_FILE  = BASE_DIR / "zone_calibration.json"
HOME_FILE      = BASE_DIR / "home_position.json"
SETTINGS_FILE  = BASE_DIR / "settings.json"
CAM_FILE       = BASE_DIR / "camera_settings.json"
RECORDINGS_DIR.mkdir(exist_ok=True)

# ─── Camera Settings ─────────────────────────────────────────────────────────
DEFAULT_CAM = {"brightness":0.0,"contrast":1.0,"saturation":1.0,"sharpness":1.0,"ev":0.0}
cam_settings = dict(DEFAULT_CAM)
if CAM_FILE.exists():
    cam_settings.update(json.loads(CAM_FILE.read_text()))

def save_cam_settings():
    CAM_FILE.write_text(json.dumps(cam_settings))

# ─── Zone Calibration ────────────────────────────────────────────────────────
zone_cal = {"home_pan":90.0,"home_tilt":90.0,"depth_m":None,"calibration_points":[]}
if ZONE_CAL_FILE.exists():
    zone_cal.update(json.loads(ZONE_CAL_FILE.read_text()))

def save_zone_cal():
    ZONE_CAL_FILE.write_text(json.dumps(zone_cal))

def interpolate_zone(pan, tilt):
    pts = zone_cal["calibration_points"]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]["vertices"]
    distances = [max(np.sqrt((pan-p["pan"])**2+(tilt-p["tilt"])**2),0.001) for p in pts]
    min_d = min(distances)
    if min_d < 0.1:
        return pts[distances.index(min_d)]["vertices"]
    weights = [1.0/d**2 for d in distances]
    total_w = sum(weights)
    weights = [w/total_w for w in weights]
    n_verts = len(pts[0]["vertices"])
    result = []
    for i in range(n_verts):
        x = sum(weights[j]*pts[j]["vertices"][i][0] for j in range(len(pts)))
        y = sum(weights[j]*pts[j]["vertices"][i][1] for j in range(len(pts)))
        if zone_cal["depth_m"] and zone_cal["depth_m"] > 0:
            d = zone_cal["depth_m"]
            dpan  = np.radians(pan  - zone_cal["home_pan"])
            dtilt = np.radians(tilt - zone_cal["home_tilt"])
            x -= np.tan(dpan)  * d * (FRAME_W / 66.0)
            y -= np.tan(dtilt) * d * (FRAME_H / 41.0)
        result.append([int(round(x)), int(round(y))])
    return result

def compute_calibration_strength():
    pts = zone_cal["calibration_points"]
    n = len(pts)
    if n < 3:
        return {"position_confidence":None,"coverage_score":None,"combined":None,"n_points":n}
    errors = []
    for i, p in enumerate(pts):
        remaining = [pts[j] for j in range(n) if j != i]
        original = zone_cal["calibration_points"]
        zone_cal["calibration_points"] = remaining
        predicted = interpolate_zone(p["pan"], p["tilt"])
        zone_cal["calibration_points"] = original
        if predicted is None or len(predicted) != len(p["vertices"]):
            continue
        act_cx = np.mean([v[0] for v in p["vertices"]])
        act_cy = np.mean([v[1] for v in p["vertices"]])
        pre_cx = np.mean([v[0] for v in predicted])
        pre_cy = np.mean([v[1] for v in predicted])
        err = np.sqrt((act_cx-pre_cx)**2+(act_cy-pre_cy)**2)
        xs = [v[0] for v in p["vertices"]]; ys = [v[1] for v in p["vertices"]]
        diam = max(np.sqrt((max(xs)-min(xs))**2+(max(ys)-min(ys))**2), 1.0)
        errors.append(err/diam)
    if not errors:
        return {"position_confidence":None,"coverage_score":None,"combined":None,"n_points":n}
    pos_conf = max(0.0, 1.0 - np.mean(errors)) * 100.0
    pans  = [p["pan"]  for p in pts]; tilts = [p["tilt"] for p in pts]
    cov   = ((max(pans)-min(pans))/(PAN_MAX-PAN_MIN)*0.5 +
             (max(tilts)-min(tilts))/(TILT_MAX-TILT_MIN)*0.5) * 100.0
    raw   = pos_conf*0.7 + cov*0.3
    combined = min(raw, 50.0) if n < 8 else raw
    return {"position_confidence":round(pos_conf,1),"coverage_score":round(cov,1),
            "combined":round(combined,1),"n_points":n}

# ─── MJPEG Streaming ─────────────────────────────────────────────────────────
mjpeg_buffer = b""
mjpeg_lock   = threading.Lock()
rpicam_proc  = None

def build_rpicam_cmd():
    return [
        "rpicam-vid","--width","640","--height","480","--framerate","30",
        "--codec","mjpeg","--inline","-o","-","-t","0","--nopreview",
        "--vflip","--hflip",
        "--brightness", str(cam_settings["brightness"]),
        "--contrast",   str(cam_settings["contrast"]),
        "--saturation", str(cam_settings["saturation"]),
        "--sharpness",  str(cam_settings["sharpness"]),
        "--ev",         str(cam_settings["ev"]),
    ]

def start_streaming():
    global rpicam_proc
    rpicam_proc = subprocess.Popen(build_rpicam_cmd(),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    print("Streaming started")

def restart_rpicam():
    global rpicam_proc
    if rpicam_proc:
        rpicam_proc.terminate()
        rpicam_proc.wait()
    rpicam_proc = subprocess.Popen(build_rpicam_cmd(),
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

def mjpeg_reader_loop():
    global mjpeg_buffer
    while rpicam_proc is None:
        time.sleep(0.1)
    buf = b""
    SOI, EOI = b'\xff\xd8', b'\xff\xd9'
    while True:
        try:
            chunk = rpicam_proc.stdout.read(65536)
            if not chunk:
                time.sleep(0.1); continue
            buf += chunk
            while True:
                s = buf.find(SOI); e = buf.find(EOI, s+2)
                if s == -1 or e == -1: break
                with mjpeg_lock:
                    mjpeg_buffer = buf[s:e+2]
                buf = buf[e+2:]
        except Exception as ex:
            print(f"MJPEG reader: {ex}"); time.sleep(0.1)

# ─── GPIO ────────────────────────────────────────────────────────────────────
GPIO.setmode(GPIO.BCM)
GPIO.setup(PUMP_PIN,     GPIO.OUT); GPIO.output(PUMP_PIN,     GPIO.LOW)
GPIO.setup(SOLENOID_PIN, GPIO.OUT); GPIO.output(SOLENOID_PIN, GPIO.LOW)

# ─── Servo Hardware ──────────────────────────────────────────────────────────
kit = ServoKit(channels=16)
kit.servo[PAN_CH].angle  = 90
kit.servo[TILT_CH].angle = 90

# ─── YOLO ────────────────────────────────────────────────────────────────────
model = YOLO("yolov8n.pt")

# ─── App State ───────────────────────────────────────────────────────────────
state = {
    "armed":False,"firing_mode":"single","burst_length":1.0,
    "reload_time":10.0,"semi_auto_delay":2.0,"confidence_threshold":0.5,
    "on_target_tolerance":20,"setup_phase":None,
}
if SETTINGS_FILE.exists():
    saved = json.loads(SETTINGS_FILE.read_text())
    for k in ["firing_mode","burst_length","reload_time","semi_auto_delay",
               "confidence_threshold","on_target_tolerance"]:
        if k in saved: state[k] = saved[k]

def save_settings():
    SETTINGS_FILE.write_text(json.dumps({k:state[k] for k in
        ["firing_mode","burst_length","reload_time","semi_auto_delay",
         "confidence_threshold","on_target_tolerance"]}))

# ─── Servo State ─────────────────────────────────────────────────────────────
servo_angles = {"pan":90.0,"tilt":90.0}
home_position = {"pan":90.0,"tilt":90.0}
if HOME_FILE.exists():
    home_position.update(json.loads(HOME_FILE.read_text()))
last_activity_time = time.time()
HOME_TIMEOUT = 60.0

def save_home_position():
    HOME_FILE.write_text(json.dumps(home_position))

def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def move_servos(pan, tilt, slow=False):
    global last_activity_time
    pan  = clamp(pan,  PAN_MIN,  PAN_MAX)
    tilt = clamp(tilt, TILT_MIN, TILT_MAX)
    if slow:
        cp, ct = servo_angles["pan"], servo_angles["tilt"]
        steps = max(int(abs(pan-cp)/2), int(abs(tilt-ct)/2), 1)
        for i in range(1, steps+1):
            p = cp + (pan-cp)*i/steps; t = ct + (tilt-ct)*i/steps
            kit.servo[PAN_CH].angle  = p
            kit.servo[TILT_CH].angle = t
            servo_angles["pan"] = p; servo_angles["tilt"] = t
            time.sleep(0.02)
    else:
        kit.servo[PAN_CH].angle  = pan
        kit.servo[TILT_CH].angle = tilt
        servo_angles["pan"] = pan; servo_angles["tilt"] = tilt
    last_activity_time = time.time()

def servo_pct(angle, center, rang):
    return round((angle-center)/rang*100.0, 1)

# ─── Zone Runtime State ──────────────────────────────────────────────────────
zone_vertices = []
zone_closed   = False

def get_setup_zone():
    """Return exact saved zone for current servo position, or interpolated if not saved."""
    pan, tilt = servo_angles["pan"], servo_angles["tilt"]
    pts = zone_cal["calibration_points"]
    if not pts:
        return zone_vertices, zone_closed, False
    # Check for exact saved point at current position
    exact = next((p for p in pts
                  if abs(p["pan"]-pan)<0.5 and abs(p["tilt"]-tilt)<0.5), None)
    if exact:
        return exact["vertices"], True, True  # vertices, closed, is_exact
    # Interpolate
    interp = interpolate_zone(pan, tilt)
    if interp:
        return interp, True, False
    return zone_vertices, zone_closed, False

def get_current_zone():
    verts, closed, _ = get_setup_zone()
    return verts, closed

def point_in_zone(cx, cy):
    verts, closed = get_current_zone()
    if not closed or len(verts) < 3: return False
    pts = np.array(verts, dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0

# ─── Detection State ─────────────────────────────────────────────────────────
frame_lock        = threading.Lock()
latest_frame      = None
latest_detections = []
detection_lock    = threading.Lock()

# ─── Tracking & Firing ───────────────────────────────────────────────────────
tracking = {"primary_target_id":None,"target_entry_times":{},"last_seen":{}}
firing   = {"active":False,"last_fire_time":0}
targeting_active = False

# ─── Recording ───────────────────────────────────────────────────────────────
recording = {"active":False,"writer":None,"last_detection_time":0,"filename":None}

# ─── Threads ─────────────────────────────────────────────────────────────────
def capture_loop():
    global latest_frame
    while True:
        with mjpeg_lock:
            fd = mjpeg_buffer
        if fd:
            try:
                arr = np.frombuffer(fd, dtype=np.uint8)
                f   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if f is not None:
                    f = cv2.resize(f, (INFER_W, INFER_H))
                    with frame_lock:
                        latest_frame = f.copy()
            except Exception:
                pass
        time.sleep(1/10)

def inference_loop():
    global latest_detections
    while True:
        if not state["armed"]:
            latest_detections = []; time.sleep(0.1); continue
        with frame_lock:
            if latest_frame is None: time.sleep(0.05); continue
            frame = latest_frame.copy()
        results = model(frame, verbose=False)
        sx, sy = FRAME_W/INFER_W, FRAME_H/INFER_H
        dets = []
        for r in results[0].boxes:
            cls = int(r.cls); conf = float(r.conf)
            if cls == CAT_CLASS and conf >= state["confidence_threshold"]:
                x1,y1,x2,y2 = map(int, r.xyxy[0])
                x1,y1,x2,y2 = int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
                cx,cy = (x1+x2)//2,(y1+y2)//2
                dets.append({"id":f"{cx}_{cy}","x1":x1,"y1":y1,"x2":x2,"y2":y2,
                             "cx":cx,"cy":cy,"conf":conf,"in_zone":point_in_zone(cx,cy)})
        with detection_lock:
            latest_detections = dets

def select_target(detections):
    in_zone = [d for d in detections if d["in_zone"]]
    if not in_zone:
        tracking["primary_target_id"] = None; return None
    now = time.time()
    for d in in_zone:
        if d["id"] not in tracking["target_entry_times"]:
            tracking["target_entry_times"][d["id"]] = now
        tracking["last_seen"][d["id"]] = now
    cur_ids = {d["id"] for d in in_zone}
    for tid in list(tracking["target_entry_times"]):
        if tid not in cur_ids:
            tracking["target_entry_times"].pop(tid,None)
            tracking["last_seen"].pop(tid,None)
            if tracking["primary_target_id"] == tid:
                tracking["primary_target_id"] = None; firing["last_fire_time"] = 0
    if tracking["primary_target_id"]:
        for d in in_zone:
            if d["id"] == tracking["primary_target_id"]: return d
    def pri(d):
        return (tracking["target_entry_times"].get(d["id"],now),
                np.sqrt((d["cx"]-FRAME_W//2)**2+(d["cy"]-FRAME_H//2)**2))
    t = min(in_zone, key=pri)
    tracking["primary_target_id"] = t["id"]; firing["last_fire_time"] = 0
    return t

def servo_tracking_loop():
    while True:
        if not state["armed"] or state["setup_phase"] is not None:
            time.sleep(0.05); continue
        with detection_lock:
            dets = list(latest_detections)
        t = select_target(dets)
        if t:
            pan  = servo_angles["pan"]  - ((t["cx"]-FRAME_W/2)/FRAME_W)*66.0
            tilt = servo_angles["tilt"] - ((t["cy"]-FRAME_H/2)/FRAME_H)*41.0
            move_servos(pan, tilt)
        time.sleep(0.05)

def firing_loop():
    while True:
        if not state["armed"] or state["setup_phase"] is not None:
            GPIO.output(PUMP_PIN,GPIO.LOW); GPIO.output(SOLENOID_PIN,GPIO.LOW)
            firing["active"] = False; time.sleep(0.1); continue
        with detection_lock:
            dets = list(latest_detections)
        t = select_target(dets); now = time.time()
        if not t:
            GPIO.output(PUMP_PIN,GPIO.LOW); GPIO.output(SOLENOID_PIN,GPIO.LOW)
            firing["active"] = False; time.sleep(0.1); continue
        dist = np.sqrt((t["cx"]-FRAME_W//2)**2+(t["cy"]-FRAME_H//2)**2)
        if dist > state["on_target_tolerance"]:
            time.sleep(0.05); continue
        if state["firing_mode"] == "single":
            if now - firing["last_fire_time"] >= state["reload_time"]:
                _fire_burst(); firing["last_fire_time"] = time.time()
        elif state["firing_mode"] == "semi_auto":
            if now - firing["last_fire_time"] >= state["semi_auto_delay"]:
                _fire_burst(); firing["last_fire_time"] = time.time()
        time.sleep(0.05)

def _fire_burst():
    firing["active"] = True
    GPIO.output(PUMP_PIN,GPIO.HIGH); GPIO.output(SOLENOID_PIN,GPIO.HIGH)
    time.sleep(state["burst_length"])
    GPIO.output(SOLENOID_PIN,GPIO.LOW); GPIO.output(PUMP_PIN,GPIO.LOW)
    firing["active"] = False

def targeting_loop():
    while True:
        if targeting_active:
            GPIO.output(PUMP_PIN,GPIO.HIGH); time.sleep(0.5)
            GPIO.output(SOLENOID_PIN,GPIO.HIGH); time.sleep(1.0)
            GPIO.output(SOLENOID_PIN,GPIO.LOW); time.sleep(1.0)
            GPIO.output(PUMP_PIN,GPIO.LOW); time.sleep(0.5)
        else:
            GPIO.output(PUMP_PIN,GPIO.LOW); GPIO.output(SOLENOID_PIN,GPIO.LOW)
            time.sleep(0.1)

def recording_loop():
    while True:
        with detection_lock:
            cats = len(latest_detections) > 0
        now = time.time()
        if cats:
            recording["last_detection_time"] = now
            if not recording["active"]:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = RECORDINGS_DIR / f"catblastor_{ts}.mp4"
                recording["filename"] = str(fn)
                recording["writer"] = cv2.VideoWriter(str(fn),
                    cv2.VideoWriter_fourcc(*"mp4v"), 15, (FRAME_W,FRAME_H))
                recording["active"] = True
        if recording["active"]:
            with mjpeg_lock:
                fd = mjpeg_buffer
            if fd:
                try:
                    arr = np.frombuffer(fd, dtype=np.uint8)
                    f   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if f is not None: recording["writer"].write(f)
                except Exception: pass
            if not cats and (now-recording["last_detection_time"]) > 10:
                recording["writer"].release()
                recording["writer"] = None; recording["active"] = False
        time.sleep(1/15)

def home_position_loop():
    while True:
        time.sleep(5)
        if not state["armed"] or state["setup_phase"] is not None: continue
        with detection_lock:
            cats = len(latest_detections)
        if cats == 0 and not firing["active"]:
            if time.time() - last_activity_time > HOME_TIMEOUT:
                move_servos(home_position["pan"], home_position["tilt"])

# ─── Start Threads ───────────────────────────────────────────────────────────
threading.Thread(target=start_streaming,     daemon=True).start()
threading.Thread(target=mjpeg_reader_loop,   daemon=True).start()
threading.Thread(target=capture_loop,        daemon=True).start()
threading.Thread(target=inference_loop,      daemon=True).start()
threading.Thread(target=servo_tracking_loop, daemon=True).start()
threading.Thread(target=firing_loop,         daemon=True).start()
threading.Thread(target=targeting_loop,      daemon=True).start()
threading.Thread(target=recording_loop,      daemon=True).start()
threading.Thread(target=home_position_loop,  daemon=True).start()

# ─── FastAPI ─────────────────────────────────────────────────────────────────
app = FastAPI()

@app.get("/stream")
def stream():
    def generate():
        while True:
            with mjpeg_lock:
                frame = mjpeg_buffer
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(1/30)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/arm")
def arm():
    state["armed"] = True; return {"status":"armed"}

@app.get("/disarm")
def disarm():
    global targeting_active
    state["armed"] = False; targeting_active = False
    GPIO.output(PUMP_PIN,GPIO.LOW); GPIO.output(SOLENOID_PIN,GPIO.LOW)
    return {"status":"disarmed"}

@app.get("/status")
def status():
    with detection_lock:
        cats = len(latest_detections)
        in_z = len([d for d in latest_detections if d["in_zone"]])
        dets = [{"x1":d["x1"],"y1":d["y1"],"x2":d["x2"],"y2":d["y2"],
                 "conf":round(d["conf"],2),"in_zone":d["in_zone"],
                 "is_primary":d["id"]==tracking["primary_target_id"]}
                for d in latest_detections]
    zone_px, z_closed = get_current_zone()
    setup_verts, setup_closed, is_exact = get_setup_zone()
    strength = compute_calibration_strength()
    return {
        "armed":state["armed"],"setup_phase":state["setup_phase"],
        "cats_detected":cats,"cats_in_zone":in_z,
        "firing":firing["active"],"recording":recording["active"],
        "targeting_active":targeting_active,
        "pan":round(servo_angles["pan"],1),"tilt":round(servo_angles["tilt"],1),
        "pan_pct":servo_pct(servo_angles["pan"],PAN_CENTER,PAN_RANGE),
        "tilt_pct":servo_pct(servo_angles["tilt"],TILT_CENTER,TILT_RANGE),
        "detections":dets,"zone_px":zone_px,"zone_closed":z_closed,
        "setup_zone":setup_verts,"setup_zone_closed":setup_closed,"setup_zone_exact":is_exact,
        "n_cal_points":len(zone_cal["calibration_points"]),
        "strength":strength,
        "cam":cam_settings,"depth_m":zone_cal["depth_m"],
    }

@app.post("/settings")
async def update_settings(request: Request):
    data = await request.json()
    for k in ["firing_mode","burst_length","reload_time","semi_auto_delay",
               "confidence_threshold","on_target_tolerance"]:
        if k in data: state[k] = data[k]
    save_settings(); return {"status":"updated"}

@app.post("/camera")
async def update_camera(request: Request):
    data = await request.json()
    for k in ["brightness","contrast","saturation","sharpness","ev"]:
        if k in data: cam_settings[k] = float(data[k])
    save_cam_settings()
    threading.Thread(target=restart_rpicam, daemon=True).start()
    return cam_settings

@app.post("/servos/move")
async def move_servos_ep(request: Request):
    data = await request.json()
    pan  = clamp(servo_angles["pan"]  - data.get("pan_delta",0),  PAN_MIN,  PAN_MAX)
    tilt = clamp(servo_angles["tilt"] + data.get("tilt_delta",0), TILT_MIN, TILT_MAX)
    move_servos(pan, tilt); return {"pan":pan,"tilt":tilt}

@app.get("/servos/center")
def center_servos():
    move_servos(PAN_CENTER,TILT_CENTER); return {"status":"centered"}

@app.get("/servos/home/set")
def set_home():
    home_position["pan"]  = servo_angles["pan"]
    home_position["tilt"] = servo_angles["tilt"]
    save_home_position(); return home_position

@app.get("/servos/home/go")
def go_home():
    move_servos(home_position["pan"],home_position["tilt"]); return {"status":"going home"}

# ─── Setup Endpoints ──────────────────────────────────────────────────────────
@app.get("/setup/start")
def setup_start():
    state["setup_phase"] = "home"; return {"phase":"home"}

@app.get("/setup/set_home")
def setup_set_home():
    zone_cal["home_pan"]  = servo_angles["pan"]
    zone_cal["home_tilt"] = servo_angles["tilt"]
    home_position["pan"]  = servo_angles["pan"]
    home_position["tilt"] = servo_angles["tilt"]
    save_home_position(); state["setup_phase"] = "draw"
    return {"phase":"draw"}

@app.post("/setup/zone")
async def setup_zone(request: Request):
    global zone_vertices, zone_closed
    data = await request.json()
    zone_vertices = data.get("vertices",[])
    zone_closed   = data.get("closed",False)
    return {"status":"ok","n_verts":len(zone_vertices)}

@app.get("/setup/begin_forced_cal")
def begin_forced_cal():
    if zone_closed and len(zone_vertices) >= 3:
        zone_cal["calibration_points"] = [{
            "pan":zone_cal["home_pan"],"tilt":zone_cal["home_tilt"],
            "vertices":[list(v) for v in zone_vertices]}]
        save_zone_cal()
    state["setup_phase"] = "forced_L"
    def go():
        time.sleep(0.5)
        move_servos(PAN_MAX, zone_cal["home_tilt"], slow=True)  # PAN_MAX = visual left (camera flipped)
    threading.Thread(target=go, daemon=True).start()
    return {"phase":"forced_L"}

@app.post("/setup/save_forced_point")
async def save_forced_point(request: Request):
    data     = await request.json()
    vertices = data.get("vertices", zone_vertices)
    phase    = state["setup_phase"]
    existing = next((p for p in zone_cal["calibration_points"]
                     if abs(p["pan"]-servo_angles["pan"])<0.5 and
                        abs(p["tilt"]-servo_angles["tilt"])<0.5), None)
    point = {"pan":servo_angles["pan"],"tilt":servo_angles["tilt"],
             "vertices":[list(v) for v in vertices]}
    if existing: zone_cal["calibration_points"].remove(existing)
    zone_cal["calibration_points"].append(point)
    save_zone_cal()
    next_map = {"forced_L":"forced_R","forced_R":"forced_up",
                "forced_up":"forced_down","forced_down":"extra"}
    next_phase = next_map.get(phase,"extra")
    state["setup_phase"] = next_phase
    def go():
        time.sleep(0.3)
        if phase == "forced_L":   move_servos(PAN_MIN,             zone_cal["home_tilt"], slow=True)  # PAN_MIN = visual right
        elif phase == "forced_R": move_servos(zone_cal["home_pan"], TILT_MIN,              slow=True)
        elif phase == "forced_up":move_servos(zone_cal["home_pan"], TILT_MAX,              slow=True)
        elif phase == "forced_down": move_servos(zone_cal["home_pan"], zone_cal["home_tilt"], slow=True)
    threading.Thread(target=go, daemon=True).start()
    return {"phase":next_phase}

@app.post("/setup/add_extra_point")
async def add_extra_point(request: Request):
    data     = await request.json()
    vertices = data.get("vertices", zone_vertices)
    existing = next((p for p in zone_cal["calibration_points"]
                     if abs(p["pan"]-servo_angles["pan"])<0.5 and
                        abs(p["tilt"]-servo_angles["tilt"])<0.5), None)
    point = {"pan":servo_angles["pan"],"tilt":servo_angles["tilt"],
             "vertices":[list(v) for v in vertices]}
    if existing: zone_cal["calibration_points"].remove(existing)
    zone_cal["calibration_points"].append(point)
    save_zone_cal()
    return {"n_points":len(zone_cal["calibration_points"]),
            "strength":compute_calibration_strength()}

@app.post("/setup/remove_point")
async def remove_point(request: Request):
    data = await request.json()
    idx  = data.get("index",-1)
    if 0 <= idx < len(zone_cal["calibration_points"]):
        zone_cal["calibration_points"].pop(idx)
        save_zone_cal()
    return {"n_points":len(zone_cal["calibration_points"]),
            "strength":compute_calibration_strength()}

@app.post("/setup/goto_point")
async def goto_point(request: Request):
    data = await request.json(); idx = data.get("index",0)
    if 0 <= idx < len(zone_cal["calibration_points"]):
        p = zone_cal["calibration_points"][idx]
        move_servos(p["pan"], p["tilt"], slow=True)
    return {"status":"ok"}

@app.get("/setup/cal_points")
def get_cal_points():
    return {"points":[{"index":i,"pan":round(p["pan"],1),"tilt":round(p["tilt"],1)}
                      for i,p in enumerate(zone_cal["calibration_points"])]}

@app.post("/setup/depth")
async def set_depth(request: Request):
    data = await request.json()
    zone_cal["depth_m"] = data.get("depth_m",None)
    save_zone_cal(); return {"depth_m":zone_cal["depth_m"]}

@app.get("/setup/finish")
def setup_finish():
    state["setup_phase"] = None; return {"status":"complete"}

@app.get("/setup/reset")
def setup_reset():
    global zone_vertices, zone_closed
    zone_cal["calibration_points"] = []
    zone_cal["home_pan"]  = 90.0; zone_cal["home_tilt"] = 90.0
    zone_cal["depth_m"]   = None
    zone_vertices = []; zone_closed = False
    save_zone_cal(); state["setup_phase"] = "home"
    move_servos(PAN_CENTER, TILT_CENTER)
    return {"status":"reset","phase":"home"}

@app.get("/setup/targeting/start")
def targeting_start():
    global targeting_active
    targeting_active = True; return {"targeting":True}

@app.get("/setup/targeting/stop")
def targeting_stop():
    global targeting_active
    targeting_active = False
    GPIO.output(PUMP_PIN,GPIO.LOW); GPIO.output(SOLENOID_PIN,GPIO.LOW)
    return {"targeting":False}

@app.get("/recordings")
def list_recordings():
    files = sorted(RECORDINGS_DIR.glob("*.mp4"), reverse=True)
    return {"recordings":[f.name for f in files]}

@app.get("/recordings/{filename}")
def get_recording(filename: str):
    path = RECORDINGS_DIR / filename
    if path.exists(): return FileResponse(str(path), media_type="video/mp4")
    return {"error":"not found"}

@app.get("/", response_class=HTMLResponse)
def index(): return HTML

# ─── Frontend ────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html>
<head>
<title>CatBlastor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#111;color:#eee;font-family:sans-serif;}
header{background:#1a1a1a;padding:10px 20px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #333;}
h1{color:#ff4444;font-size:1.4em;}
nav button{background:none;border:none;color:#aaa;font-size:1em;padding:8px 16px;cursor:pointer;border-bottom:2px solid transparent;}
nav button.active{color:#fff;border-bottom:2px solid #ff4444;}
.page{display:none;padding:16px;}.page.active{display:block;}
.vw{position:relative;display:inline-block;}
.vw canvas{position:absolute;top:0;left:0;}
.ctrl{margin:8px 0;display:flex;flex-wrap:wrap;gap:8px;align-items:center;}
button.btn{padding:7px 14px;border:none;border-radius:4px;cursor:pointer;font-size:0.88em;}
.g{background:#2d7a2d;color:#fff;}.r{background:#7a2d2d;color:#fff;}
.b{background:#2d4a7a;color:#fff;}.y{background:#7a6a2d;color:#fff;}
.gr{background:#444;color:#fff;}.btn.on{outline:2px solid #ff4444;}
#sb{background:#1a1a1a;border-radius:4px;padding:8px 12px;margin:8px 0;font-size:0.85em;display:flex;gap:16px;flex-wrap:wrap;}
.si{display:flex;align-items:center;gap:6px;}
.dot{width:8px;height:8px;border-radius:50%;background:#555;}
.dot.on{background:#44ff44;}.dot.warn{background:#ff4444;}
.dot.rec{background:#ff4444;animation:blink 1s infinite;}
@keyframes blink{50%{opacity:0;}}
.sbar-wrap{display:flex;align-items:center;gap:6px;font-size:0.82em;}
.sbar{width:110px;height:10px;background:#333;border-radius:5px;position:relative;}
.sind{position:absolute;width:4px;height:10px;background:#4af;border-radius:2px;transform:translateX(-50%);}
.sg{display:grid;grid-template-columns:1fr 1fr;gap:14px;max-width:600px;}
.set{display:flex;flex-direction:column;gap:4px;}
label{font-size:0.85em;color:#aaa;}
input[type=range]{width:100%;}
input[type=number]{width:64px;background:#222;color:#eee;border:1px solid #444;padding:2px 4px;border-radius:3px;}
select{background:#222;color:#eee;border:1px solid #444;padding:4px 8px;border-radius:4px;}
.vl{color:#fff;font-size:0.9em;}
.rl{display:flex;flex-direction:column;gap:8px;max-width:600px;}
.ri{background:#1a1a1a;border-radius:4px;padding:10px;display:flex;justify-content:space-between;align-items:center;}
.ph{background:#1a1a2a;border:1px solid #334;border-radius:6px;padding:12px;margin:8px 0;max-width:660px;}
.ph h3{color:#aaf;margin-bottom:6px;font-size:1em;}
.ph p{color:#aaa;font-size:0.85em;margin-bottom:8px;}
.sbar2{height:16px;background:#333;border-radius:8px;overflow:hidden;margin:4px 0;}
.sf{height:100%;background:linear-gradient(to right,#f44,#fa0,#4f4);border-radius:8px;transition:width .5s;}
.cpl{background:#1a1a1a;border-radius:4px;padding:7px 12px;display:flex;justify-content:space-between;align-items:center;margin:3px 0;cursor:pointer;}
.cpl:hover{background:#252525;}
</style>
</head>
<body>
<header>
  <h1>🐱 CatBlastor</h1>
  <nav>
    <button class="active" onclick="showPage('live',this)">Live</button>
    <button onclick="showPage('setup',this)">Setup</button>
    <button onclick="showPage('settings',this)">Settings</button>
    <button onclick="showPage('recordings',this)">Recordings</button>
  </nav>
</header>

<!-- LIVE -->
<div id="live" class="page active">
  <div id="sb">
    <div class="si"><div class="dot" id="da"></div><span id="ta">Disarmed</span></div>
    <div class="si"><div class="dot" id="dc"></div><span id="tc">No cats</span></div>
    <div class="si"><div class="dot" id="dz"></div><span id="tz">Zone clear</span></div>
    <div class="si"><div class="dot" id="df"></div><span id="tf">Idle</span></div>
    <div class="si"><div class="dot" id="dr"></div><span id="tr">Not recording</span></div>
  </div>
  <div class="vw">
    <img src="/stream" width="640" height="480" style="display:block;background:#000">
    <canvas id="ov-live" width="640" height="480" style="cursor:default"></canvas>
  </div>
  <div class="ctrl">
    <button class="btn g" onclick="fetch('/arm')">ARM</button>
    <button class="btn r" onclick="fetch('/disarm')">DISARM</button>
    <button class="btn gr" onclick="fetch('/servos/center')">Center</button>
    <button class="btn gr" onclick="fetch('/servos/home/go')">🏠 Home</button>
    <button class="btn gr" onclick="fetch('/servos/home/set')">📌 Set Home</button>
  </div>
  <div class="ctrl">
    <span style="color:#aaa;font-size:0.85em">Pan/Tilt:</span>
    <button class="btn gr" onclick="mv(0,-5)">▲</button>
    <button class="btn gr" onclick="mv(0,5)">▼</button>
    <button class="btn gr" onclick="mv(-5,0)">◀</button>
    <button class="btn gr" onclick="mv(5,0)">▶</button>
  </div>
  <div class="ctrl" style="gap:20px">
    <div class="sbar-wrap">Pan:<span id="pp">0%</span><div class="sbar"><div class="sind" id="pi" style="left:50%"></div></div></div>
    <div class="sbar-wrap">Tilt:<span id="tp">0%</span><div class="sbar"><div class="sind" id="ti" style="left:50%"></div></div></div>
  </div>
</div>

<!-- SETUP -->
<div id="setup" class="page">
  <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
    <div>
      <div class="vw">
        <img src="/stream" width="640" height="480" style="display:block;background:#000">
        <canvas id="ov-setup" width="640" height="480" style="cursor:crosshair"></canvas>
      </div>
      <div class="ctrl" style="gap:20px;margin-top:6px">
        <div class="sbar-wrap">Pan:<span id="pp-s">0%</span><div class="sbar"><div class="sind" id="pi-s" style="left:50%"></div></div></div>
        <div class="sbar-wrap">Tilt:<span id="tp-s">0%</span><div class="sbar"><div class="sind" id="ti-s" style="left:50%"></div></div></div>
      </div>
      <div class="ctrl">
        <span style="color:#aaa;font-size:0.85em">Pan/Tilt:</span>
        <button class="btn gr" onclick="mv(0,-5)">▲</button>
        <button class="btn gr" onclick="mv(0,5)">▼</button>
        <button class="btn gr" onclick="mv(-5,0)">◀</button>
        <button class="btn gr" onclick="mv(5,0)">▶</button>
        <button class="btn gr" onclick="fetch('/servos/home/go')">🏠 Go Home</button>
      </div>
    </div>
    <div style="min-width:220px;max-width:260px">
      <div style="background:#1a1a2a;border:1px solid #334;border-radius:6px;padding:12px">
        <strong style="font-size:0.9em;color:#aaf">Calibration Strength</strong>
        <div class="sbar2" style="margin:8px 0"><div class="sf" id="sfill" style="width:0%"></div></div>
        <div style="font-size:0.78em;color:#aaa;line-height:1.6" id="stext">Insufficient data</div>
        <div style="font-size:0.78em;color:#888;margin-top:4px" id="spts">0 points saved</div>
      </div>
    </div>
  </div>

  <div id="ph-home" class="ph" style="display:none">
    <h3>Step 1 of 6 — Set Home Position</h3>
    <p>Pan/tilt until camera points where you want it to rest. This is the zone drawing reference position.</p>
    <button class="btn g" onclick="setupSetHome()">✓ Set Home &amp; Draw Zone</button>
    <button class="btn gr" style="margin-left:8px" onclick="setupReset()">Start Over</button>
  </div>

  <div id="ph-draw" class="ph" style="display:none">
    <h3>Step 2 of 6 — Draw Forbidden Zone</h3>
    <p>Click to place vertices. Close zone when done, then begin calibration.</p>
    <div class="ctrl">
      <button class="btn b on" id="btn-add" onclick="vmode('add')">+ Add</button>
      <button class="btn r" id="btn-del" onclick="vmode('delete')">✕ Delete</button>
      <button class="btn gr" onclick="clearZone()">Clear All</button>
      <button class="btn y" id="btn-cz" onclick="closeZone()" disabled>Close Zone</button>
    </div>
    <div style="margin:8px 0;font-size:0.85em">
      Depth to zone center (m, optional):
      <input type="number" id="depth-in" step="0.1" min="0.1" style="width:70px"
             onchange="setDepth(this.value)">
    </div>
    <div class="ctrl" style="margin-top:4px">
      <button class="btn g" id="btn-dd" onclick="beginForcedCal()" disabled>✓ Done — Begin Calibration</button>
      <button class="btn gr" onclick="setupReset()">Start Over</button>
    </div>
  </div>

  <div id="ph-forced" class="ph" style="display:none">
    <h3 id="ftitle">Forced Calibration</h3>
    <p id="fmsg"></p>
    <div class="ctrl">
      <button class="btn y" id="btn-tgt" onclick="toggleTgt()">▶ Targeting</button>
    </div>
    <p style="font-size:0.82em;color:#888;margin-top:4px">Drag vertices to move them. Click inside zone and drag to move the whole zone.</p>
    <button class="btn g" id="btn-df" onclick="saveForcedPt()">✓ Done</button>
    <button class="btn gr" style="margin-left:8px" onclick="setupReset()">Start Over</button>
  </div>

  <div id="ph-extra" class="ph" style="display:none">
    <h3>Step 6 — Additional Calibration Points</h3>
    <p>Pan/tilt to new positions and save points. Add at least 4 more for full strength.</p>
    <p style="font-size:0.82em;color:#888;margin-top:4px">Drag vertices to move them. Click inside zone and drag to move the whole zone.</p>
    <div class="ctrl">
      <button class="btn y" id="btn-tgt-e" onclick="toggleTgt()">▶ Targeting</button>
      <button class="btn g" onclick="addExtraPt()">📍 Save Point Here</button>
    </div>
    <div id="cpl-list" style="max-height:200px;overflow-y:auto;margin:6px 0"></div>
    <div class="ctrl" style="margin-top:8px">
      <button class="btn g" onclick="setupFinish()">✓ Finish Setup</button>
      <button class="btn gr" onclick="setupReset()">Start Over</button>
    </div>
  </div>
</div>

<!-- SETTINGS -->
<div id="settings" class="page">
  <h2 style="margin-bottom:14px">Settings</h2>
  <h3 style="color:#aaa;margin-bottom:8px;font-size:0.9em">Firing</h3>
  <div class="sg" style="margin-bottom:16px">
    <div class="set"><label>Firing Mode</label>
      <select id="firing_mode" onchange="updSettings()">
        <option value="single">Single Fire</option>
        <option value="semi_auto">Semi-Auto</option>
      </select></div>
    <div class="set"><label>Confidence: <span class="vl" id="vl-c">0.5</span></label>
      <input type="range" min="0.1" max="1" step="0.05" value="0.5" id="confidence_threshold"
             oninput="document.getElementById('vl-c').textContent=this.value" onchange="updSettings()"></div>
    <div class="set"><label>Burst Length (s): <span class="vl" id="vl-b">1.0</span></label>
      <input type="range" min="0.1" max="5" step="0.1" value="1" id="burst_length"
             oninput="document.getElementById('vl-b').textContent=this.value" onchange="updSettings()"></div>
    <div class="set"><label>Reload Time (s): <span class="vl" id="vl-r">10</span></label>
      <input type="range" min="1" max="60" step="1" value="10" id="reload_time"
             oninput="document.getElementById('vl-r').textContent=this.value" onchange="updSettings()"></div>
    <div class="set"><label>Semi-Auto Delay (s): <span class="vl" id="vl-s">2</span></label>
      <input type="range" min="0.5" max="10" step="0.5" value="2" id="semi_auto_delay"
             oninput="document.getElementById('vl-s').textContent=this.value" onchange="updSettings()"></div>
    <div class="set"><label>On-Target Tolerance (px): <span class="vl" id="vl-t">20</span></label>
      <input type="range" min="5" max="100" step="5" value="20" id="on_target_tolerance"
             oninput="document.getElementById('vl-t').textContent=this.value" onchange="updSettings()"></div>
  </div>
  <h3 style="color:#aaa;margin-bottom:8px;font-size:0.9em">Camera</h3>
  <div class="sg">
    <div class="set"><label>Brightness (-1 to 1)</label>
      <div style="display:flex;gap:6px;align-items:center">
        <input type="range" min="-1" max="1" step="0.05" value="0" id="cam-brightness"
               oninput="syncN('brightness',this.value)" onchange="updCam()">
        <input type="number" id="num-brightness" value="0" step="0.05" min="-1" max="1"
               oninput="syncS('brightness',this.value)" onchange="updCam()"></div></div>
    <div class="set"><label>Contrast (0 to 2)</label>
      <div style="display:flex;gap:6px;align-items:center">
        <input type="range" min="0" max="2" step="0.05" value="1" id="cam-contrast"
               oninput="syncN('contrast',this.value)" onchange="updCam()">
        <input type="number" id="num-contrast" value="1" step="0.05" min="0" max="2"
               oninput="syncS('contrast',this.value)" onchange="updCam()"></div></div>
    <div class="set"><label>Saturation (0 to 2)</label>
      <div style="display:flex;gap:6px;align-items:center">
        <input type="range" min="0" max="2" step="0.05" value="1" id="cam-saturation"
               oninput="syncN('saturation',this.value)" onchange="updCam()">
        <input type="number" id="num-saturation" value="1" step="0.05" min="0" max="2"
               oninput="syncS('saturation',this.value)" onchange="updCam()"></div></div>
    <div class="set"><label>Sharpness (0 to 2)</label>
      <div style="display:flex;gap:6px;align-items:center">
        <input type="range" min="0" max="2" step="0.05" value="1" id="cam-sharpness"
               oninput="syncN('sharpness',this.value)" onchange="updCam()">
        <input type="number" id="num-sharpness" value="1" step="0.05" min="0" max="2"
               oninput="syncS('sharpness',this.value)" onchange="updCam()"></div></div>
    <div class="set"><label>EV Compensation (-4 to 4)</label>
      <div style="display:flex;gap:6px;align-items:center">
        <input type="range" min="-4" max="4" step="0.5" value="0" id="cam-ev"
               oninput="syncN('ev',this.value)" onchange="updCam()">
        <input type="number" id="num-ev" value="0" step="0.5" min="-4" max="4"
               oninput="syncS('ev',this.value)" onchange="updCam()"></div></div>
  </div>
</div>

<!-- RECORDINGS -->
<div id="recordings" class="page">
  <h2 style="margin-bottom:14px">Recordings</h2>
  <button class="btn gr" onclick="loadRecs()" style="margin-bottom:12px">Refresh</button>
  <div class="rl" id="rl">Loading...</div>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let curPage   = 'live';
let vMode     = 'add';
let verts     = [];
let zClosed   = false;
let dragIdx   = -1;
let dragOff   = {x:0,y:0};
let tgtOn     = false;
let curPhase  = null;
let camInit   = false;

// ── Page Nav ───────────────────────────────────────────────────────────────
function showPage(page, btn) {
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
  document.getElementById(page).classList.add('active');
  btn.classList.add('active');
  curPage = page;
  if (page==='recordings') loadRecs();
  if (page==='setup' && curPhase===null)
    fetch('/setup/start').then(r=>r.json()).then(d=>applyPhase(d.phase));
}

// ── Servo Move ─────────────────────────────────────────────────────────────
function mv(pd,td){
  fetch('/servos/move',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({pan_delta:pd,tilt_delta:td})});
}

// ── Servo Bars ─────────────────────────────────────────────────────────────
function updServoBars(pp,tp){
  const pl = ((pp+100)/200*100).toFixed(1)+'%';
  const tl = ((tp+100)/200*100).toFixed(1)+'%';
  const pairs = [['pp','pi','tp','ti'],['pp-s','pi-s','tp-s','ti-s']];
  pairs.forEach(([ppid,piid,tpid,tiid])=>{
    const pe=document.getElementById(ppid),pii=document.getElementById(piid);
    const te=document.getElementById(tpid),tii=document.getElementById(tiid);
    if(pe)pe.textContent=pp+'%'; if(pii)pii.style.left=pl;
    if(te)te.textContent=tp+'%'; if(tii)tii.style.left=tl;
  });
}

// ── Phase Management ───────────────────────────────────────────────────────
function applyPhase(phase){
  curPhase = phase;
  ['ph-home','ph-draw','ph-forced','ph-extra'].forEach(id=>{
    document.getElementById(id).style.display='none';
  });
  if(phase==='home'){
    document.getElementById('ph-home').style.display='block';
  } else if(phase==='draw'){
    document.getElementById('ph-draw').style.display='block';
    vmode('add');
  } else if(['forced_L','forced_R','forced_up','forced_down'].includes(phase)){
    document.getElementById('ph-forced').style.display='block';
    const T={forced_L:'Step 3 of 6 — Left Pan Endpoint (visual left)',
             forced_R:'Step 4 of 6 — Right Pan Endpoint (visual right)',
             forced_up:'Step 5a of 6 — Tilt Up Endpoint',
             forced_down:'Step 5b of 6 — Tilt Down Endpoint'};
    const M={forced_L:'Camera moved to visual left limit. If zone is off-screen, pan right until visible, then drag zone or vertices.',
             forced_R:'Camera moved to visual right limit. If zone is off-screen, pan left until visible, then drag zone or vertices.',
             forced_up:'Camera tilted up to limit. Drag zone or vertices to correct position.',
             forced_down:'Camera tilted down to limit. Drag zone or vertices to correct position. Camera returns home when done.'};
    document.getElementById('ftitle').textContent=T[phase]||phase;
    document.getElementById('fmsg').textContent=M[phase]||'';
    document.getElementById('btn-df').textContent=
      phase==='forced_down'?'✓ Done — Return to Home':'✓ Done';
  } else if(phase==='extra'){
    document.getElementById('ph-extra').style.display='block';
    loadCalPoints();
  }
}

function setupSetHome(){fetch('/setup/set_home').then(r=>r.json()).then(d=>applyPhase(d.phase));}
function setupReset(){verts=[];zClosed=false;fetch('/setup/reset').then(r=>r.json()).then(d=>applyPhase(d.phase));}
function clearZone(){verts=[];zClosed=false;sendZone();updDrawBtns();}
function closeZone(){if(verts.length<3)return;zClosed=true;sendZone();updDrawBtns();}
function beginForcedCal(){fetch('/setup/begin_forced_cal').then(r=>r.json()).then(d=>applyPhase(d.phase));}
function saveForcedPt(){
  fetch('/setup/save_forced_point',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({vertices:verts})}).then(r=>r.json()).then(d=>applyPhase(d.phase));
}
function addExtraPt(){
  fetch('/setup/add_extra_point',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({vertices:verts})}).then(r=>r.json()).then(d=>{
    updStrength(d); loadCalPoints();});
}
function setupFinish(){fetch('/setup/finish');curPhase=null;applyPhase(null);}
function setDepth(v){
  fetch('/setup/depth',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({depth_m:parseFloat(v)||null})});
}
function toggleTgt(){
  tgtOn=!tgtOn;
  ['btn-tgt','btn-tgt-e'].forEach(id=>{
    const el=document.getElementById(id);
    if(el){el.textContent=tgtOn?'⏹ Stop Targeting':'▶ Targeting';
           tgtOn?el.classList.add('on'):el.classList.remove('on');}
  });
  fetch(tgtOn?'/setup/targeting/start':'/setup/targeting/stop');
}

function updStrength(data){
  const s=data.strength||{};
  const fill=document.getElementById('sfill');
  const text=document.getElementById('stext');
  const pts=document.getElementById('spts');
  const n=data.n_points!==undefined?data.n_points:(s.n_points||0);
  if(pts) pts.textContent=`${n} calibration point${n!==1?'s':''} saved`;
  if(!fill||!text) return;
  if(s.combined!=null){
    fill.style.width=s.combined+'%';
    text.innerHTML=
      `<b style="color:#4af">Combined: ${s.combined}%</b><br>`+
      `LOOCV Position: ${s.position_confidence}%<br>`+
      `Coverage: ${s.coverage_score}%`+
      (n<8?`<br><span style="color:#fa0">⚠ Add ${8-n} more point${8-n!==1?'s':''} to unlock full strength</span>`:'');
  } else {
    fill.style.width='0%';
    text.textContent=n<3?`Need ${3-n} more point${3-n!==1?'s':''} for LOOCV assessment`:'Calculating...';
  }
}

function loadCalPoints(){
  fetch('/setup/cal_points').then(r=>r.json()).then(d=>{
    const list=document.getElementById('cpl-list');
    if(!list)return;
    if(!d.points||!d.points.length){list.innerHTML='<p style="color:#666;font-size:0.85em">No points saved</p>';return;}
    list.innerHTML=d.points.map((p,i)=>`
      <div class="cpl" onclick="gotoCalPt(${p.index})">
        <span>Point ${p.index+1} — Pan ${p.pan}° Tilt ${p.tilt}°</span>
        <button class="btn r" style="padding:3px 8px;font-size:0.8em"
          onclick="event.stopPropagation();removeCalPt(${p.index})">✕</button>
      </div>`).join('');
  });
}

function gotoCalPt(idx){
  fetch('/setup/goto_point',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({index:idx})});
}
function removeCalPt(idx){
  fetch('/setup/remove_point',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({index:idx})}).then(r=>r.json()).then(d=>{updStrength(d);loadCalPoints();});
}

// ── Vertex Mode ────────────────────────────────────────────────────────────
function vmode(m){
  vMode=m;
  ['btn-add','btn-del'].forEach(id=>{
    const el=document.getElementById(id);if(el)el.classList.remove('on');
  });
  const map={add:'btn-add',delete:'btn-del'};
  const el=document.getElementById(map[m]);if(el)el.classList.add('on');
}

function updDrawBtns(){
  const cz=document.getElementById('btn-cz');
  const dd=document.getElementById('btn-dd');
  if(cz)cz.disabled=verts.length<3||zClosed;
  if(dd)dd.disabled=!zClosed;
}

function sendZone(){
  fetch('/setup/zone',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({vertices:verts,closed:zClosed})});
}

// ── Setup Canvas ───────────────────────────────────────────────────────────
const sc=document.getElementById('ov-setup');
let zoneDrag = false;
let zoneDragStart = null;
let zoneDragOrigin = null;
let lastServerPan = null;
let lastServerTilt = null;

function ptInZone(x, y){
  if(verts.length < 3) return false;
  let inside = false;
  for(let i=0,j=verts.length-1;i<verts.length;j=i++){
    const xi=verts[i][0],yi=verts[i][1],xj=verts[j][0],yj=verts[j][1];
    if(((yi>y)!=(yj>y))&&(x<(xj-xi)*(y-yi)/(yj-yi)+xi)) inside=!inside;
  }
  return inside;
}

sc.addEventListener('mousedown',e=>{
  const r=sc.getBoundingClientRect();
  const x=Math.round(e.clientX-r.left),y=Math.round(e.clientY-r.top);

  if(vMode==='add'&&!zClosed){
    verts.push([x,y]);
    sendZone();
    updDrawBtns();
    return;
  }
  if(vMode==='delete'){
    const i=nearV(x,y,15);
    if(i>=0){verts.splice(i,1);if(verts.length<3)zClosed=false;sendZone();updDrawBtns();}
    return;
  }
  // Default: vertex grab takes priority, then whole zone drag
  const vi = nearV(x,y,15);
  if(vi>=0){
    dragIdx=vi;
    dragOff={x:x-verts[vi][0],y:y-verts[vi][1]};
    sc.style.cursor='grabbing';
  } else if(zClosed && ptInZone(x,y)){
    zoneDrag=true;
    zoneDragStart={x,y};
    zoneDragOrigin=verts.map(v=>[v[0],v[1]]);
    sc.style.cursor='grabbing';
  }
});

sc.addEventListener('mousemove',e=>{
  const r=sc.getBoundingClientRect();
  const x=Math.round(e.clientX-r.left),y=Math.round(e.clientY-r.top);
  if(dragIdx>=0){
    verts[dragIdx]=[x-dragOff.x,y-dragOff.y];
  } else if(zoneDrag&&zoneDragStart&&zoneDragOrigin){
    const dx=x-zoneDragStart.x, dy=y-zoneDragStart.y;
    verts=zoneDragOrigin.map(v=>[v[0]+dx,v[1]+dy]);
  }
});

sc.addEventListener('mouseup',()=>{
  if(dragIdx>=0){sendZone();dragIdx=-1;sc.style.cursor=zClosed?'grab':'crosshair';}
  else if(zoneDrag){sendZone();zoneDrag=false;zoneDragStart=null;zoneDragOrigin=null;sc.style.cursor='grab';}
});

function nearV(x,y,thresh){
  let best=-1,bestD=thresh;
  verts.forEach(([vx,vy],i)=>{
    const d=Math.sqrt((x-vx)**2+(y-vy)**2);if(d<bestD){bestD=d;best=i;}
  });
  return best;
}

// ── Overlay Drawing ────────────────────────────────────────────────────────
function drawOvs(d){
  const cvs=[
    {id:'ov-live', useServer:true},
    {id:'ov-setup',useServer:curPhase!=='draw'},
  ];
  cvs.forEach(({id,useServer})=>{
    const c=document.getElementById(id);if(!c)return;
    const ctx=c.getContext('2d');
    ctx.clearRect(0,0,c.width,c.height);

    // Zone — in draw phase use local verts; in calibration phases use local verts (synced from server)
    const inCalPhase = ['forced_L','forced_R','forced_up','forced_down','extra'].includes(curPhase);
    const useLocal = (id==='ov-setup') && (curPhase==='draw' || inCalPhase);
    const zv = useLocal ? verts : (d.zone_px||[]);
    const zc = useLocal ? zClosed : d.zone_closed;
    const zColor = inCalPhase && d.setup_zone_exact ? '#00ff88' : (zc?'#00ff00':'#00ffff');
    if(zv.length>=1) drawZone(ctx,zv,zc,zColor);

    // Off-screen arrow
    if(useServer&&zv.length>0){
      const cx=zv.reduce((s,p)=>s+p[0],0)/zv.length;
      const cy=zv.reduce((s,p)=>s+p[1],0)/zv.length;
      if(cx<0||cx>c.width||cy<0||cy>c.height) drawArrow(ctx,cx,cy,c.width,c.height);
    }

    // Detections
    (d.detections||[]).forEach(det=>{
      const col=det.in_zone?'#ff0000':'#ff8800';
      ctx.strokeStyle=col;ctx.lineWidth=det.is_primary?3:1;
      ctx.strokeRect(det.x1,det.y1,det.x2-det.x1,det.y2-det.y1);
      ctx.fillStyle=col;ctx.font='12px sans-serif';
      ctx.fillText('CAT'+(det.is_primary?' [TGT]':'')+' '+det.conf,det.x1,det.y1-4);
    });

    // REC dot
    if(d.recording){
      ctx.fillStyle='#f00';ctx.beginPath();ctx.arc(20,20,8,0,2*Math.PI);ctx.fill();
      ctx.fillStyle='#f00';ctx.font='bold 12px sans-serif';ctx.fillText('REC',32,25);
    }
  });
}

function drawZone(ctx,verts,closed,color){
  if(!verts.length)return;
  ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(verts[0][0],verts[0][1]);
  for(let i=1;i<verts.length;i++)ctx.lineTo(verts[i][0],verts[i][1]);
  if(closed)ctx.closePath();ctx.stroke();
  verts.forEach(([vx,vy])=>{ctx.beginPath();ctx.arc(vx,vy,5,0,2*Math.PI);ctx.fill();});
}

function drawArrow(ctx,cx,cy,w,h){
  const ex=Math.max(20,Math.min(w-20,cx));
  const ey=Math.max(20,Math.min(h-20,cy));
  const a=Math.atan2(cy-h/2,cx-w/2);
  ctx.save();ctx.translate(ex,ey);ctx.rotate(a);
  ctx.fillStyle='#ff0';ctx.beginPath();
  ctx.moveTo(15,0);ctx.lineTo(-10,-8);ctx.lineTo(-10,8);
  ctx.closePath();ctx.fill();ctx.restore();
  ctx.fillStyle='#ff0';ctx.font='bold 11px sans-serif';
  ctx.fillText('Zone →',Math.max(5,ex-20),Math.max(15,ey-10));
}

// ── Settings ───────────────────────────────────────────────────────────────
function updSettings(){
  fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      firing_mode:document.getElementById('firing_mode').value,
      burst_length:parseFloat(document.getElementById('burst_length').value),
      reload_time:parseFloat(document.getElementById('reload_time').value),
      semi_auto_delay:parseFloat(document.getElementById('semi_auto_delay').value),
      confidence_threshold:parseFloat(document.getElementById('confidence_threshold').value),
      on_target_tolerance:parseInt(document.getElementById('on_target_tolerance').value),
    })});
}

function syncN(k,v){const el=document.getElementById('num-'+k);if(el)el.value=parseFloat(v).toFixed(2);}
function syncS(k,v){const el=document.getElementById('cam-'+k);if(el)el.value=v;}
function updCam(){
  fetch('/camera',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      brightness:parseFloat(document.getElementById('cam-brightness').value),
      contrast:parseFloat(document.getElementById('cam-contrast').value),
      saturation:parseFloat(document.getElementById('cam-saturation').value),
      sharpness:parseFloat(document.getElementById('cam-sharpness').value),
      ev:parseFloat(document.getElementById('cam-ev').value),
    })});
}

// ── Status Polling ─────────────────────────────────────────────────────────
setInterval(()=>{
  fetch('/status').then(r=>r.json()).then(d=>{
    // Status bar
    document.getElementById('da').className='dot '+(d.armed?'on':'');
    document.getElementById('ta').textContent=d.armed?'Armed':'Disarmed';
    document.getElementById('dc').className='dot '+(d.cats_detected>0?'warn':'');
    document.getElementById('tc').textContent=d.cats_detected>0?d.cats_detected+' cat(s)':'No cats';
    document.getElementById('dz').className='dot '+(d.cats_in_zone>0?'warn':'');
    document.getElementById('tz').textContent=d.cats_in_zone>0?d.cats_in_zone+' in zone':'Zone clear';
    document.getElementById('df').className='dot '+(d.firing?'warn':'');
    document.getElementById('tf').textContent=d.firing?'💦 FIRING':'Idle';
    document.getElementById('dr').className='dot '+(d.recording?'rec':'');
    document.getElementById('tr').textContent=d.recording?'⏺ Recording':'Not recording';

    // Servo bars
    updServoBars(d.pan_pct||0,d.tilt_pct||0);

    // Sync verts from server when servo position changes (camera moved)
    const inSetupCalPhase = ['forced_L','forced_R','forced_up','forced_down','extra'].includes(curPhase);
    if(inSetupCalPhase && d.setup_zone && d.setup_zone.length > 0){
      const panChanged = lastServerPan !== null && Math.abs(d.pan - lastServerPan) > 0.3;
      const tiltChanged = lastServerTilt !== null && Math.abs(d.tilt - lastServerTilt) > 0.3;
      if(panChanged || tiltChanged || verts.length === 0){
        verts = d.setup_zone.map(p=>[p[0],p[1]]);
        zClosed = d.setup_zone_closed;
      }
    }
    lastServerPan = d.pan;
    lastServerTilt = d.tilt;

    // Phase sync
    if(d.setup_phase!==curPhase&&curPage==='setup') applyPhase(d.setup_phase);

    // Strength update whenever on setup page
    if(curPage==='setup'&&d.strength) updStrength({strength:d.strength,n_points:d.n_cal_points});

    // Camera settings init
    if(d.cam&&!camInit){
      camInit=true;
      ['brightness','contrast','saturation','sharpness','ev'].forEach(k=>{
        const s=document.getElementById('cam-'+k);
        const n=document.getElementById('num-'+k);
        if(s)s.value=d.cam[k];
        if(n)n.value=parseFloat(d.cam[k]).toFixed(2);
      });
    }

    // Sync verts from server in forced/extra phases
    if(['forced_L','forced_R','forced_up','forced_down','extra'].includes(curPhase)){
      if(d.zone_px&&d.zone_px.length>0&&verts.length===0){
        verts=d.zone_px.map(p=>[p[0],p[1]]);
        zClosed=d.zone_closed;
      }
    }

    // Draw overlays
    drawOvs(d);
  });
},200);

// ── Recordings ─────────────────────────────────────────────────────────────
function loadRecs(){
  fetch('/recordings').then(r=>r.json()).then(d=>{
    const list=document.getElementById('rl');
    if(!d.recordings.length){list.innerHTML='<p style="color:#666">No recordings yet</p>';return;}
    list.innerHTML=d.recordings.map(f=>`
      <div class="ri"><span>${f}</span>
        <a href="/recordings/${f}" target="_blank"><button class="btn b">▶ Play</button></a>
      </div>`).join('');
  });
}
</script>
</body>
</html>
"""
