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
import logging
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

LOG_FILE = BASE_DIR / "catblastor_events.log"
_log_handler = logging.FileHandler(str(LOG_FILE))
_log_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
_logger = logging.getLogger("cb")
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_log_handler)

def log(msg):
    _logger.info(msg)

# ─── Camera Settings ─────────────────────────────────────────────────────────
DEFAULT_CAM = {"brightness":0.0,"contrast":1.0,"saturation":1.0,"sharpness":1.0,"ev":0.0,
               "awb_gain_r":1.0,"awb_gain_b":1.0}
cam_settings = dict(DEFAULT_CAM)
if CAM_FILE.exists():
    cam_settings.update(json.loads(CAM_FILE.read_text()))

def save_cam_settings():
    CAM_FILE.write_text(json.dumps(cam_settings))

# ─── Zone Calibration ────────────────────────────────────────────────────────
zone_cal = {"home_pan":90.0,"home_tilt":90.0,"depth_m":None,
            "calibration_points":[],"home_vertices":[],"home_closed":False}
if ZONE_CAL_FILE.exists():
    zone_cal.update(json.loads(ZONE_CAL_FILE.read_text()))

def save_zone_cal():
    ZONE_CAL_FILE.write_text(json.dumps(zone_cal))

def interpolate_zone(pan, tilt):
    """Interpolate zone at given pan/tilt. Never modifies global state."""
    return _interpolate_with_cal(zone_cal, pan, tilt)

def get_interpolation_debug(pan, tilt):
    pts = zone_cal["calibration_points"]
    n_total = int(((PAN_MAX-PAN_MIN)/GRID_STEP+1)*((TILT_MAX-TILT_MIN)/GRID_STEP+1))
    base = {"n_points":len(pts),"n_total_possible":n_total,
            "coverage_pct":round(len(pts)/n_total*100,1)}
    if not pts:
        return {**base,"method":"no_data","details":"No calibration points saved"}
    def dist(p): return float(np.sqrt((pan-p["pan"])**2+(tilt-p["tilt"])**2))
    nearest = min(pts, key=dist)
    nd = round(dist(nearest), 2)
    if nd < 2.5:
        return {**base,"method":"exact",
                "details":f"Nearest saved point {nd}° away (pan={nearest['pan']} tilt={nearest['tilt']}) — shown exactly"}
    if len(pts) >= 3:
        return {**base,"method":"linear_regression",
                "details":f"Nearest saved point {nd}° away — interpolating from {len(pts)} pts"}
    return {**base,"method":"nearest_neighbor",
            "details":f"Nearest saved point {nd}° away — only {len(pts)} pts (need 3 for regression)"}

loocv_counter = 0

def compute_calibration_strength():
    global loocv_counter
    # Work on a deep copy — NEVER modify zone_cal["calibration_points"] in place
    import copy
    pts = copy.deepcopy(zone_cal["calibration_points"])
    n = len(pts)
    if n < 3:
        return {"position_confidence":None,"coverage_score":None,"combined":None,
                "n_points":n,"loocv_runs":loocv_counter}
    errors = []
    for i, p in enumerate(pts):
        remaining = [pts[j] for j in range(n) if j != i]
        # Temporarily build a fake zone_cal for interpolation
        fake_cal = copy.deepcopy(zone_cal)
        fake_cal["calibration_points"] = remaining
        predicted = _interpolate_with_cal(fake_cal, p["pan"], p["tilt"])
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
    loocv_counter += 1
    if not errors:
        return {"position_confidence":None,"coverage_score":None,"combined":None,
                "n_points":n,"loocv_runs":loocv_counter}
    pos_conf = max(0.0, 1.0 - np.mean(errors)) * 100.0
    pans  = [p["pan"]  for p in pts]; tilts = [p["tilt"] for p in pts]
    cov   = ((max(pans)-min(pans))/(PAN_MAX-PAN_MIN)*0.5 +
             (max(tilts)-min(tilts))/(TILT_MAX-TILT_MIN)*0.5) * 100.0
    raw   = pos_conf*0.7 + cov*0.3
    n_user = sum(1 for p in pts if p.get("user_set", False))
    combined = min(raw, 50.0) if n_user < 8 else raw
    return {"position_confidence":round(pos_conf,1),"coverage_score":round(cov,1),
            "combined":round(combined,1),"n_points":n,"n_user_set":n_user,
            "loocv_runs":loocv_counter}

def _interpolate_with_cal(cal, pan, tilt):
    """Interpolate zone using a specific calibration dict (for LOOCV — never uses global)."""
    pts = cal["calibration_points"]
    if not pts: return None
    if len(pts) == 1: return [list(v) for v in pts[0]["vertices"]]
    # Find nearest point
    def dist(p): return float(np.sqrt((pan-p["pan"])**2+(tilt-p["tilt"])**2))
    nearest = min(pts, key=dist)
    if dist(nearest) < 2.5: return [list(v) for v in nearest["vertices"]]
    n_verts = len(pts[0]["vertices"])
    if len(pts) >= 3:
        try:
            A = np.array([[p["pan"], p["tilt"], 1.0] for p in pts])
            result = []
            for i in range(n_verts):
                bx = np.array([p["vertices"][i][0] for p in pts if i < len(p["vertices"])])
                by = np.array([p["vertices"][i][1] for p in pts if i < len(p["vertices"])])
                if len(bx) < 3: break
                cx, _, _, _ = np.linalg.lstsq(A[:len(bx)], bx, rcond=None)
                cy, _, _, _ = np.linalg.lstsq(A[:len(by)], by, rcond=None)
                result.append([int(round(float(cx[0]*pan + cx[1]*tilt + cx[2]))),
                                int(round(float(cy[0]*pan + cy[1]*tilt + cy[2])))])
            if len(result) == n_verts: return result
        except Exception: pass
    return [list(v) for v in nearest["vertices"]]

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
        "--awbgains",   f"{cam_settings['awb_gain_r']},{cam_settings['awb_gain_b']}",
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
    "reticle_x":FRAME_W//2,"reticle_y":FRAME_H//2,
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

GRID_STEP = 5.0  # degrees per grid step
last_grid_pos = {"pan": None, "tilt": None}

def grid_snap(angle):
    return round(round(angle / GRID_STEP) * GRID_STEP, 1)

def auto_save_calibration_point():
    """Auto-save zone at current grid position when in calibration phase."""
    if state["setup_phase"] not in ("forced_L","forced_R","forced_up","forced_down","extra"):
        return
    gp = grid_snap(servo_angles["pan"])
    gt = grid_snap(servo_angles["tilt"])
    # Check if already saved at this grid position
    existing = next((p for p in zone_cal["calibration_points"]
                     if abs(p["pan"]-gp)<0.1 and abs(p["tilt"]-gt)<0.1), None)
    if existing:
        return  # already have this position, don't overwrite unless user drags
    # Get zone — interpolated from existing points or fall back to home zone
    zone, closed, _ = get_setup_zone()
    if not zone or not closed:
        return
    zone_cal["calibration_points"].append({
        "pan": gp, "tilt": gt,
        "vertices": [list(v) for v in zone]
    })
    save_zone_cal()

def move_servos(pan, tilt, slow=False):
    global last_activity_time, last_grid_pos
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
    gp = grid_snap(servo_angles["pan"])
    gt = grid_snap(servo_angles["tilt"])
    last_grid_pos["pan"] = gp
    last_grid_pos["tilt"] = gt

def servo_pct(angle, center, rang):
    return round((angle-center)/rang*100.0, 1)

# ─── Zone Runtime State ──────────────────────────────────────────────────────
zone_vertices = list(zone_cal.get("home_vertices", []))
zone_closed   = bool(zone_cal.get("home_closed", False))

def get_live_zone():
    """For live mode: always interpolate — smooth continuous zone tracking.
    Saved points anchor the model; zone moves fluidly with the camera."""
    pan, tilt = servo_angles["pan"], servo_angles["tilt"]
    pts = zone_cal["calibration_points"]
    if not pts:
        return zone_vertices, zone_closed
    if len(pts) == 1:
        return pts[0]["vertices"], True
    interp = interpolate_zone(pan, tilt)
    if interp:
        return interp, True
    return zone_vertices, zone_closed

def get_setup_zone():
    """For setup mode: snap to exact saved position if within 0.6°, else interpolate.
    User sees exactly what they saved at each calibration position."""
    pan, tilt = servo_angles["pan"], servo_angles["tilt"]
    pts = zone_cal["calibration_points"]
    if not pts:
        return zone_vertices, zone_closed, False
    def dist(p): return float(np.sqrt((pan-p["pan"])**2 + (tilt-p["tilt"])**2))
    nearest = min(pts, key=dist)
    if dist(nearest) < 0.6:
        return nearest["vertices"], True, True
    interp = interpolate_zone(pan, tilt)
    if interp:
        return interp, True, False
    return zone_vertices, zone_closed, False

def get_current_zone():
    verts, closed = get_live_zone()
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
world_target = {"pan":None,"tilt":None}  # absolute angle where cat is in world space
world_target_lock = threading.Lock()
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
        # Snapshot servo angles at the exact moment inference ran
        pan_now  = servo_angles["pan"]
        tilt_now = servo_angles["tilt"]
        results = model(frame, verbose=False)
        # Also snapshot previous target for direction verification
        with world_target_lock:
            prev_target_pan  = world_target["pan"]
            prev_target_tilt = world_target["tilt"]
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
            prev_in_zone = {d["id"] for d in latest_detections if d["in_zone"]}
            prev_ids     = {d["id"] for d in latest_detections}
            latest_detections = dets

        # Compute world target from primary in-zone cat at inference-time servo angles
        in_zone_dets = [d for d in dets if d["in_zone"]]
        with world_target_lock:
            if in_zone_dets:
                t = in_zone_dets[0]
                rx = state["reticle_x"]; ry = state["reticle_y"]
                # Log how cat pixel moved vs how servo moved — tells us the sign empirically
                if prev_target_pan is not None:
                    dpan  = round(pan_now  - prev_target_pan,  1)
                    dtilt = round(tilt_now - prev_target_tilt, 1)
                    log(f"DIRECTION servo_delta=({dpan},{dtilt}) cat_pixel=({t['cx']},{t['cy']}) "
                        f"[if servo moved RIGHT (+pan) and cat moved LEFT (-cx), signs are NORMAL]")
                world_target["pan"]  = pan_now  - ((t["cx"] - rx) / FRAME_W) * 66.0
                world_target["tilt"] = tilt_now - ((t["cy"] - ry) / FRAME_H) * 41.0
                log(f"TARGET pan_now={pan_now:.1f} tilt_now={tilt_now:.1f} "
                    f"cx={t['cx']} cy={t['cy']} rx={rx} ry={ry} "
                    f"→ target_pan={world_target['pan']:.1f} target_tilt={world_target['tilt']:.1f}")
            else:
                world_target["pan"]  = None
                world_target["tilt"] = None
        new_ids     = {d["id"] for d in dets}
        new_in_zone = {d["id"] for d in dets if d["in_zone"]}
        for d in dets:
            if d["id"] not in prev_ids:
                log(f"DETECT cx={d['cx']} cy={d['cy']} conf={d['conf']:.2f}")
            if d["id"] not in prev_in_zone and d["in_zone"]:
                log(f"ZONE_IN cx={d['cx']} cy={d['cy']} pan={servo_angles['pan']:.1f} tilt={servo_angles['tilt']:.1f}")
        for tid in prev_in_zone - new_in_zone:
            log(f"ZONE_OUT id={tid}")
        for tid in prev_ids - new_ids:
            log(f"LOST id={tid}")

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
    _last_log = 0.0
    while True:
        if not state["armed"] or state["setup_phase"] is not None:
            time.sleep(0.05); continue

        with world_target_lock:
            target_pan  = world_target["pan"]
            target_tilt = world_target["tilt"]

        if target_pan is None or target_tilt is None:
            time.sleep(0.05); continue

        cur_pan  = servo_angles["pan"]
        cur_tilt = servo_angles["tilt"]
        err_pan  = target_pan  - cur_pan
        err_tilt = target_tilt - cur_tilt

        # If already within 0.5° on both axes, no movement needed
        if abs(err_pan) < 0.5 and abs(err_tilt) < 0.5:
            time.sleep(0.05); continue

        MAX_STEP = 2.0
        step_pan  = max(-MAX_STEP, min(MAX_STEP, err_pan))
        step_tilt = max(-MAX_STEP, min(MAX_STEP, err_tilt))
        new_pan  = cur_pan  + step_pan
        new_tilt = cur_tilt + step_tilt

        # Clamp to 80% pan / full 60-120° tilt range from home
        pan_lim  = PAN_RANGE  * 0.8
        tilt_lim = 30.0  # allows 60° to 120° from home at 90°
        new_pan  = clamp(new_pan,
                         max(PAN_MIN,  home_position["pan"]  - pan_lim),
                         min(PAN_MAX,  home_position["pan"]  + pan_lim))
        new_tilt = clamp(new_tilt,
                         max(60.0, home_position["tilt"] - tilt_lim),
                         min(120.0, home_position["tilt"] + tilt_lim))

        now = time.time()
        if now - _last_log >= 0.5:
            _last_log = now
            log(f"TRACK err=({err_pan:.1f},{err_tilt:.1f}) "
                f"step=({step_pan:.1f},{step_tilt:.1f}) "
                f"pan:{cur_pan:.1f}->{new_pan:.1f} tilt:{cur_tilt:.1f}->{new_tilt:.1f}")

        move_servos(new_pan, new_tilt)
        time.sleep(0.05)

def firing_loop():
    while True:
        if not state["armed"] or state["setup_phase"] is not None:
            # Don't touch GPIO if targeting calibration is active — targeting_loop owns the pins
            if not targeting_active:
                GPIO.output(PUMP_PIN,GPIO.LOW); GPIO.output(SOLENOID_PIN,GPIO.LOW)
            firing["active"] = False; time.sleep(0.1); continue
        with detection_lock:
            dets = list(latest_detections)
        t = select_target(dets); now = time.time()
        if not t:
            GPIO.output(PUMP_PIN,GPIO.LOW); GPIO.output(SOLENOID_PIN,GPIO.LOW)
            firing["active"] = False; time.sleep(0.1); continue
        dist = np.sqrt((servo_angles["pan"]  - (world_target["pan"]  or 9999))**2 +
                       (servo_angles["tilt"] - (world_target["tilt"] or 9999))**2)
        if dist > 2.0:  # within 2° = on target
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
    log(f"FIRE pan={servo_angles['pan']:.1f} tilt={servo_angles['tilt']:.1f} burst={state['burst_length']}s")
    GPIO.output(PUMP_PIN,GPIO.HIGH); GPIO.output(SOLENOID_PIN,GPIO.HIGH)
    time.sleep(state["burst_length"])
    GPIO.output(SOLENOID_PIN,GPIO.LOW); GPIO.output(PUMP_PIN,GPIO.LOW)
    log("FIRE_END")
    firing["active"] = False

def targeting_loop():
    while True:
        if targeting_active:
            GPIO.output(PUMP_PIN,GPIO.HIGH); time.sleep(0.5)
            GPIO.output(SOLENOID_PIN,GPIO.HIGH); time.sleep(1.0)
            GPIO.output(SOLENOID_PIN,GPIO.LOW); time.sleep(1.0)
            GPIO.output(PUMP_PIN,GPIO.LOW); time.sleep(0.5)
        else:
            # Only touch GPIO if firing_loop isn't actively firing
            if not firing["active"]:
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

_last_cat_seen = time.time()

def home_position_loop():
    global _last_cat_seen
    while True:
        time.sleep(0.5)
        if not state["armed"] or state["setup_phase"] is not None:
            continue
        with detection_lock:
            cats = len(latest_detections)
        if cats > 0:
            _last_cat_seen = time.time()
            continue
        away = (abs(servo_angles["pan"]  - home_position["pan"])  > 1.0 or
                abs(servo_angles["tilt"] - home_position["tilt"]) > 1.0)
        if away and (time.time() - _last_cat_seen) > 5.0:
            log(f"HOME pan={servo_angles['pan']:.1f}->{home_position['pan']:.1f} "
                f"tilt={servo_angles['tilt']:.1f}->{home_position['tilt']:.1f}")
            move_servos(home_position["pan"], home_position["tilt"], slow=True)

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
    state["armed"] = True
    log(f"ARM pan={servo_angles['pan']:.1f} tilt={servo_angles['tilt']:.1f}")
    return {"status":"armed"}

@app.get("/disarm")
def disarm():
    global targeting_active
    state["armed"] = False; targeting_active = False
    GPIO.output(PUMP_PIN,GPIO.LOW); GPIO.output(SOLENOID_PIN,GPIO.LOW)
    log("DISARM")
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
    zone_px, z_closed = get_live_zone()
    setup_verts, setup_closed, is_exact = get_setup_zone()
    strength = compute_calibration_strength()
    gp = grid_snap(servo_angles["pan"])
    gt = grid_snap(servo_angles["tilt"])
    n_total = int(((PAN_MAX-PAN_MIN)/GRID_STEP+1)*((TILT_MAX-TILT_MIN)/GRID_STEP+1))
    return {
        "armed":state["armed"],"setup_phase":state["setup_phase"],
        "cats_detected":cats,"cats_in_zone":in_z,
        "firing":firing["active"],"recording":recording["active"],
        "targeting_active":targeting_active,
        "pan":round(servo_angles["pan"],1),"tilt":round(servo_angles["tilt"],1),
        "pan_pct":servo_pct(servo_angles["pan"],PAN_CENTER,PAN_RANGE),
        "tilt_pct":servo_pct(servo_angles["tilt"],TILT_CENTER,TILT_RANGE),
        "grid_pan":gp,"grid_tilt":gt,
        "grid_pan_pct":round(servo_pct(gp,PAN_CENTER,PAN_RANGE),1),
        "grid_tilt_pct":round(servo_pct(gt,TILT_CENTER,TILT_RANGE),1),
        "detections":dets,"zone_px":zone_px,"zone_closed":z_closed,
        "setup_zone":setup_verts,"setup_zone_closed":setup_closed,"setup_zone_exact":is_exact,
        "n_cal_points":len(zone_cal["calibration_points"]),
        "n_total_positions":n_total,
        "strength":strength,
        "cam":cam_settings,"depth_m":zone_cal["depth_m"],
        "reticle_x":state["reticle_x"],"reticle_y":state["reticle_y"],
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
    for k in ["brightness","contrast","saturation","sharpness","ev","awb_gain_r","awb_gain_b"]:
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
@app.get("/setup/interp_debug")
def interp_debug():
    return get_interpolation_debug(servo_angles["pan"], servo_angles["tilt"])

@app.get("/setup/start")
def setup_start():
    has_cal = len(zone_cal["calibration_points"]) > 0
    has_zone = len(zone_vertices) >= 3 or len(zone_cal.get("home_vertices",[])) >= 3
    if has_cal and has_zone:
        state["setup_phase"] = "extra"
        return {"phase":"extra","resuming":True}
    if has_zone and not has_cal:
        state["setup_phase"] = "draw"
        return {"phase":"draw","resuming":False}
    state["setup_phase"] = "home"
    return {"phase":"home","resuming":False}

@app.get("/setup/set_home")
def setup_set_home():
    zone_cal["home_pan"]  = servo_angles["pan"]
    zone_cal["home_tilt"] = servo_angles["tilt"]
    home_position["pan"]  = servo_angles["pan"]
    home_position["tilt"] = servo_angles["tilt"]
    save_home_position(); state["setup_phase"] = "draw"
    return {"phase":"draw"}

def _same_position(p, pan, tilt):
    """True if calibration point p is at the same physical position (within 0.6° Euclidean)."""
    return float(np.sqrt((p["pan"]-pan)**2 + (p["tilt"]-tilt)**2)) < 0.6

@app.post("/setup/zone")
async def setup_zone(request: Request):
    global zone_vertices, zone_closed
    data = await request.json()
    zone_vertices = data.get("vertices",[])
    zone_closed   = data.get("closed",False)
    # Only update home_vertices during the draw phase
    if state["setup_phase"] == "draw":
        zone_cal["home_vertices"] = [list(v) for v in zone_vertices]
        zone_cal["home_closed"]   = zone_closed
    # Save to calibration_points — only replace if same position within 0.6°
    if zone_closed and len(zone_vertices) >= 3:
        pan  = round(servo_angles["pan"],  1)
        tilt = round(servo_angles["tilt"], 1)
        zone_cal["calibration_points"] = [
            p for p in zone_cal["calibration_points"]
            if not _same_position(p, pan, tilt)
        ]
        zone_cal["calibration_points"].append({
            "pan":pan,"tilt":tilt,
            "vertices":[list(v) for v in zone_vertices],
            "user_set":True
        })
    save_zone_cal()
    return {"status":"ok","n_verts":len(zone_vertices)}

@app.get("/setup/begin_forced_cal")
def begin_forced_cal():
    if zone_closed and len(zone_vertices) >= 3:
        zone_cal["home_vertices"] = [list(v) for v in zone_vertices]
        zone_cal["home_closed"]   = True
        hp = round(zone_cal["home_pan"], 1)
        ht = round(zone_cal["home_tilt"], 1)
        zone_cal["calibration_points"] = [{
            "pan": hp, "tilt": ht,
            "vertices": [list(v) for v in zone_vertices],
            "user_set": True}]
        save_zone_cal()
    state["setup_phase"] = "forced_L"
    def go():
        time.sleep(0.5)
        move_servos(PAN_MAX, zone_cal["home_tilt"], slow=True)
    threading.Thread(target=go, daemon=True).start()
    return {"phase":"forced_L"}

@app.post("/setup/save_forced_point")
async def save_forced_point(request: Request):
    data     = await request.json()
    vertices = data.get("vertices", zone_vertices)
    phase    = state["setup_phase"]
    pan  = round(servo_angles["pan"],  1)
    tilt = round(servo_angles["tilt"], 1)
    existing = next((p for p in zone_cal["calibration_points"]
                     if _same_position(p, pan, tilt)), None)
    point = {"pan":pan,"tilt":tilt,
             "vertices":[list(v) for v in vertices],"user_set":True}
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
    pan  = round(servo_angles["pan"],  1)
    tilt = round(servo_angles["tilt"], 1)
    existing = next((p for p in zone_cal["calibration_points"]
                     if _same_position(p, pan, tilt)), None)
    point = {"pan":pan,"tilt":tilt,
             "vertices":[list(v) for v in vertices],"user_set":True}
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
    state["setup_phase"] = None
    return {"status":"complete"}

@app.get("/setup/reset")
def setup_reset():
    global zone_vertices, zone_closed
    zone_cal["calibration_points"] = []
    zone_cal["home_pan"]      = 90.0
    zone_cal["home_tilt"]     = 90.0
    zone_cal["depth_m"]       = None
    zone_cal["home_vertices"] = []
    zone_cal["home_closed"]   = False
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
    result = []
    for f in files:
        size = f.stat().st_size
        size_mb = round(size / 1024 / 1024, 1)
        # Try to get duration via cv2
        try:
            cap = cv2.VideoCapture(str(f))
            fps = cap.get(cv2.CAP_PROP_FPS) or 15
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            duration = round(frames / fps, 1) if fps > 0 else 0
        except Exception:
            duration = 0
        result.append({"name":f.name,"size_mb":size_mb,"duration_s":duration})
    return {"recordings":result}

@app.delete("/recordings/{filename}")
def delete_recording(filename: str):
    path = RECORDINGS_DIR / filename
    if path.exists(): path.unlink()
    return {"status":"deleted"}

@app.post("/recordings/{filename}/rename")
async def rename_recording(filename: str, request: Request):
    data = await request.json()
    new_name = data.get("name","").strip()
    if not new_name.endswith(".mp4"): new_name += ".mp4"
    src = RECORDINGS_DIR / filename
    dst = RECORDINGS_DIR / new_name
    if src.exists() and not dst.exists():
        src.rename(dst)
        return {"status":"renamed","name":new_name}
    return {"status":"error"}

@app.get("/recordings/{filename}")
def get_recording(filename: str):
    path = RECORDINGS_DIR / filename
    if path.exists(): return FileResponse(str(path), media_type="video/mp4")
    return {"error":"not found"}

@app.post("/reticle")
async def set_reticle(request: Request):
    data = await request.json()
    state["reticle_x"] = int(data.get("x", FRAME_W//2))
    state["reticle_y"] = int(data.get("y", FRAME_H//2))
    return {"reticle_x":state["reticle_x"],"reticle_y":state["reticle_y"]}

@app.get("/log")
def get_log():
    if LOG_FILE.exists():
        return FileResponse(str(LOG_FILE), media_type="text/plain", filename="catblastor_events.log")
    return {"error":"no log yet"}

@app.get("/log/clear")
def clear_log():
    LOG_FILE.write_text(""); log("LOG_CLEARED"); return {"status":"cleared"}

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
      <button class="btn gr" onclick="confirmReset()">Start Over</button>
    </div>
  </div>

  <div id="ph-forced" class="ph" style="display:none">
    <h3 id="ftitle">Forced Calibration</h3>
    <p id="fmsg"></p>
    <div class="ctrl">
      <button class="btn y" id="btn-tgt" onclick="toggleTgt()">▶ Targeting</button>
    </div>
    <p style="font-size:0.82em;color:#888;margin-top:4px">Drag zone or vertices to correct position.</p>
    <button class="btn g" id="btn-df" onclick="saveForcedPt()">✓ Done</button>
    <button class="btn gr" style="margin-left:8px" onclick="confirmReset()">Start Over</button>
  </div>

  <div id="ph-extra" class="ph" style="display:none">
    <p style="font-size:0.85em;color:#aaa">Pan/tilt anywhere, drag zone to correct it, then save the point.</p>
    <div class="ctrl">
      <button class="btn y" id="btn-tgt-e" onclick="toggleTgt()">▶ Targeting</button>
      <button class="btn g" onclick="savePointHere()">📍 Save Point Here</button>
      <button class="btn gr" onclick="confirmReset()">Start Over</button>
    </div>
    <div id="interp-debug" style="font-size:0.78em;color:#aaa;margin:6px 0;padding:6px;background:#111;border-radius:3px;line-height:1.6"></div>
  </div>

  <!-- Settings inside setup for live preview -->
  <div id="ph-settings" class="ph" style="margin-top:8px">
    <details open>
      <summary style="cursor:pointer;color:#aaf;font-size:0.95em;margin-bottom:8px">⚙ Firing Settings</summary>
      <div class="sg" style="margin-top:8px">
        <div class="set"><label>Firing Mode</label>
          <select id="firing_mode" onchange="updSettings()">
            <option value="single">Single Fire</option>
            <option value="semi_auto">Semi-Auto</option>
          </select></div>
        <div class="set"><label>Confidence: <span class="vl" id="vl-c">0.5</span></label>
          <input type="range" min="0.1" max="1" step="0.05" value="0.5" id="confidence_threshold"
                 oninput="document.getElementById('vl-c').textContent=this.value" onchange="updSettings()"></div>
        <div class="set"><label>Burst (s): <span class="vl" id="vl-b">1.0</span></label>
          <input type="range" min="0.1" max="5" step="0.1" value="1" id="burst_length"
                 oninput="document.getElementById('vl-b').textContent=this.value" onchange="updSettings()"></div>
        <div class="set"><label>Reload (s): <span class="vl" id="vl-r">10</span></label>
          <input type="range" min="1" max="60" step="1" value="10" id="reload_time"
                 oninput="document.getElementById('vl-r').textContent=this.value" onchange="updSettings()"></div>
        <div class="set"><label>Semi-Auto Delay (s): <span class="vl" id="vl-s">2</span></label>
          <input type="range" min="0.5" max="10" step="0.5" value="2" id="semi_auto_delay"
                 oninput="document.getElementById('vl-s').textContent=this.value" onchange="updSettings()"></div>
        <div class="set"><label>Tolerance (px): <span class="vl" id="vl-t">20</span></label>
          <input type="range" min="5" max="100" step="5" value="20" id="on_target_tolerance"
                 oninput="document.getElementById('vl-t').textContent=this.value" onchange="updSettings()"></div>
      </div>
    </details>
    <details style="margin-top:8px">
      <summary style="cursor:pointer;color:#aaf;font-size:0.95em;margin-bottom:8px">📷 Camera Settings</summary>
      <div class="sg" style="margin-top:8px">
        <div class="set"><label>Brightness (-1 to 1)</label>
          <div style="display:flex;gap:6px;align-items:center">
            <input type="range" min="-1" max="1" step="0.05" value="0" id="cam-brightness"
                   oninput="syncN('brightness',this.value)" onchange="updCam()">
            <input type="number" id="num-brightness" value="0" step="0.05" min="-1" max="1"
                   oninput="syncS('brightness',this.value)" onchange="updCam()"></div></div>
        <div class="set"><label>Contrast (0–2)</label>
          <div style="display:flex;gap:6px;align-items:center">
            <input type="range" min="0" max="2" step="0.05" value="1" id="cam-contrast"
                   oninput="syncN('contrast',this.value)" onchange="updCam()">
            <input type="number" id="num-contrast" value="1" step="0.05" min="0" max="2"
                   oninput="syncS('contrast',this.value)" onchange="updCam()"></div></div>
        <div class="set"><label>Saturation (0–2)</label>
          <div style="display:flex;gap:6px;align-items:center">
            <input type="range" min="0" max="2" step="0.05" value="1" id="cam-saturation"
                   oninput="syncN('saturation',this.value)" onchange="updCam()">
            <input type="number" id="num-saturation" value="1" step="0.05" min="0" max="2"
                   oninput="syncS('saturation',this.value)" onchange="updCam()"></div></div>
        <div class="set"><label>Sharpness (0–2)</label>
          <div style="display:flex;gap:6px;align-items:center">
            <input type="range" min="0" max="2" step="0.05" value="1" id="cam-sharpness"
                   oninput="syncN('sharpness',this.value)" onchange="updCam()">
            <input type="number" id="num-sharpness" value="1" step="0.05" min="0" max="2"
                   oninput="syncS('sharpness',this.value)" onchange="updCam()"></div></div>
        <div class="set"><label>EV Compensation (-4–4)</label>
          <div style="display:flex;gap:6px;align-items:center">
            <input type="range" min="-4" max="4" step="0.5" value="0" id="cam-ev"
                   oninput="syncN('ev',this.value)" onchange="updCam()">
            <input type="number" id="num-ev" value="0" step="0.5" min="-4" max="4"
                   oninput="syncS('ev',this.value)" onchange="updCam()"></div></div>
        <div class="set"><label>AWB Red Gain / Warm←→Cool (0.1–3)</label>
          <div style="display:flex;gap:6px;align-items:center">
            <input type="range" min="0.1" max="3" step="0.05" value="1" id="cam-awb_gain_r"
                   oninput="syncN('awb_gain_r',this.value)" onchange="updCam()">
            <input type="number" id="num-awb_gain_r" value="1" step="0.05" min="0.1" max="3"
                   oninput="syncS('awb_gain_r',this.value)" onchange="updCam()"></div></div>
        <div class="set"><label>AWB Blue Gain / Hue shift (0.1–3)</label>
          <div style="display:flex;gap:6px;align-items:center">
            <input type="range" min="0.1" max="3" step="0.05" value="1" id="cam-awb_gain_b"
                   oninput="syncN('awb_gain_b',this.value)" onchange="updCam()">
            <input type="number" id="num-awb_gain_b" value="1" step="0.05" min="0.1" max="3"
                   oninput="syncS('awb_gain_b',this.value)" onchange="updCam()"></div></div>
      </div>
    </details>
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
let reticleDragging = false;
let lastStatus = {reticle_x:320, reticle_y:240};
let curPhase  = null;
let camInit   = false;

// ── Page Nav ───────────────────────────────────────────────────────────────
function showPage(page, btn) {
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
  document.getElementById(page).classList.add('active');
  btn.classList.add('active');
  curPage = page;
  if(page==='recordings') loadRecs();
  if(page==='setup'){
    // When entering setup, always get current state and sync verts from server
    fetch('/setup/start').then(r=>r.json()).then(d=>{
      applyPhase(d.phase);
      // Immediately sync verts from server zone data
      fetch('/status').then(r=>r.json()).then(s=>{
        if(s.setup_zone && s.setup_zone.length > 0){
          verts = s.setup_zone.map(p=>[p[0],p[1]]);
          zClosed = s.setup_zone_closed;
        }
        lastServerPan = s.pan;
        lastServerTilt = s.tilt;
      });
    });
  }
  if(page!=='setup'){
    // When leaving setup, ensure setup_phase is cleared
    fetch('/setup/finish');
    curPhase = null;
  }
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
function confirmReset(){
  if(confirm('Delete all calibration data and start over? This cannot be undone.'))
    setupReset();
}
function setupReset(){verts=[];zClosed=false;fetch('/setup/reset').then(r=>r.json()).then(d=>applyPhase(d.phase));}
function clearZone(){verts=[];zClosed=false;sendZone();updDrawBtns();}
function closeZone(){if(verts.length<3)return;zClosed=true;sendZone();updDrawBtns();}
function beginForcedCal(){fetch('/setup/begin_forced_cal').then(r=>r.json()).then(d=>applyPhase(d.phase));}
function saveForcedPt(){
  fetch('/setup/save_forced_point',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({vertices:verts})}).then(r=>r.json()).then(d=>applyPhase(d.phase));
}
function savePointHere(){
  fetch('/setup/add_extra_point',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({vertices:verts})}).then(r=>r.json()).then(d=>{
    updStrength(d);
  });
}
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
  const nu=s.n_user_set||0;
  const lr=s.loocv_runs||0;
  if(pts) pts.textContent=`${n} saved (${nu} user-set) | LOOCV runs: ${lr}`;
  if(!fill||!text) return;
  if(s.combined!=null){
    fill.style.width=s.combined+'%';
    text.innerHTML=
      `<b style="color:#4af">Combined: ${s.combined}%</b><br>`+
      `LOOCV Position: ${s.position_confidence}% | Coverage: ${s.coverage_score}%`+
      (nu<8?`<br><span style="color:#fa0">⚠ Add ${8-nu} more user-set point${8-nu!==1?'s':''} for full strength</span>`:'');
  } else {
    fill.style.width='0%';
    text.textContent=n<3?`Need ${3-n} more point${3-n!==1?'s':''} for LOOCV`:'Calculating...';
  }
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
  if(curPage !== 'setup') return;
  const r=sc.getBoundingClientRect();
  const x=Math.round(e.clientX-r.left),y=Math.round(e.clientY-r.top);

  // When targeting active, dragging reticle takes priority
  if(tgtOn){
    const rx=lastStatus.reticle_x||320, ry=lastStatus.reticle_y||240;
    if(Math.sqrt((x-rx)**2+(y-ry)**2)<30){
      reticleDragging=true; sc.style.cursor='crosshair'; return;
    }
  }

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
  if(reticleDragging){
    // Draw reticle at new position immediately (overlay will update on next status poll)
    lastStatus.reticle_x=x; lastStatus.reticle_y=y;
    return;
  }
  if(dragIdx>=0){
    verts[dragIdx]=[x-dragOff.x,y-dragOff.y];
  } else if(zoneDrag&&zoneDragStart&&zoneDragOrigin){
    const dx=x-zoneDragStart.x, dy=y-zoneDragStart.y;
    verts=zoneDragOrigin.map(v=>[v[0]+dx,v[1]+dy]);
  }
});

sc.addEventListener('mouseup',e=>{
  if(reticleDragging){
    reticleDragging=false;
    sc.style.cursor='crosshair';
    const r=sc.getBoundingClientRect();
    const x=Math.round(e.clientX-r.left),y=Math.round(e.clientY-r.top);
    fetch('/reticle',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({x,y})});
    return;
  }
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

    // Reticle
    const rx = d.reticle_x||320, ry = d.reticle_y||240;
    ctx.strokeStyle='#ff4400'; ctx.lineWidth=2;
    ctx.beginPath(); ctx.arc(rx,ry,18,0,2*Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(rx-28,ry); ctx.lineTo(rx+28,ry); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(rx,ry-28); ctx.lineTo(rx,ry+28); ctx.stroke();
    ctx.fillStyle='#ff4400'; ctx.beginPath(); ctx.arc(rx,ry,3,0,2*Math.PI); ctx.fill();
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
      awb_gain_r:parseFloat(document.getElementById('cam-awb_gain_r').value),
      awb_gain_b:parseFloat(document.getElementById('cam-awb_gain_b').value),
    })});
}

// ── Status Polling ─────────────────────────────────────────────────────────
setInterval(()=>{
  fetch('/status').then(r=>r.json()).then(d=>{
    lastStatus = d;
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

    // Interpolation debug in extra phase
    if(curPhase==='extra'){
      const el=document.getElementById('interp-debug');
      if(el){
        const n=d.n_cal_points||0;
        const tot=d.n_total_positions||361;
        const pct=Math.round(n/tot*100);
        const exact=d.setup_zone_exact?'<span style="color:#0f8">● Exact saved point</span>':'<span style="color:#fa0">○ Interpolated</span>';
        el.innerHTML=`${exact} | Grid: Pan ${d.grid_pan}° (${d.grid_pan_pct}%) Tilt ${d.grid_tilt}° (${d.grid_tilt_pct}%)<br>`+
          `Saved ${n}/${tot} positions (${pct}%) — pan/tilt to explore, zone auto-saves`;
      }
    }

    // Camera settings init
    if(d.cam&&!camInit){
      camInit=true;
      ['brightness','contrast','saturation','sharpness','ev','awb_gain_r','awb_gain_b'].forEach(k=>{
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
      <div class="ri" style="flex-direction:column;align-items:flex-start;gap:6px">
        <div style="display:flex;justify-content:space-between;width:100%;align-items:center">
          <span id="rn-${CSS.escape(f.name)}" style="font-size:0.9em">${f.name}</span>
          <span style="color:#888;font-size:0.8em">${f.size_mb}MB · ${f.duration_s}s</span>
        </div>
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          <a href="/recordings/${f.name}" download><button class="btn b" style="padding:4px 10px;font-size:0.8em">⬇ Download</button></a>
          <a href="/recordings/${f.name}" target="_blank"><button class="btn gr" style="padding:4px 10px;font-size:0.8em">▶ Play</button></a>
          <button class="btn y" style="padding:4px 10px;font-size:0.8em" onclick="renameRec('${f.name}')">✏ Rename</button>
          <button class="btn r" style="padding:4px 10px;font-size:0.8em" onclick="deleteRec('${f.name}')">✕ Delete</button>
        </div>
      </div>`).join('');
  });
}

function deleteRec(name){
  if(!confirm('Delete '+name+'?')) return;
  fetch('/recordings/'+name,{method:'DELETE'}).then(()=>loadRecs());
}

function renameRec(name){
  const newName=prompt('Rename to:',name.replace('.mp4',''));
  if(!newName) return;
  fetch('/recordings/'+name+'/rename',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({name:newName})}).then(()=>loadRecs());
}
</script>
</body>
</html>
"""
