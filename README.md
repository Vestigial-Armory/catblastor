# CatBlastor

Autonomous fixed-mount water turret that uses computer vision to detect cats (or any COCO object class) in a user-defined forbidden zone and deters them with targeted water spray. Pan-tilt servo bracket moves camera and nozzle together as a unit.

---

## Hardware

| Component | Detail |
|---|---|
| SBC | Raspberry Pi 4 2GB, hostname `catblastor.local`, user `wolfhard` |
| Camera | Camera Module 3 NoIR — mounted **upside down**, requires `--vflip --hflip` |
| Servos | MG996R × 2 (pan/tilt), PCA9685 PWM driver (I2C 0x40) via Adafruit ServoKit |
| Servo channels | Pan = channel 0, Tilt = channel 1 |
| Pump | 800L/h submersible 12V pump, L298N motor driver, GPIO BCM 27 (OUT3/4) |
| Solenoid | 12V normally-closed solenoid valve, GPIO BCM 17 (OUT1/2) via L298N |
| Audio | PAM8302A amp + 8Ω 2W speaker — plays via ffplay/mpg123/aplay |
| Power | 12V 5A PSU + buck converter 12V→5V for Pi |

### Servo Constants
```
PAN_MIN=45°  PAN_MAX=135°  PAN_CENTER=90°  PAN_RANGE=45°
TILT_MIN=45° TILT_MAX=135° TILT_CENTER=90° TILT_RANGE=45°
```

**Confirmed directions:**
- Increasing pan → camera pans LEFT → scene shifts RIGHT
- Increasing tilt → camera tilts DOWN → scene shifts UP (cat moves up in frame)
- Tracking uses `-sign(err_x)` for pan, `+sign(err_y)` for tilt

---

## Software Stack

- Raspberry Pi OS Lite 64-bit, headless
- Python 3.13, venv at `~/catblastor-env`
- FastAPI + Uvicorn web server on port 8080
- rpicam-vid (MJPEG) for camera streaming
- YOLOv8n NCNN model for inference (~2.2fps on Pi 4 CPU)
- Inference runs in a **separate process** (not thread) to avoid GIL blocking servo/GPIO threads

### Run
```bash
cd ~/catblastor && source ~/catblastor-env/bin/activate && uvicorn main:app --host 0.0.0.0 --port 8080
```

### Key dependencies
```
fastapi uvicorn ultralytics torch numpy RPi.GPIO adafruit-circuitpython-servokit opencv-python
```

---

## Architecture

### Threads (all daemon threads in main process)
| Thread | Function | Purpose |
|---|---|---|
| `start_streaming` | Launches rpicam-vid subprocess | Camera MJPEG source |
| `mjpeg_reader_loop` | Parses MJPEG stream from rpicam-vid stdout | Buffers latest frame, feeds recording |
| `capture_loop` | Decodes MJPEG→numpy for inference | Feeds inference at ~2fps |
| `inference_loop` | Sends frames to inference process, reads results | Updates `latest_detections` |
| `servo_tracking_loop` | Reads detections, steps servos toward cat centroid | 20fps, 500ms freshness timeout |
| `firing_loop` | Controls pump and solenoid based on detection state | Independent pump/solenoid logic |
| `targeting_loop` | Manual targeting calibration mode GPIO control | Only active during calibration |
| `recording_loop` | Manages ffmpeg recording start/stop | Triggered by cat detection |
| `home_position_loop` | Returns camera to home after 5s idle | No cat + no user input for 5s |
| `audio_loop` | Plays audio deterrent during firing events | Per reinforcement schedule |

### Separate Process
- `_inference_worker` — runs YOLO in its own process with its own GIL. Receives frames via `mp.Queue`, returns detection results via `mp.Queue`. Eliminates inference blocking servo/HTTP response latency.

### Camera Pipeline
```
rpicam-vid (MJPEG, sensor mode 2304x1296 full FOV, output 640x480)
  → mjpeg_reader_loop (buffers latest JPEG, also writes to ffmpeg for recording)
  → capture_loop (decodes to numpy, downscales to 320x240 for inference)
  → _inference_worker (YOLOv8n NCNN, returns bbox list)
  → inference_loop (zone check, updates latest_detections, increments inference_seq)
```

### Firing Logic

**Pump** (independent of solenoid):
- ON when any cat detected in zone
- OFF 2 seconds after zone clears

**Solenoid** fires only when ALL four conditions are simultaneously true:
1. Cat detected anywhere in frame
2. Cat has been in zone within last 2 seconds
3. Reticle pixel position is within `on_target_tolerance` px of cat bounding box edge
4. Firing mode conditions met (reload timer / semi-auto delay)

Solenoid stops immediately when any condition fails.

**Manual fire sequence** (user-triggered, ignores all above conditions):
pump ON → 0.5s → solenoid ON → 1.0s → solenoid OFF → 0.5s → pump OFF

**`_manual_firing` flag** prevents `firing_loop` and `targeting_loop` from overriding GPIO during manual fire.

### Tracking
Pixel-space closed-loop. Each inference cycle:
- `err_x = cat_cx - reticle_x`, `err_y = cat_cy - reticle_y`
- `step_pan = -sign(err_x) * min(3.0, |err_x|/40 * 3.0)`
- `step_tilt = +sign(err_y) * min(3.0, |err_y|/40 * 3.0)`
- Dead zone: 10px both axes
- Tracking loop runs at 20fps, skips moves if last inference >700ms ago

### Zone Calibration
- User draws forbidden zone polygon in Setup tab
- Zone stored in pixel-space per calibration point `{pan, tilt, vertices[]}`
- Multiple calibration points allow zone to follow camera movement via linear regression interpolation
- LOOCV calibration strength metric, capped at 50% until 8 user-set points
- Live mode: always interpolates zone for current servo position

### Recording
- ffmpeg subprocess receives MJPEG frames directly from `mjpeg_reader_loop` (true 30fps)
- Encoded as H264 libx264 ultrafast with `-movflags +faststart` for browser playback
- Auto-starts on cat detection, stops 10s after last detection
- Test recording (10s) available in Setup tab

### Audio Deterrent
- Plays audio file when firing conditions active
- Plays to completion, 1s silence, checks if still firing, repeats
- Reinforcement schedule: paired / sound-always-spray-pct / sound-always-spray-prob / spray-always-sound-pct / spray-always-sound-prob
- Probability/percentage applies per firing EVENT not per repetition
- Playback via ffplay → aplay/mpg123 fallback

---

## Persisted Files

| File | Contents |
|---|---|
| `settings.json` | firing_mode, burst_length, reload_time, semi_auto_delay, confidence_threshold, on_target_tolerance, target_class |
| `zone_calibration.json` | All calibration points with pan/tilt/vertices, depth_m |
| `home_position.json` | Home pan/tilt angles |
| `reticle.json` | Reticle pixel position |
| `camera_settings.json` | Brightness, contrast, saturation, sharpness, EV, AWB gains |
| `audio_settings.json` | enabled, schedule_mode, pct, probability, volume, active_file |
| `catblastor_events.log` | ARM/DISARM/DETECT/ZONE_IN/ZONE_OUT/LOST/TRACK/FIRE/MANUAL_FIRE events |
| `recordings/` | MP4 recordings, auto-named `catblastor_YYYYMMDD_HHMMSS.mp4` |
| `audio/` | Uploaded audio deterrent files |
| `yolov8n_ncnn_model/` | NCNN model files |

---

## API Routes

### Control
| Method | Route | Description |
|---|---|---|
| GET | `/` | Web UI (single-page HTML) |
| GET | `/arm` | Arm the system |
| GET | `/disarm` | Disarm the system |
| GET | `/stream` | MJPEG stream |
| GET | `/status` | Full system state JSON |
| GET | `/fire/manual` | Manual fire sequence |
| GET | `/log` | Download event log |
| GET | `/log/clear` | Clear event log |

### Servos
| Method | Route | Description |
|---|---|---|
| POST | `/servos/move` | `{pan_delta, tilt_delta}` relative move |
| GET | `/servos/center` | Move to 90/90 |
| GET | `/servos/home/set` | Save current position as home |
| GET | `/servos/home/go` | Return to home |

### Settings
| Method | Route | Description |
|---|---|---|
| POST | `/settings` | Update firing/detection settings |
| POST | `/camera` | Update camera settings |
| POST | `/reticle` | `{x, y}` set reticle pixel position |

### Setup / Calibration
| Method | Route | Description |
|---|---|---|
| GET | `/setup/start` | Enter setup mode |
| GET | `/setup/finish` | Exit setup mode |
| GET | `/setup/reset` | Reset calibration |
| GET | `/setup/set_home` | Set home in setup flow |
| POST | `/setup/zone` | Save zone polygon for current position |
| GET | `/setup/begin_forced_cal` | Start forced calibration step |
| POST | `/setup/save_forced_point` | Save forced calibration point |
| POST | `/setup/add_extra_point` | Add extra calibration point |
| POST | `/setup/remove_point` | Remove calibration point |
| POST | `/setup/goto_point` | Move servos to saved point |
| GET | `/setup/cal_points` | List calibration points |
| POST | `/setup/depth` | Set zone depth in meters |
| GET | `/setup/targeting/start` | Start targeting calibration mode |
| GET | `/setup/targeting/stop` | Stop targeting calibration mode |
| GET | `/setup/interp_debug` | Interpolation debug info |

### Recordings
| Method | Route | Description |
|---|---|---|
| GET | `/recordings` | List recordings with metadata |
| DELETE | `/recordings/{filename}` | Delete recording |
| POST | `/recordings/{filename}/rename` | Rename recording |
| GET | `/recording/test/start` | Start 10s test recording |
| GET | `/recording/test/stop` | Stop test recording |
| GET | `/rec_files/{filename}` | Serve recording file (StaticFiles mount) |

### Audio
| Method | Route | Description |
|---|---|---|
| GET | `/audio/files` | List uploaded audio files |
| POST | `/audio/upload` | Upload audio (raw body, `x-filename` header) |
| POST | `/audio/settings` | Update audio deterrent settings |
| GET | `/audio/preview` | Preview active audio file |
| DELETE | `/audio/files/{filename}` | Delete audio file |

---

## Known Issues / Pending

- **Manual fire not working** — `_manual_firing` flag added to prevent GPIO race but firing still not activating consistently. `firing_loop` sets `firing["active"] = solenoid_allowed` every 50ms which may still interfere. Needs investigation.
- **Audio preview** — ffplay may not be installed on Pi; fallback to aplay/mpg123 added. Confirm `which ffplay mpg123 aplay` on Pi.
- Detection IDs use `cx_cy` string which changes every frame — causes ZONE_IN/ZONE_OUT log churn. Consider bbox overlap tracking.
- Batch delete/download for recordings not yet implemented.
- Cloudflare Tunnel for remote HTTPS access deferred.
- Enclosure/box design pending.

---

## Development Rules (learned the hard way)

1. **Always confirm plan with user before writing any code**
2. After every `str_replace`, verify the decorator above wasn't accidentally removed
3. Run `python3 -c "import ast; ast.parse(open('main.py').read())"` before every commit
4. Never insert code before a variable it depends on — ordering bugs are common
5. When reverting: `git checkout <hash> -- main.py`
6. All GPIO writes must go through `set_pump()`, `set_solenoid()`, or `set_gpio()` — never call `GPIO.output()` directly outside of initial setup and the helpers themselves
7. `inference_seq` requires `global inference_seq` declaration in `inference_loop`
8. Multiprocessing worker cannot access main process globals — pass all needed values via queue
9. StaticFiles mounts must use different paths than API routes (use `/rec_files/` not `/recordings/`)
10. `atexit` handles cleanup of inference subprocess on Ctrl+C — do not add competing signal handlers
