#!/usr/bin/env python3
"""
Persistent inference subprocess using Google Coral Edge TPU.
Reads frames from stdin as pickled (frame, target_class, confidence_threshold),
writes detection results to stdout as pickled list of dicts.
Protocol: 4-byte big-endian length prefix + pickle payload.
Runs under Python 3.9 with coral-env.
"""
import sys
import pickle
import numpy as np
import cv2
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

MODEL_PATH = sys.argv[1]
FRAME_W    = int(sys.argv[2])
FRAME_H    = int(sys.argv[3])

# Load model onto Edge TPU
interp = make_interpreter(MODEL_PATH)
interp.allocate_tensors()
input_shape = interp.get_input_details()[0]['shape']  # [1, H, W, 3]
MODEL_H, MODEL_W = input_shape[1], input_shape[2]

sys.stderr.write(f"Coral inference ready: model={MODEL_PATH} input={MODEL_W}x{MODEL_H}\n")
sys.stderr.flush()

while True:
    try:
        # Read length-prefixed pickle from stdin
        raw_len = sys.stdin.buffer.read(4)
        if len(raw_len) < 4:
            break
        length  = int.from_bytes(raw_len, 'big')
        payload = sys.stdin.buffer.read(length)
        if len(payload) < length:
            break

        frame, target_class, confidence_threshold = pickle.loads(payload)

        # Resize frame to model input size
        resized = cv2.resize(frame, (MODEL_W, MODEL_H))
        resized = resized.astype(np.uint8)

        # Run inference
        common.set_input(interp, resized)
        interp.invoke()
        objs = detect.get_objects(interp, score_threshold=confidence_threshold)

        # Scale boxes back to original frame size
        sx = FRAME_W / MODEL_W
        sy = FRAME_H / MODEL_H

        dets = []
        for obj in objs:
            if obj.id != target_class:
                continue
            x1 = int(obj.bbox.xmin * sx)
            y1 = int(obj.bbox.ymin * sy)
            x2 = int(obj.bbox.xmax * sx)
            y2 = int(obj.bbox.ymax * sy)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            dets.append({
                "id":   f"{cx}_{cy}",
                "x1":   x1, "y1": y1, "x2": x2, "y2": y2,
                "cx":   cx, "cy": cy,
                "conf": float(obj.score),
                "in_zone": False
            })

        # Write length-prefixed pickle result to stdout
        result  = pickle.dumps(dets)
        rlen    = len(result).to_bytes(4, 'big')
        sys.stdout.buffer.write(rlen + result)
        sys.stdout.buffer.flush()

    except Exception as e:
        sys.stderr.write(f"coral_infer error: {e}\n")
        sys.stderr.flush()
        # Write empty result so main process doesn't hang
        result = pickle.dumps([])
        sys.stdout.buffer.write(len(result).to_bytes(4, 'big') + result)
        sys.stdout.buffer.flush()
