import time
import cv2
import serial
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, models

MODEL_PATH = "C:/Users/salma/OneDrive/Desktop/images_sort_bin/eSort_class_best.pt"
COM_PORT = "COM6"
BAUDRATE = 115200

CLASS_NAMES = ['biodegradable','hazardous','nonrecyclable','papers','recyclable']

# Map class -> Arduino command
CLASS_TO_CMD = {
    'biodegradable': 'C:1',
    'recyclable':    'C:2',
    'papers':        'C:3',
    'nonrecyclable': 'C:4',
    'hazardous':     'C:5'
}

SKIP_HAZARDOUS_SEND = False   # If True, will NOT send any serial command for hazardous items.

CENTER_RATIO = 0.70       # center crop bounding box ratio (0.0-1.0)
CONF_THRESHOLD = 0.83     # accept detection when probability >= this
COOLDOWN_S = 1.0          # seconds after DONE before allowing next send
SERIAL_TIMEOUT_S = 8.0    # how long to wait for DONE from Arduino
SHOW_WINDOW = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hazard-clear tuning:
HAZARD_CLEAR_FRAMES = 6   # number of consecutive frames that must NOT show hazardous before sending CLEAR
# ----------------------------------------------------

# Preprocess (EfficientNet-B0 style)
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def open_serial(port, baud, timeout=1.0):
    try:
        ser = serial.Serial(port, baud, timeout=0.5)  # short timeout for reads
        time.sleep(2.0)  # allow Arduino to reset
        # flush startup prints
        try:
            while ser.in_waiting:
                _ = ser.readline()
        except Exception:
            pass
        print(f"[SERIAL] Opened {port} @ {baud}")
        return ser
    except Exception as e:
        print("[SERIAL] Could not open serial:", e)
        return None

def load_classifier(path, num_classes):
    print("[MODEL] Loading", path)
    net = models.efficientnet_b0(pretrained=False)
    try:
        if isinstance(net.classifier, torch.nn.Sequential):
            in_f = net.classifier[1].in_features
        else:
            in_f = net.classifier.in_features
    except Exception:
        in_f = 1280
    net.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=True),
                                         torch.nn.Linear(in_f, num_classes))

    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if 'model' in ckpt and isinstance(ckpt['model'], dict):
            state = ckpt['model']
            net.load_state_dict(state, strict=False)
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
            new_state = {k.replace('module.','') if k.startswith('module.') else k: v for k,v in state.items()}
            net.load_state_dict(new_state, strict=False)
        else:
            # treat as raw state_dict
            sample = list(ckpt.items())[:1]
            if sample and isinstance(sample[0][1], torch.Tensor):
                new_state = {k.replace('module.','') if k.startswith('module.') else k: v for k,v in ckpt.items()}
                net.load_state_dict(new_state, strict=False)
            else:
                # try full model object
                try:
                    ckpt.eval()
                    return ckpt
                except Exception as e:
                    raise RuntimeError("Unknown checkpoint format: " + str(e))
    else:
        try:
            ckpt.eval()
            return ckpt
        except Exception as e:
            raise RuntimeError("Can't load checkpoint: " + str(e))

    net.to(DEVICE)
    net.eval()
    print("[MODEL] Loaded classifier to", DEVICE)
    return net

def classify_image(model, img):
    x = preprocess(img)
    x = x.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return idx, probs

def center_crop(frame, ratio):
    h,w = frame.shape[:2]
    size = int(min(w,h) * ratio)
    cx, cy = w//2, h//2
    x1 = max(0, cx - size//2); y1 = max(0, cy - size//2)
    x2 = min(w, cx + size//2); y2 = min(h, cy + size//2)
    return frame[y1:y2, x1:x2], (x1,y1,x2,y2)

def send_and_wait_done(ser, cmd, timeout_s=SERIAL_TIMEOUT_S, expect_done=True):
    """
    Send `cmd` (like 'C:2') and wait for Arduino reply.
    If expect_done==True: wait for 'DONE' (returns True on DONE).
    If expect_done==False: wait for 'RECEIVED' only (returns True if RECEIVED seen).
    Returns True on expected ack, False otherwise.
    """
    if ser is None:
        print("[SERIAL] Serial not open — dry-run: would send", cmd)
        return False

    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    s = cmd.strip() + "\n"
    try:
        ser.write(s.encode())
    except Exception as e:
        print("[SERIAL] Write error:", e)
        return False

    start = time.time()
    got_received = False
    while time.time() - start < timeout_s:
        try:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode(errors="ignore").strip()
        except Exception:
            continue
        if not line:
            continue
        print("[ARDUINO]", line)
        if "RECEIVED" in line:
            got_received = True
            if not expect_done:
                return True
        if "DONE" in line:
            return True
    # timed out
    if not expect_done:
        print("[TIMEOUT] No RECEIVED received for", cmd)
    else:
        print("[TIMEOUT] No DONE received for", cmd, "(got RECEIVED=" + str(got_received) + ")")
    return False

def main():
    model = load_classifier(MODEL_PATH, len(CLASS_NAMES))
    ser = open_serial(COM_PORT, BAUDRATE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open camera (index 0).")
        return

    last_done_time = 0.0
    hazard_pending = False
    non_hazard_frames = 0

    print("[INFO] Press q or ESC in the camera window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No camera frame. Exiting.")
            break

        crop, (x1,y1,x2,y2) = center_crop(frame, CENTER_RATIO)

        # classify crop
        try:
            idx, probs = classify_image(model, crop)
            classname = CLASS_NAMES[idx]
            prob = float(probs[idx])
        except Exception as e:
            classname = None
            prob = 0.0
            print("[MODEL ERR]", e)

        now = time.time()

        # overlay
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
        if classname is None or prob < CONF_THRESHOLD:
            cv2.putText(frame, "No item detected", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2)
        else:
            cv2.putText(frame, f"{classname} {prob:.2f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2)

        # Hazard handling: if hazard pending wait for clear condition
        if hazard_pending:
            # if current frame is NOT hazardous (either different class or below threshold)
            if not (classname == 'hazardous' and prob >= CONF_THRESHOLD):
                non_hazard_frames += 1
            else:
                non_hazard_frames = 0

            # once we've seen N consecutive non-hazard frames, send CLEAR
            if non_hazard_frames >= HAZARD_CLEAR_FRAMES:
                print("[HAZARD] No hazardous seen for", non_hazard_frames, "frames -> sending CLEAR")
                ok = send_and_wait_done(ser, "CLEAR", timeout_s=SERIAL_TIMEOUT_S, expect_done=True)
                if ok:
                    hazard_pending = False
                    last_done_time = time.time()
                    non_hazard_frames = 0
                    print("[HAZARD] Cleared (Arduino DONE). Resuming normal ops.")
                else:
                    print("[HAZARD] CLEAR timed out or failed. Will retry after more frames.")
                    
            
            # show status in terminal
            if SHOW_WINDOW:
                cv2.putText(frame, "HAZARD MODE - wait to clear", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            if SHOW_WINDOW:
                cv2.imshow("eSort Classifier", frame)
                k = cv2.waitKey(10) & 0xFF
                if k == ord('q') or k == 27:
                    break
            continue  # go next frame loop without sending other commands

        # Normal decision 
        if classname is not None and prob >= CONF_THRESHOLD and (now - last_done_time) > COOLDOWN_S:
            if classname not in CLASS_TO_CMD:
                print("[WARN] class not mapped:", classname)
            else:
                if classname == 'hazardous':
                    if SKIP_HAZARDOUS_SEND:
                        print(f"[ACTION] Detected hazardous ({prob:.2f}) - skipping serial send (configured).")
                        last_done_time = time.time()
                    else:
                        cmd = CLASS_TO_CMD[classname]
                        print(f"[ACTION] Detected hazardous ({prob:.2f}) -> sending {cmd} (expect RECEIVED, wait for CLEAR later)")
                        ok = send_and_wait_done(ser, cmd, timeout_s=4.0, expect_done=False)
                        if ok:
                            hazard_pending = True
                            non_hazard_frames = 0
                            print("[HAZARD] Hazard acknowledged by Arduino (RECEIVED). Waiting for CLEAR.")
                        else:
                            # failed to get RECEIVED - report but do not set hazard_pending
                            print("[WARN] Hazard command write failed or no RECEIVED - try again later.")
                else:
                    cmd = CLASS_TO_CMD[classname]
                    print(f"[ACTION] Detected {classname} ({prob:.2f}) -> sending {cmd}")
                    ok = send_and_wait_done(ser, cmd, timeout_s=SERIAL_TIMEOUT_S, expect_done=True)
                    if ok:
                        last_done_time = time.time()
                        print("[INFO] Movement complete.")
                    else:
                        print("[WARN] No DONE ack — check Arduino or connection.")

        # show frame and key handling
        if SHOW_WINDOW:
            cv2.imshow("eSort Classifier", frame)
            k = cv2.waitKey(10) & 0xFF
            if k == ord('q') or k == 27:
                break

    cap.release()
    if ser:
        ser.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
