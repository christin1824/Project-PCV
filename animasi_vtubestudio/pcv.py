import cv2
import mediapipe as mp
import json
import websocket
import threading
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import time
import math

# --- VTS Config ---
VTS_URL = "ws://localhost:8001"
PLUGIN_NAME = "Python Body Tracker"
PLUGIN_DEVELOPER = "Natania"

# --- Initialize MediaPipe Holistic (pose + face + hands) ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Global WebSocket variable ---
ws = None
auth_token = None
connected = False
last_param_sent = None
last_send_time = 0

# --- Eye parameter configuration ---
# Nama parameter di model (sesuaikan dengan yang ada di VTube Studio)
EYE_PARAM_NAME = "ParamEyeOpen"
# Output range yang sesuai dengan setting VTS (sesuaikan jika perlu)
EYE_OUT_MIN = 0.0
EYE_OUT_MAX = 0.55
# Calibration EAR min/max (empirik). Jika hasil kebuka/ketutup terbalik,
# invert atau adjust nilai ini.
EYE_EAR_MIN = 0.12
EYE_EAR_MAX = 0.30

# --- GUI setup ---
root = tk.Tk()
root.title("Body Tracking + VTube Studio")
root.geometry("900x700")

video_label = Label(root)
video_label.pack()

status_label = Label(root, text="Status: Belum Terhubung", font=("Arial", 12), fg="red")
status_label.pack(pady=10)


# === Function to connect to VTube Studio ===
def connect_vtube():
    global ws
    try:
        ws = websocket.WebSocket()
        ws.connect(VTS_URL)
        status_label.config(text="Terhubung ke VTube Studio!", fg="green")
        print("[✅] Terhubung ke VTube Studio!")

        # Kirim request plugin info (agar muncul Allow/Deny)
        req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth_req",
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_DEVELOPER
            }
        }
        ws.send(json.dumps(req))
        print("[📨] Mengirim AuthenticationTokenRequest...")

        resp = json.loads(ws.recv())
        print("[🔍] Response dari VTS:", resp)

        if "data" in resp and "authenticationToken" in resp["data"]:
            global auth_token
            auth_token = resp["data"]["authenticationToken"]

            # Kirim AuthenticationRequest untuk memunculkan prompt Allow/Deny
            auth_req = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "auth_final",
                "messageType": "AuthenticationRequest",
                "data": {
                    "pluginName": PLUGIN_NAME,
                    "pluginDeveloper": PLUGIN_DEVELOPER,
                    "authenticationToken": auth_token
                }
            }
            ws.send(json.dumps(auth_req))
            print("[📨] Mengirim AuthenticationRequest...")
            final_resp = json.loads(ws.recv())
            print("[🔍] Response:", final_resp)

            if "authenticated" in final_resp.get("data", {}):
                if final_resp["data"]["authenticated"]:
                    status_label.config(text="Autentikasi Berhasil ✅", fg="green")
                    print("[✅] Autentikasi Berhasil!")
                    # Tandai koneksi siap digunakan untuk mengirim parameter
                    global connected
                    connected = True
                    # Setelah autentikasi, minta daftar parameter model dan tampilkan
                    try:
                        params = request_parameter_list()
                        if params:
                            print("[ℹ️] Daftar parameter model:")
                            for p in params:
                                print("   -", p)
                        else:
                            print("[⚠️] Tidak menerima daftar parameter dari VTS.")
                    except Exception as e:
                        print("[⚠️] Gagal mengambil daftar parameter:", e)
                else:
                    status_label.config(text="Autentikasi Gagal ❌", fg="red")
                    print("[❌] Autentikasi Gagal!")
        else:
            status_label.config(text="Token tidak diterima ❌", fg="red")

    except Exception as e:
        print("[⚠️] Gagal terhubung ke VTS:", e)
        status_label.config(text=f"Gagal terhubung ke VTS: {e}", fg="red")


def send_model_parameter(parameter_name, value, model_name=None):
    """Kirim SetModelParameter ke VTube Studio.

    Format request mengikuti VTube Studio Public API (wiki). Jika belum
    terhubung, fungsi ini tidak melakukan apa-apa.
    """
    global ws, connected, auth_token, last_param_sent, last_send_time
    try:
        if not connected or ws is None:
            return False

        # Throttle: hanya kirim jika beda signifikan atau lebih dari 0.05s
        now = time.time()
        if last_param_sent is not None and abs(last_param_sent - value) < 0.002 and (now - last_send_time) < 0.05:
            return False

        req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"set_param_{int(now*1000)}",
            "messageType": "SetModelParameter",
            "data": {
                "parameter": parameter_name,
                "value": float(value),
                "modelName": model_name
            }
        }
        ws.send(json.dumps(req))
        last_param_sent = float(value)
        last_send_time = now
        return True
    except Exception as e:
        print("[⚠️] Gagal mengirim parameter ke VTS:", e)
        return False


def request_parameter_list(timeout=1.0):
    """Minta daftar parameter model dari VTube Studio dan kembalikan list nama parameter.

    Mengirim messageType 'GetParameterList' (atau fallback ke 'GetModelInfo') dan
    membalas ws.recv(). Karena format response bervariasi, fungsi ini mencari
    property yang umum seperti 'parameters' atau 'parameterList'.
    """
    global ws, connected
    if not connected or ws is None:
        return None

    req = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": f"get_params_{int(time.time()*1000)}",
        "messageType": "GetParameterList",
        "data": {}
    }
    try:
        ws.send(json.dumps(req))
        # coba baca response
        ws.settimeout(timeout)
        resp_raw = ws.recv()
        try:
            resp = json.loads(resp_raw)
        except Exception:
            return None

        # response mungkin berisi data.parameters atau data.parameterList atau data.parameters
        data = resp.get("data", {})
        candidates = []
        for key in ("parameters", "parameterList", "parameterNames"):
            if key in data and isinstance(data[key], list):
                # parameter item bisa berupa dict atau string
                for item in data[key]:
                    if isinstance(item, dict):
                        # common fields: 'id' or 'name' or 'parameter'
                        for f in ("id", "name", "parameter", "parameterId"):
                            if f in item:
                                candidates.append(item[f])
                                break
                    elif isinstance(item, str):
                        candidates.append(item)
                if candidates:
                    return candidates

        # fallback: coba GetModelInfo
        req2 = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"get_modelinfo_{int(time.time()*1000)}",
            "messageType": "GetModelInfo",
            "data": {}
        }
        ws.send(json.dumps(req2))
        resp_raw = ws.recv()
        resp = json.loads(resp_raw)
        data = resp.get("data", {})
        # model info may include 'parameters'
        params = data.get("parameters") or data.get("parameterList")
        out = []
        if isinstance(params, list):
            for p in params:
                if isinstance(p, dict):
                    for f in ("id", "name", "parameter"):
                        if f in p:
                            out.append(p[f])
                            break
                elif isinstance(p, str):
                    out.append(p)
        return out
    except Exception as e:
        # jangan crash script kalau gagal
        print("[⚠️] request_parameter_list error:", e)
        try:
            ws.settimeout(None)
        except Exception:
            pass
        return None


# === Function for camera & MediaPipe frame ===
def camera_loop():
    cap = cv2.VideoCapture(0)
    # Simple exponential smoothing for parameter to reduce jitter
    smoothing_alpha = 0.25  # 0..1, lebih kecil = lebih halus
    smoothed_param = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # Draw all landmarks
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Convert to ImageTk for Tkinter display
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        root.update_idletasks()
        root.update()

        # --- compute left wrist -> model parameter mapping ---
        try:
            if results.left_hand_landmarks:
                wrist = results.left_hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
                # MediaPipe landmarks are normalized (x,y in 0..1). Use y for vertical mapping.
                wrist_y = wrist.y
                # Invert so that raising hand (smaller y) -> larger value (optional)
                normalized = 1.0 - wrist_y
                # Clip just in case
                normalized = max(0.0, min(1.0, normalized))

                # Map normalized [0..1] to output range (VTube setting -10..10 in your screenshot)
                out_min, out_max = -10.0, 10.0
                mapped = normalized * (out_max - out_min) + out_min

                # Apply simple exponential smoothing to reduce jitter before sending
                if smoothed_param is None:
                    smoothed_param = mapped
                else:
                    smoothed_param = smoothed_param * (1 - smoothing_alpha) + mapped * smoothing_alpha

                # Kirim ke VTube (nama parameter sesuai yang kamu pakai)
                send_model_parameter("PARAM_ARM_L_A", smoothed_param)
        except Exception:
            pass

        time.sleep(0.01)

    cap.release()


# === Run threads ===
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=connect_vtube, daemon=True).start()

root.mainloop()
