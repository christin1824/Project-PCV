import cv2
import mediapipe as mp
import json
import websocket
import threading
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import math
import time

# --- VTS Config ---
VTS_URL = "ws://localhost:8001"
PLUGIN_NAME = "Python Face Tracker"
PLUGIN_DEVELOPER = "Natania"

# --- Initialize MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Global Variables ---
ws = None
connected = False
last_yaw = last_pitch = last_roll = 0.0
smoothing_alpha = 0.3
tracking_active = False

# --- VTube Studio Functions ---
def send_model_parameter(name, value):
    """Send a model parameter update to VTube Studio"""
    global ws
    if ws and connected:
        data = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "param_update_" + name,
            "messageType": "InjectParameterDataRequest",
            "data": {
                "parameterValues": [{"id": name, "value": value}]
            }
        }
        ws.send(json.dumps(data))

def register_plugin():
    """Register plugin to VTube Studio"""
    global ws, connected
    payload = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "register_plugin",
        "messageType": "AuthenticationTokenRequest",
        "data": {
            "pluginName": PLUGIN_NAME,
            "pluginDeveloper": PLUGIN_DEVELOPER
        }
    }
    try:
        ws.send(json.dumps(payload))
        connected = True
        print("✅ Connected to VTube Studio")
    except Exception as e:
        print("❌ Failed to connect:", e)
        connected = False

def connect_vts():
    """Connect WebSocket to VTube Studio"""
    global ws
    def on_open(wsapp):
        register_plugin()

    def on_error(wsapp, error):
        print("WebSocket Error:", error)

    def on_close(wsapp, close_status_code, close_msg):
        global connected
        connected = False
        print("🔴 Disconnected from VTS")

    ws = websocket.WebSocketApp(
        VTS_URL,
        on_open=on_open,
        on_error=on_error,
        on_close=on_close
    )

    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    time.sleep(1)

# --- Calculate Head Rotation ---
def get_head_rotation(landmarks, w, h):
    """Calculate approximate head rotation (yaw, pitch, roll)"""
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose_tip = landmarks[1]
    chin = landmarks[152]

    left = (int(left_eye.x * w), int(left_eye.y * h))
    right = (int(right_eye.x * w), int(right_eye.y * h))
    nose = (int(nose_tip.x * w), int(nose_tip.y * h))
    chin_p = (int(chin.x * w), int(chin.y * h))

    dx = right[0] - left[0]
    dy = right[1] - left[1]
    yaw = math.degrees(math.atan2(dy, dx))

    face_h = chin_p[1] - nose[1]
    pitch = (nose[1] - chin_p[1]) / face_h * 40 if face_h != 0 else 0

    roll = (nose[0] - (left[0] + right[0]) / 2) / dx * 50 if dx != 0 else 0

    return yaw, pitch, roll

# --- Main Tracking Thread ---
def start_tracking():
    global tracking_active, ws, connected
    tracking_active = True
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Tidak bisa membuka kamera.")
        return

    label_tracking.config(text="🟢 Tracking Active: Face")

    while tracking_active:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

                yaw, pitch, roll = get_head_rotation(face_landmarks.landmark, w, h)
                yaw = max(-30, min(30, yaw))
                pitch = max(-30, min(30, pitch))
                roll = max(-30, min(30, roll))

                # Smoothing
                global last_yaw, last_pitch, last_roll
                last_yaw = last_yaw * (1 - smoothing_alpha) + yaw * smoothing_alpha
                last_pitch = last_pitch * (1 - smoothing_alpha) + pitch * smoothing_alpha
                last_roll = last_roll * (1 - smoothing_alpha) + roll * smoothing_alpha

                if connected:
                    send_model_parameter("ParamAngleX", last_yaw / 30)
                    send_model_parameter("ParamAngleY", last_pitch / 30)
                    send_model_parameter("ParamAngleZ", last_roll / 30)

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
        window.update_idletasks()

    cap.release()
    cv2.destroyAllWindows()
    label_tracking.config(text="🔴 Tracking Stopped")

def stop_tracking():
    global tracking_active
    tracking_active = False

# --- GUI Setup ---
window = tk.Tk()
window.title("VTS Face Tracker")
window.geometry("800x600")
window.configure(bg="#1e1e1e")

label_tracking = Label(window, text="🔴 Not Tracking", font=("Arial", 14), fg="white", bg="#1e1e1e")
label_tracking.pack(pady=10)

lbl_video = Label(window)
lbl_video.pack()

btn_start = tk.Button(window, text="Start Face Tracking", font=("Arial", 12), bg="#4CAF50", fg="white", command=lambda: threading.Thread(target=start_tracking).start())
btn_start.pack(pady=10)

btn_stop = tk.Button(window, text="Stop Tracking", font=("Arial", 12), bg="#f44336", fg="white", command=stop_tracking)
btn_stop.pack(pady=5)

# Connect to VTS
connect_vts()

window.mainloop()
