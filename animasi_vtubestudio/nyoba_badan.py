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
PLUGIN_NAME = "Python Face Tracker"
PLUGIN_DEVELOPER = "Natania"

# --- Initialize MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Pose (Body)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Global WebSocket variable ---
ws = None
auth_token = None
connected = False
last_send_time = 0
available_parameters = []
reconnect_lock = threading.Lock()
last_heartbeat = 0

# Smoothing for face tracking
smoothed_params = {
    'head_x': None,
    'head_y': None,
    'head_z': None,
    'mouth_open': None,
    'eye_left': None,
    'eye_right': None,
    'hand_left_x': None,
    'hand_right_x': None
}

# --- GUI setup ---
root = tk.Tk()
root.title("Face Tracking + VTube Studio")
root.geometry("900x750")

video_label = Label(root)
video_label.pack()

status_label = Label(root, text="Status: Belum Terhubung", font=("Arial", 12), fg="red")
status_label.pack(pady=10)

tracking_label = Label(root, text="Tracking: Waiting...", font=("Arial", 10), fg="blue")
tracking_label.pack(pady=5)

param_label = Label(root, text="Parameters: Loading...", font=("Arial", 9), fg="gray", wraplength=850, justify="left")
param_label.pack(pady=5)


# === Function to connect to VTube Studio ===
def connect_vtube():
    global ws, available_parameters
    try:
        ws = websocket.WebSocket()
        ws.connect(VTS_URL)
        status_label.config(text="Terhubung ke VTube Studio!", fg="green")
        print("[✅] Terhubung ke VTube Studio!")

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
                    global connected
                    connected = True
                    
                    try:
                        params = request_parameter_list()
                        if params:
                            available_parameters = params
                            print("\n[ℹ️] ========== DAFTAR PARAMETER MODEL ==========")
                            for i, p in enumerate(params, 1):
                                print(f"   {i}. {p}")
                            print("=" * 50)
                            
                            # Update GUI
                            param_text = "Available Parameters:\n" + ", ".join(params[:15])
                            if len(params) > 15:
                                param_text += f"\n... and {len(params) - 15} more (check console)"
                            param_label.config(text=param_text)
                        else:
                            print("[⚠️] Tidak menerima daftar parameter dari VTS.")
                            param_label.config(text="Parameters: Not available")
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


def send_face_data(params_dict):
    """Kirim multiple parameters sekaligus ke VTube Studio."""
    global ws, connected, last_send_time
    try:
        if not connected or ws is None:
            return False

        now = time.time()
        
        # Build parameter values array
        param_values = []
        for param_name, value in params_dict.items():
            param_values.append({
                "id": param_name,
                "value": float(value)
            })

        req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"face_data_{int(now*1000)}",
            "messageType": "InjectParameterDataRequest",
            "data": {
                "parameterValues": param_values
            }
        }
        
        try:
            ws.send(json.dumps(req))
            last_send_time = now
            
            # Update tracking label
            summary = ", ".join([f"{k}: {v:.2f}" for k, v in list(params_dict.items())[:3]])
            tracking_label.config(text=f"Tracking: {summary}...")
            return True
        except Exception as send_err:
            print(f"[⚠️] Send failed: {send_err}. Attempting reconnect...")
            connected = False
            threading.Thread(target=reconnect_vtube, daemon=True).start()
            return False
            
    except Exception as e:
        print("[⚠️] Gagal mengirim face data ke VTS:", e)
        return False


def reconnect_vtube():
    """Reconnect to VTube Studio if connection is lost."""
    global ws, connected, reconnect_lock
    
    with reconnect_lock:
        if connected:
            return
            
        print("[🔄] Reconnecting to VTube Studio...")
        status_label.config(text="Reconnecting...", fg="orange")
        
        try:
            if ws:
                try:
                    ws.close()
                except:
                    pass
            
            time.sleep(1)
            connect_vtube()
            
        except Exception as e:
            print(f"[❌] Reconnect failed: {e}")
            status_label.config(text="Disconnected - Retrying...", fg="red")
            time.sleep(3)
            threading.Thread(target=reconnect_vtube, daemon=True).start()


def send_heartbeat():
    """Send periodic heartbeat to keep connection alive."""
    global ws, connected, last_heartbeat
    
    while True:
        time.sleep(5)
        
        if connected and ws:
            now = time.time()
            if now - last_send_time > 5:
                try:
                    req = {
                        "apiName": "VTubeStudioPublicAPI",
                        "apiVersion": "1.0",
                        "requestID": f"heartbeat_{int(now*1000)}",
                        "messageType": "APIStateRequest",
                        "data": {}
                    }
                    ws.send(json.dumps(req))
                    last_heartbeat = now
                except Exception as e:
                    print(f"[⚠️] Heartbeat failed: {e}")
                    connected = False
                    threading.Thread(target=reconnect_vtube, daemon=True).start()


def request_parameter_list(timeout=3.0):
    """Minta daftar parameter model dari VTube Studio."""
    global ws, connected
    if not connected or ws is None:
        return None

    req = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": f"get_params_{int(time.time()*1000)}",
        "messageType": "InputParameterListRequest",
        "data": {}
    }
    try:
        ws.send(json.dumps(req))
        ws.settimeout(timeout)
        resp_raw = ws.recv()
        resp = json.loads(resp_raw)
        
        data = resp.get("data", {})
        param_names = []
        
        for key in ["customParameters", "defaultParameters", "parameters", "modelParameters"]:
            if key in data and isinstance(data[key], list):
                for p in data[key]:
                    if isinstance(p, dict):
                        name = p.get("name") or p.get("id") or p.get("parameter")
                        if name:
                            param_names.append(name)
                    elif isinstance(p, str):
                        param_names.append(p)
                if param_names:
                    ws.settimeout(None)
                    return param_names
        
        ws.settimeout(None)
        return param_names if param_names else None
        
    except Exception as e:
        print(f"[⚠️] request_parameter_list error: {e}")
        try:
            ws.settimeout(None)
        except Exception:
            pass
        return None


# === FACE TRACKING FUNCTIONS ===
def calculate_face_rotation(face_landmarks, image_width, image_height):
    """Calculate face rotation angles from landmarks."""
    nose_tip = face_landmarks.landmark[1]
    nose_bridge = face_landmarks.landmark[168]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    
    nose_x = nose_tip.x * image_width
    nose_y = nose_tip.y * image_height
    bridge_x = nose_bridge.x * image_width
    bridge_y = nose_bridge.y * image_height
    
    center_x = image_width / 2
    head_x = (nose_x - center_x) / (image_width / 2)
    head_x = max(-1.0, min(1.0, head_x))
    
    center_y = image_height / 2
    head_y = (nose_y - center_y) / (image_height / 2)
    head_y = max(-1.0, min(1.0, head_y))
    
    eye_dx = right_eye.x - left_eye.x
    eye_dy = right_eye.y - left_eye.y
    head_z = math.atan2(eye_dy, eye_dx)
    head_z = head_z / (math.pi / 4)
    head_z = max(-1.0, min(1.0, head_z))
    
    return head_x, head_y, head_z


def calculate_mouth_open(face_landmarks):
    """Calculate mouth openness."""
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]
    
    mouth_open = abs(upper_lip.y - lower_lip.y) * 100
    mouth_open = max(0.0, min(1.0, mouth_open * 5))
    
    return mouth_open


def calculate_eye_openness(face_landmarks):
    """Calculate eye openness for both eyes."""
    left_eye_top = face_landmarks.landmark[159]
    left_eye_bottom = face_landmarks.landmark[145]
    left_eye_open = abs(left_eye_top.y - left_eye_bottom.y) * 100
    left_eye_open = max(0.0, min(1.0, left_eye_open * 10))
    
    right_eye_top = face_landmarks.landmark[386]
    right_eye_bottom = face_landmarks.landmark[374]
    right_eye_open = abs(right_eye_top.y - right_eye_bottom.y) * 100
    right_eye_open = max(0.0, min(1.0, right_eye_open * 10))
    
    return left_eye_open, right_eye_open


def smooth_value(current, new_value, alpha=0.3):
    """Apply exponential smoothing."""
    if current is None:
        return new_value
    return current * (1 - alpha) + new_value * alpha


def calculate_hand_positions(multi_hand_landmarks, multi_handedness):
    """Calculate hand X positions from hand landmarks."""
    left_x = None
    right_x = None
    
    for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
        wrist = hand_landmarks.landmark[0]  # wrist
        hand_x = (wrist.x - 0.5) * 2.0  # Normalize to -1 to 1
        
        if handedness.classification[0].label == "Left":
            left_x = hand_x
        else:
            right_x = hand_x
    
    return left_x, right_x


# === Function for camera & MediaPipe frame ===
def camera_loop():
    cap = cv2.VideoCapture(0)
    
    # VTube Studio parameter names
    PARAM_MAPPING = {
        'head_x': 'FaceAngleX',        # Head turn left/right
        'head_y': 'FaceAngleY',        # Head tilt up/down
        'head_z': 'FaceAngleZ',        # Head tilt left/right
        'mouth_open': 'MouthOpen',     # Mouth openness
        'eye_left': 'EyeOpenLeft',     # Left eye open
        'eye_right': 'EyeOpenRight',   # Right eye open
        'hand_left_x': 'PARAM_ARM_L_A',   # Left hand position
        'hand_right_x': 'PARAM_ARM_R_B',  # Right hand position
    }
    
    print("\n[ℹ️] Face tracking dimulai!")
    print("Mapping parameters:")
    for k, v in PARAM_MAPPING.items():
        print(f"  {k} → {v}")
    print()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_results = face_mesh.process(rgb)
        hand_results = hands.process(rgb)

        # Draw face mesh
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
        
        # Draw hands
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )

        # Show video feed
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        root.update_idletasks()
        root.update()

        # Process tracking
        try:
            h, w, _ = frame.shape
            vts_params = {}
            tracking_parts = []
            
            # FACE TRACKING
            if face_results.multi_face_landmarks and len(face_results.multi_face_landmarks) > 0:
                face_landmarks = face_results.multi_face_landmarks[0]
                
                # Calculate all face parameters
                head_x, head_y, head_z = calculate_face_rotation(face_landmarks, w, h)
                mouth_open = calculate_mouth_open(face_landmarks)
                eye_left, eye_right = calculate_eye_openness(face_landmarks)
                
                # Apply smoothing
                smoothed_params['head_x'] = smooth_value(smoothed_params['head_x'], head_x)
                smoothed_params['head_y'] = smooth_value(smoothed_params['head_y'], head_y)
                smoothed_params['head_z'] = smooth_value(smoothed_params['head_z'], head_z)
                smoothed_params['mouth_open'] = smooth_value(smoothed_params['mouth_open'], mouth_open)
                smoothed_params['eye_left'] = smooth_value(smoothed_params['eye_left'], eye_left)
                smoothed_params['eye_right'] = smooth_value(smoothed_params['eye_right'], eye_right)
                
                # Scale values to VTube Studio range
                vts_params[PARAM_MAPPING['head_x']] = smoothed_params['head_x'] * 30
                vts_params[PARAM_MAPPING['head_y']] = smoothed_params['head_y'] * 30
                vts_params[PARAM_MAPPING['head_z']] = smoothed_params['head_z'] * 30
                vts_params[PARAM_MAPPING['mouth_open']] = smoothed_params['mouth_open']
                vts_params[PARAM_MAPPING['eye_left']] = smoothed_params['eye_left']
                vts_params[PARAM_MAPPING['eye_right']] = smoothed_params['eye_right']
                
                tracking_parts.append("Face ✓")
            
            # HAND TRACKING
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                hand_left_x, hand_right_x = calculate_hand_positions(
                    hand_results.multi_hand_landmarks,
                    hand_results.multi_handedness
                )
                
                if hand_left_x is not None:
                    smoothed_params['hand_left_x'] = smooth_value(smoothed_params['hand_left_x'], hand_left_x)
                    vts_params[PARAM_MAPPING['hand_left_x']] = smoothed_params['hand_left_x'] * 10  # Scale -10 to 10
                    tracking_parts.append("Left Hand ✓")
                
                if hand_right_x is not None:
                    smoothed_params['hand_right_x'] = smooth_value(smoothed_params['hand_right_x'], hand_right_x)
                    vts_params[PARAM_MAPPING['hand_right_x']] = smoothed_params['hand_right_x'] * 10  # Scale -10 to 10
                    tracking_parts.append("Right Hand ✓")
            
            # Send to VTube Studio
            if vts_params:
                success = send_face_data(vts_params)
                tracking_label.config(text="Tracking: " + " | ".join(tracking_parts))
                
                # Debug output every 30 frames (~1 second)
                frame_count += 1
                if success and frame_count % 30 == 0:
                    print(f"[📊] Tracking active: {', '.join(tracking_parts)}")
                    if "Face ✓" in tracking_parts:
                        print(f"     Head X/Y/Z: {smoothed_params['head_x']:.2f}, {smoothed_params['head_y']:.2f}, {smoothed_params['head_z']:.2f}")
                        print(f"     Mouth: {smoothed_params['mouth_open']:.2f}, Eyes: {smoothed_params['eye_left']:.2f}/{smoothed_params['eye_right']:.2f}")
                    if "Left Hand ✓" in tracking_parts:
                        print(f"     Left Hand X: {smoothed_params['hand_left_x']:.2f}")
                    if "Right Hand ✓" in tracking_parts:
                        print(f"     Right Hand X: {smoothed_params['hand_right_x']:.2f}")
                    print()
            else:
                if frame_count % 30 == 0:
                    tracking_label.config(text="Tracking: Nothing detected")
                    
        except Exception as e:
            print(f"[⚠️] Error in tracking: {e}")

        time.sleep(0.01)

    cap.release()


# === Run threads ===
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=connect_vtube, daemon=True).start()
threading.Thread(target=send_heartbeat, daemon=True).start()

root.mainloop()