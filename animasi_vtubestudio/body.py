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
PLUGIN_NAME = "Python Full Body Tracker"
PLUGIN_DEVELOPER = "Natania"

# --- Initialize MediaPipe Holistic (Face + Pose + Hands) ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
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

# Smoothing for all tracking
smoothed_params = {
    # Face
    'head_x': None,
    'head_y': None,
    'head_z': None,
    'mouth_open': None,
    'eye_left': None,
    'eye_right': None,
    # Body
    'body_x': None,
    'body_y': None,
    # Arms
    'arm_left_x': None,
    'arm_left_y': None,
    'arm_right_x': None,
    'arm_right_y': None,
    # Hands
    'hand_left_x': None,
    'hand_left_y': None,
    'hand_right_x': None,
    'hand_right_y': None,
}

# --- GUI setup ---
root = tk.Tk()
root.title("Full Body Tracking + VTube Studio")
root.geometry("900x800")

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
                            
                            # Cari parameter yang berhubungan dengan body/arm/hand
                            body_params = [p for p in params if any(keyword in p.lower() for keyword in ['arm', 'hand', 'body', 'shoulder', 'elbow', 'wrist'])]
                            if body_params:
                                print("\n[ℹ️] Parameter body/arm/hand yang tersedia:")
                                for bp in body_params:
                                    print(f"   - {bp}")
                            print()
                            
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


def send_tracking_data(params_dict):
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
            "requestID": f"tracking_data_{int(now*1000)}",
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
        print("[⚠️] Gagal mengirim tracking data ke VTS:", e)
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
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    
    nose_x = nose_tip.x * image_width
    nose_y = nose_tip.y * image_height
    
    # Head rotation X (left/right turn)
    center_x = image_width / 2
    head_x = (nose_x - center_x) / (image_width / 2)
    head_x = max(-1.0, min(1.0, head_x))
    
    # Head rotation Y (up/down tilt)
    center_y = image_height / 2
    head_y = (nose_y - center_y) / (image_height / 2)
    head_y = max(-1.0, min(1.0, head_y))
    
    # Head rotation Z (tilt left/right)
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


# === BODY TRACKING FUNCTIONS ===
def calculate_body_position(pose_landmarks):
    """Calculate body tilt from pose landmarks."""
    # Get shoulder landmarks
    left_shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
    
    # Calculate body tilt X (lean left/right)
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    body_x = (shoulder_center_x - 0.5) * 2  # Normalize to -1 to 1
    body_x = max(-1.0, min(1.0, body_x))
    
    # Calculate body tilt Y (lean forward/back)
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
    body_y = (shoulder_center_y - 0.5) * 2
    body_y = max(-1.0, min(1.0, body_y))
    
    return body_x, body_y


def calculate_arm_position(pose_landmarks, side='left'):
    """Calculate arm position (shoulder to wrist)."""
    if side == 'left':
        shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        elbow = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW]
        wrist = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
    else:
        shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        elbow = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
        wrist = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
    
    # Arm X position (horizontal)
    arm_x = (wrist.x - shoulder.x) * 2  # Relative to shoulder
    arm_x = max(-1.0, min(1.0, arm_x))
    
    # Arm Y position (vertical) - inverted so up is positive
    arm_y = (shoulder.y - wrist.y) * 2  # Inverted
    arm_y = max(-1.0, min(1.0, arm_y))
    
    return arm_x, arm_y


def calculate_hand_position(hand_landmarks):
    """Calculate hand position from hand landmarks."""
    if not hand_landmarks:
        return None, None
    
    # Get wrist position
    wrist = hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
    
    # Hand X position
    hand_x = (wrist.x - 0.5) * 2
    hand_x = max(-1.0, min(1.0, hand_x))
    
    # Hand Y position (inverted)
    hand_y = (0.5 - wrist.y) * 2
    hand_y = max(-1.0, min(1.0, hand_y))
    
    return hand_x, hand_y


def smooth_value(current, new_value, alpha=0.3):
    """Apply exponential smoothing."""
    if current is None:
        return new_value
    return current * (1 - alpha) + new_value * alpha


# === Function for camera & MediaPipe frame ===
def camera_loop():
    cap = cv2.VideoCapture(0)
    
    # Parameter mapping - GANTI INI sesuai model kamu!
    PARAM_MAPPING = {
        # Face parameters (standard VTS)
        'head_x': 'FaceAngleX',
        'head_y': 'FaceAngleY',
        'head_z': 'FaceAngleZ',
        'mouth_open': 'MouthOpen',
        'eye_left': 'EyeOpenLeft',
        'eye_right': 'EyeOpenRight',
        
        # Body parameters (sesuaikan dengan nama di model kamu!)
        'body_x': 'ParamBodyAngleX',
        'body_y': 'ParamBodyAngleY',
        
        # Arm parameters (GANTI dengan nama yang benar dari daftar parameter!)
        'arm_left_x': 'ParamArmLX',      # Horizontal position left arm
        'arm_left_y': 'ParamArmLY',      # Vertical position left arm
        'arm_right_x': 'ParamArmRX',     # Horizontal position right arm
        'arm_right_y': 'ParamArmRY',     # Vertical position right arm
        
        # Hand parameters (opsional, tergantung model)
        'hand_left_x': 'ParamHandLX',
        'hand_left_y': 'ParamHandLY',
        'hand_right_x': 'ParamHandRX',
        'hand_right_y': 'ParamHandRY',
    }
    
    print("\n[ℹ️] Full body tracking dimulai!")
    print("=" * 60)
    print("CATATAN: Ganti nama parameter di PARAM_MAPPING (baris ~370)")
    print("dengan nama yang sesuai dari daftar parameter model kamu!")
    print("=" * 60)
    print("\nMapping parameters:")
    for k, v in PARAM_MAPPING.items():
        print(f"  {k:15s} → {v}")
    print()
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # Draw all landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )
        
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )

        # Convert to ImageTk for Tkinter display
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        root.update_idletasks()
        root.update()

        # Process tracking
        try:
            vts_params = {}
            
            # === FACE TRACKING ===
            if results.face_landmarks:
                h, w, _ = frame.shape
                head_x, head_y, head_z = calculate_face_rotation(results.face_landmarks, w, h)
                mouth_open = calculate_mouth_open(results.face_landmarks)
                eye_left, eye_right = calculate_eye_openness(results.face_landmarks)
                
                # Smooth face values
                smoothed_params['head_x'] = smooth_value(smoothed_params['head_x'], head_x)
                smoothed_params['head_y'] = smooth_value(smoothed_params['head_y'], head_y)
                smoothed_params['head_z'] = smooth_value(smoothed_params['head_z'], head_z)
                smoothed_params['mouth_open'] = smooth_value(smoothed_params['mouth_open'], mouth_open)
                smoothed_params['eye_left'] = smooth_value(smoothed_params['eye_left'], eye_left)
                smoothed_params['eye_right'] = smooth_value(smoothed_params['eye_right'], eye_right)
                
                # Add to VTS params
                vts_params[PARAM_MAPPING['head_x']] = smoothed_params['head_x'] * 30
                vts_params[PARAM_MAPPING['head_y']] = smoothed_params['head_y'] * 30
                vts_params[PARAM_MAPPING['head_z']] = smoothed_params['head_z'] * 30
                vts_params[PARAM_MAPPING['mouth_open']] = smoothed_params['mouth_open']
                vts_params[PARAM_MAPPING['eye_left']] = smoothed_params['eye_left']
                vts_params[PARAM_MAPPING['eye_right']] = smoothed_params['eye_right']
            
            # === BODY TRACKING ===
            if results.pose_landmarks:
                body_x, body_y = calculate_body_position(results.pose_landmarks)
                
                # Smooth body values
                smoothed_params['body_x'] = smooth_value(smoothed_params['body_x'], body_x)
                smoothed_params['body_y'] = smooth_value(smoothed_params['body_y'], body_y)
                
                # Add to VTS params
                vts_params[PARAM_MAPPING['body_x']] = smoothed_params['body_x'] * 30
                vts_params[PARAM_MAPPING['body_y']] = smoothed_params['body_y'] * 30
                
                # === ARM TRACKING ===
                # Left arm
                arm_left_x, arm_left_y = calculate_arm_position(results.pose_landmarks, 'left')
                smoothed_params['arm_left_x'] = smooth_value(smoothed_params['arm_left_x'], arm_left_x)
                smoothed_params['arm_left_y'] = smooth_value(smoothed_params['arm_left_y'], arm_left_y)
                vts_params[PARAM_MAPPING['arm_left_x']] = smoothed_params['arm_left_x'] * 10
                vts_params[PARAM_MAPPING['arm_left_y']] = smoothed_params['arm_left_y'] * 10
                
                # Right arm
                arm_right_x, arm_right_y = calculate_arm_position(results.pose_landmarks, 'right')
                smoothed_params['arm_right_x'] = smooth_value(smoothed_params['arm_right_x'], arm_right_x)
                smoothed_params['arm_right_y'] = smooth_value(smoothed_params['arm_right_y'], arm_right_y)
                vts_params[PARAM_MAPPING['arm_right_x']] = smoothed_params['arm_right_x'] * 10
                vts_params[PARAM_MAPPING['arm_right_y']] = smoothed_params['arm_right_y'] * 10
            
            # === HAND TRACKING ===
            # Left hand
            if results.left_hand_landmarks:
                hand_left_x, hand_left_y = calculate_hand_position(results.left_hand_landmarks)
                smoothed_params['hand_left_x'] = smooth_value(smoothed_params['hand_left_x'], hand_left_x)
                smoothed_params['hand_left_y'] = smooth_value(smoothed_params['hand_left_y'], hand_left_y)
                vts_params[PARAM_MAPPING['hand_left_x']] = smoothed_params['hand_left_x'] * 10
                vts_params[PARAM_MAPPING['hand_left_y']] = smoothed_params['hand_left_y'] * 10
            
            # Right hand
            if results.right_hand_landmarks:
                hand_right_x, hand_right_y = calculate_hand_position(results.right_hand_landmarks)
                smoothed_params['hand_right_x'] = smooth_value(smoothed_params['hand_right_x'], hand_right_x)
                smoothed_params['hand_right_y'] = smooth_value(smoothed_params['hand_right_y'], hand_right_y)
                vts_params[PARAM_MAPPING['hand_right_x']] = smoothed_params['hand_right_x'] * 10
                vts_params[PARAM_MAPPING['hand_right_y']] = smoothed_params['hand_right_y'] * 10
            
            # Send all parameters to VTube Studio
            if vts_params:
                success = send_tracking_data(vts_params)
                
                # Debug output every 60 frames (~2 seconds)
                frame_count += 1
                if success and frame_count % 60 == 0:
                    print(f"[📊] Full body tracking active:")
                    if results.face_landmarks:
                        print(f"     Face: Head({smoothed_params['head_x']:.2f},{smoothed_params['head_y']:.2f}) Mouth({smoothed_params['mouth_open']:.2f})")
                    if results.pose_landmarks:
                        print(f"     Body: ({smoothed_params['body_x']:.2f},{smoothed_params['body_y']:.2f})")
                        print(f"     Arms: L({smoothed_params['arm_left_y']:.2f}) R({smoothed_params['arm_right_y']:.2f})")
                    if results.left_hand_landmarks or results.right_hand_landmarks:
                        print(f"     Hands detected: L={results.left_hand_landmarks is not None}, R={results.right_hand_landmarks is not None}")
                    print()
            else:
                # No tracking data
                if frame_count % 60 == 0:
                    tracking_label.config(text="Tracking: No body detected - Show full body!")
                    
        except Exception as e:
            print(f"[⚠️] Error in tracking: {e}")
            import traceback
            traceback.print_exc()

        time.sleep(0.01)

    cap.release()


# === Run threads ===
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=connect_vtube, daemon=True).start()
threading.Thread(target=send_heartbeat, daemon=True).start()

root.mainloop()