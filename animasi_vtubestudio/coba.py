import cv2
import mediapipe as mp
import json
import websocket
import threading
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import time

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
available_parameters = []  # Store available parameters
reconnect_lock = threading.Lock()
last_heartbeat = 0

# --- GUI setup ---
root = tk.Tk()
root.title("Body Tracking + VTube Studio")
root.geometry("900x750")

video_label = Label(root)
video_label.pack()

status_label = Label(root, text="Status: Belum Terhubung", font=("Arial", 12), fg="red")
status_label.pack(pady=10)

# Label untuk menampilkan nilai tracking
tracking_label = Label(root, text="Tracking: Waiting...", font=("Arial", 10), fg="blue")
tracking_label.pack(pady=5)

# Label untuk menampilkan parameter yang tersedia
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
                            available_parameters = params
                            print("\n[ℹ️] ========== DAFTAR PARAMETER MODEL ==========")
                            for i, p in enumerate(params, 1):
                                print(f"   {i}. {p}")
                            print("=" * 50)
                            print("\n⚠️  PENTING: Cek parameter di atas!")
                            print("   Ganti 'TARGET_PARAMETER' di baris ~210 dengan nama yang sesuai")
                            print("   Contoh parameter lengan yang mungkin ada:")
                            arm_params = [p for p in params if 'arm' in p.lower() or 'hand' in p.lower()]
                            if arm_params:
                                print("   Parameter yang berhubungan dengan arm/hand:")
                                for ap in arm_params:
                                    print(f"      - {ap}")
                            print("\n")
                            
                            # Update GUI dengan daftar parameter
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


def send_model_parameter(parameter_name, value, model_name=None):
    """Kirim SetModelParameter ke VTube Studio."""
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
            "messageType": "InjectParameterDataRequest",
            "data": {
                "parameterValues": [
                    {
                        "id": parameter_name,
                        "value": float(value)
                    }
                ]
            }
        }
        
        # Try to send, if fails try to reconnect
        try:
            ws.send(json.dumps(req))
            last_param_sent = float(value)
            last_send_time = now
            
            # Update tracking label
            tracking_label.config(text=f"Tracking: {parameter_name} = {value:.2f}")
            return True
        except Exception as send_err:
            print(f"[⚠️] Send failed: {send_err}. Attempting reconnect...")
            # Mark as disconnected and try to reconnect
            connected = False
            threading.Thread(target=reconnect_vtube, daemon=True).start()
            return False
            
    except Exception as e:
        print("[⚠️] Gagal mengirim parameter ke VTS:", e)
        return False


def reconnect_vtube():
    """Reconnect to VTube Studio if connection is lost."""
    global ws, connected, reconnect_lock
    
    with reconnect_lock:
        if connected:  # Already reconnected by another thread
            return
            
        print("[🔄] Reconnecting to VTube Studio...")
        status_label.config(text="Reconnecting...", fg="orange")
        
        try:
            # Close old connection if exists
            if ws:
                try:
                    ws.close()
                except:
                    pass
            
            # Wait a bit before reconnecting
            time.sleep(1)
            
            # Try to reconnect
            connect_vtube()
            
        except Exception as e:
            print(f"[❌] Reconnect failed: {e}")
            status_label.config(text="Disconnected - Retrying...", fg="red")
            # Try again after 3 seconds
            time.sleep(3)
            threading.Thread(target=reconnect_vtube, daemon=True).start()


def send_heartbeat():
    """Send periodic heartbeat to keep connection alive."""
    global ws, connected, last_heartbeat
    
    while True:
        time.sleep(5)  # Send heartbeat every 5 seconds
        
        if connected and ws:
            now = time.time()
            # Only send if we haven't sent anything in the last 5 seconds
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
                    # Don't wait for response, just send to keep alive
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

    # Try InputParameterListRequest first
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
        
        print("[DEBUG] InputParameterListRequest response:", json.dumps(resp, indent=2))
        
        # Parse response
        data = resp.get("data", {})
        
        # Try different possible fields
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
        
        # Fallback: Try ParameterValueRequest to get current values
        print("[INFO] Trying fallback method: ParameterValueRequest")
        req2 = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"get_paramvalues_{int(time.time()*1000)}",
            "messageType": "ParameterValueRequest",
            "data": {}
        }
        ws.send(json.dumps(req2))
        resp_raw = ws.recv()
        resp = json.loads(resp_raw)
        
        print("[DEBUG] ParameterValueRequest response:", json.dumps(resp, indent=2))
        
        data = resp.get("data", {})
        if "parameters" in data:
            params = data["parameters"]
            for p in params:
                if isinstance(p, dict):
                    name = p.get("name") or p.get("id")
                    if name:
                        param_names.append(name)
        
        ws.settimeout(None)
        return param_names if param_names else None
        
    except Exception as e:
        print(f"[⚠️] request_parameter_list error: {e}")
        import traceback
        traceback.print_exc()
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
    
    # ⚠️ GANTI INI dengan nama parameter yang sesuai dari daftar parameter model kamu!
    # Coba gunakan parameter yang umum ada di semua model dulu untuk testing:
    # - "ParamAngleX" (rotate head left/right)
    # - "ParamAngleY" (rotate head up/down)
    # - "ParamBodyAngleX" (tilt body left/right)
    # - "ParamBodyAngleY" (tilt body forward/back)
    TARGET_PARAMETER = "ParamBodyAngleX"  # Changed to common parameter for testing
    
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

        # --- compute wrist -> model parameter mapping ---
        # GUNAKAN TANGAN KANAN karena kamera di-flip (tangan kanan terlihat di kiri layar)
        try:
            hand_landmarks = None
            hand_side = ""
            
            # Coba deteksi tangan kanan dulu (yang terlihat di kiri layar karena mirror)
            if results.right_hand_landmarks:
                hand_landmarks = results.right_hand_landmarks
                hand_side = "RIGHT (appears left on screen)"
            elif results.left_hand_landmarks:
                hand_landmarks = results.left_hand_landmarks
                hand_side = "LEFT (appears right on screen)"
            
            if hand_landmarks:
                wrist = hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
                # MediaPipe landmarks are normalized (x,y in 0..1). Use y for vertical mapping.
                wrist_y = wrist.y
                # Invert so that raising hand (smaller y) -> larger value
                normalized = 1.0 - wrist_y
                # Clip just in case
                normalized = max(0.0, min(1.0, normalized))

                # ⚠️ UBAH RANGE INI sesuai dengan parameter model kamu!
                # Kebanyakan parameter VTube Studio pakai range -1.0 sampai 1.0
                out_min, out_max = -1.0, 1.0  # Changed from -10.0, 10.0
                mapped = normalized * (out_max - out_min) + out_min

                # Apply simple exponential smoothing to reduce jitter before sending
                if smoothed_param is None:
                    smoothed_param = mapped
                else:
                    smoothed_param = smoothed_param * (1 - smoothing_alpha) + mapped * smoothing_alpha

                # Kirim ke VTube
                success = send_model_parameter(TARGET_PARAMETER, smoothed_param)
                
                # Debug output (comment out jika terlalu banyak spam)
                if success and int(time.time() * 2) % 2 == 0:  # Print every ~0.5s
                    print(f"[📊] Sent: {TARGET_PARAMETER} = {smoothed_param:.3f} (hand: {hand_side})")
            else:
                # Tidak ada tangan terdeteksi
                if int(time.time() * 2) % 2 == 0:
                    tracking_label.config(text="Tracking: No hand detected - Show your hand!")
                    
        except Exception as e:
            print(f"[⚠️] Error in tracking: {e}")

        time.sleep(0.01)

    cap.release()


# === Run threads ===
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=connect_vtube, daemon=True).start()
threading.Thread(target=send_heartbeat, daemon=True).start()  # Keep connection alive

root.mainloop()