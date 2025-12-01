import asyncio
import websockets
import json
import cv2
import mediapipe as mp
import math
import os
import threading
import tkinter as tk
from PIL import Image, ImageTk
import time

# ========================
#  CONFIG
# ========================
VTS_URI = "ws://localhost:8001"
PLUGIN_NAME = "MediaPipeBridge"
DEVELOPER = "Christ"
TOKEN_FILE = "vts_token.json"

# ========================
#  TOKEN HANDLER
# ========================
async def get_auth_token(ws):
    print("🔹 Requesting new VTS token...")
    await ws.send(json.dumps({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "getToken",
        "messageType": "AuthenticationTokenRequest",
        "data": {
            "pluginName": PLUGIN_NAME,
            "pluginDeveloper": DEVELOPER
        }
    }))
    response = json.loads(await ws.recv())
    print("🔹 Token response:", response)

    # Handle case: VTS belum di-allow
    if not response.get("data") or "authenticationToken" not in response["data"]:
        raise Exception("No token received. Please click 'Allow' in VTube Studio.")

    token = response["data"]["authenticationToken"]
    with open(TOKEN_FILE, "w") as f:
        json.dump({"token": token}, f)
    print("✅ Token saved to vts_token.json")
    return token


# ========================
#  AUTHENTICATION
# ========================
async def authenticate(ws, label_status):
    if not os.path.exists(TOKEN_FILE):
        label_status.config(text="🟡 Requesting permission... (Check VTube Studio)", fg="orange")
        token = await get_auth_token(ws)
        await asyncio.sleep(2)  # beri waktu user klik allow
    else:
        with open(TOKEN_FILE, "r") as f:
            token = json.load(f)["token"]

    # Kirim request autentikasi
    await ws.send(json.dumps({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "auth",
        "messageType": "AuthenticationRequest",
        "data": {
            "pluginName": PLUGIN_NAME,
            "pluginDeveloper": DEVELOPER,
            "authenticationToken": token
        }
    }))
    response = json.loads(await ws.recv())
    print("Auth response:", response)

    # Jika token tidak valid, hapus dan ulangi
    if not response["data"].get("authenticated", False):
        label_status.config(text="🔴 Invalid token. Requesting new one...", fg="red")
        if os.path.exists(TOKEN_FILE):
            os.remove(TOKEN_FILE)
        await asyncio.sleep(1)
        return await authenticate(ws, label_status)

    label_status.config(text="✅ Connected & Authenticated", fg="green")
    return response


# ========================
#  TKINTER GUI
# ========================
class VTSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MediaPipe → VTube Studio")
        self.root.geometry("900x700")

        self.label_video = tk.Label(root)
        self.label_video.pack()

        self.label_status = tk.Label(root, text="Connecting...", fg="orange", font=("Segoe UI", 12))
        self.label_status.pack(pady=5)

        self.running = True

        # Setup mediapipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.cap = cv2.VideoCapture(0)

        # Start async loop in thread
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.start_async_loop, daemon=True).start()

        # Start tkinter video update
        self.update_video()

    def start_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.vtube_stream())

    async def vtube_stream(self):
        try:
            async with websockets.connect(VTS_URI) as ws:
                await authenticate(ws, self.label_status)

                while self.running:
                    success, image = self.cap.read()
                    if not success:
                        continue

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(image_rgb)

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            nose = face_landmarks.landmark[1]
                            left_eye = face_landmarks.landmark[33]
                            right_eye = face_landmarks.landmark[263]

                            dx = right_eye.x - left_eye.x
                            dy = right_eye.y - left_eye.y
                            yaw = math.degrees(math.atan2(dy, dx))
                            pitch = (nose.y - (left_eye.y + right_eye.y) / 2) * 100
                            roll = (nose.x - (left_eye.x + right_eye.x) / 2) * 100

                            payload = {
                                "apiName": "VTubeStudioPublicAPI",
                                "apiVersion": "1.0",
                                "requestID": "tracking",
                                "messageType": "InjectParameterDataRequest",
                                "data": {
                                    "parameterValues": [
                                        {"id": "FaceAngleX", "value": yaw},
                                        {"id": "FaceAngleY", "value": pitch},
                                        {"id": "FaceAngleZ", "value": roll}
                                    ]
                                }
                            }

                            try:
                                await ws.send(json.dumps(payload))
                            except Exception as e:
                                self.label_status.config(text=f"⚠️ Lost connection: {e}", fg="red")
                                break
                    await asyncio.sleep(0.02)
        except Exception as e:
            self.label_status.config(text=f"❌ Connection error: {e}", fg="red")

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label_video.imgtk = imgtk
            self.label_video.configure(image=imgtk)
        if self.running:
            self.root.after(10, self.update_video)

    def on_close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()


# ========================
#  RUN APP
# ========================
if __name__ == "__main__":
    root = tk.Tk()
    app = VTSApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
