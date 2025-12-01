import cv2
import mediapipe as mp
import json
import websocket
import threading
import math

# --- VTS Config ---
VTS_URL = "ws://localhost:8001"
PLUGIN_NAME = "Python Body Tracker"
PLUGIN_DEVELOPER = "Natania"

# FaceMesh Init
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Pose Init (for arms)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Default VTS parameters
PARAM_MAPPING = {
    'head_x': 'FaceAngleX',
    'head_y': 'FaceAngleY',
    'head_z': 'FaceAngleZ',
    'mouth_open': 'MouthOpen',
    'eye_left': 'EyeOpenLeft',
    'eye_right': 'EyeOpenRight',
    'arm_left': 'ArmLeft',
    'arm_right': 'ArmRight'
}

ws = None


def send_to_vts(data):
    try:
        ws.send(json.dumps(data))
    except:
        pass


def send_params(params_dict):
    param_values = []
    for key, value in params_dict.items():
        param_values.append({
            "name": key,
            "value": float(value)
        })

    message = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "injectParams",
        "messageType": "InjectParameterDataRequest",
        "data": {
            "parameterValues": param_values,
            "faceFound": True
        }
    }
    send_to_vts(message)


def calculate_arm_updown(pose_landmarks):
    if not pose_landmarks:
        return 0.0, 0.0

    l_sh = pose_landmarks.landmark[11]
    l_wr = pose_landmarks.landmark[15]
    r_sh = pose_landmarks.landmark[12]
    r_wr = pose_landmarks.landmark[16]

    left = (l_sh.y - l_wr.y) * 3
    right = (r_sh.y - r_wr.y) * 3

    return max(-1, min(1, left)), max(-1, min(1, right))


def camera_loop():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res_face = face_mesh.process(rgb)
        res_pose = pose.process(rgb)

        params = {}

        if res_face.multi_face_landmarks:
            face = res_face.multi_face_landmarks[0]

            nose = face.landmark[1]
            left_eye = face.landmark[33]
            right_eye = face.landmark[263]

            params[PARAM_MAPPING['head_x']] = (right_eye.x - left_eye.x) * 30
            params[PARAM_MAPPING['head_y']] = (nose.y - 0.5) * -50
            params[PARAM_MAPPING['head_z']] = 0

            mouth_up = face.landmark[13]
            mouth_low = face.landmark[14]
            params[PARAM_MAPPING['mouth_open']] = max(0, min(1, (mouth_low.y - mouth_up.y) * 15))

            le_u = face.landmark[159]
            le_l = face.landmark[145]
            re_u = face.landmark[386]
            re_l = face.landmark[374]

            params[PARAM_MAPPING['eye_left']] = max(0, min(1, (le_l.y - le_u.y) * 20))
            params[PARAM_MAPPING['eye_right']] = max(0, min(1, (re_l.y - re_u.y) * 20))

        if res_pose.pose_landmarks:
            l_arm, r_arm = calculate_arm_updown(res_pose.pose_landmarks)
            params[PARAM_MAPPING['arm_left']] = l_arm
            params[PARAM_MAPPING['arm_right']] = r_arm

        if params:
            send_params(params)

        cv2.imshow("Camera Tracking", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def connect_vts():
    global ws
    ws = websocket.WebSocket()
    ws.connect(VTS_URL)
    print("✔ Connected to VTS!")

    register_msg = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "register",
        "messageType": "PluginRegistrationRequest",
        "data": {
            "pluginName": PLUGIN_NAME,
            "pluginDeveloper": PLUGIN_DEVELOPER
        }
    }
    send_to_vts(register_msg)


if __name__ == "__main__":
    connect_vts()
    camera_loop()
