import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import math
import os

# Warna BGR untuk Rambut Coklat
HAIR_COLOR_MAIN = (50, 100, 150)      # Coklat
HAIR_COLOR_SHADOW = (30, 60, 90)      # Coklat gelap
HAIR_COLOR_HIGHLIGHT = (80, 140, 200) # Coklat terang untuk highlight

# Asset paths (relative to this script)
TOP_ASSET_PATH = "baju2.png"
SKIRT_ASSET_PATH = "rok.png"
ARM_LEFT_ASSET_PATH = "lengan_kanan.png"
ARM_RIGHT_ASSET_PATH = "lengan_kiri.png"
BACKGROUND_IMAGE_PATH = "background.png"  # Gambar background (opsional)

# Skin colors (BGR) - sample from user's peach/cream image
# Main: RGB ~ (255, 218, 165) -> BGR (165, 218, 255)
SKIN_COLOR_MAIN = (165, 218, 255)
# Shadow: slightly darker
SKIN_COLOR_SHADOW = (145, 195, 230)
# Outline / edge color
SKIN_COLOR_OUTLINE = (120, 160, 200)

SKIRT_SCALE = 1.25       # (optional) scale factor for skirt size
SKIRT_VERTICAL_OFFSET = 0.30  # fraction of skirt height to raise the skirt (0.0 = top at hip_center)
TOP_NECK_OFFSET_RATIO = 0.10  # fraction of shoulder width to place top start just below neck
TOP_SCALE_X = 1.60
TOP_SCALE_Y = 1.10       # scale the shirt vertically (1.0 = no change)
TOP_VERTICAL_OFFSET_PIX = 22  # raise the shirt upward in pixels (positive = move up)

class PoseHandFaceTracker:
    def __init__(self):
        # Inisialisasi MediaPipe Solutions
        # ==== POSE ====
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ==== HANDS ====
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ==== FACE MESH ====
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Try to load clothing assets once (with alpha channel if present)
        self.top_asset = None
        self.skirt_asset = None
        self.arm_left_asset = None
        self.arm_right_asset = None
        try:
            script_dir = os.path.dirname(__file__)
        except Exception:
            script_dir = os.getcwd()

        # Load top asset from a few candidate locations
        top_candidates = [
            TOP_ASSET_PATH,
            os.path.join(script_dir, TOP_ASSET_PATH),
            os.path.join(script_dir, "assets", os.path.basename(TOP_ASSET_PATH)),
            os.path.join(script_dir, os.path.basename(TOP_ASSET_PATH))  # e.g., coba.py folder contains atas1.png directly
        ]
        for p in top_candidates:
            try:
                if os.path.exists(p):
                    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        self.top_asset = img
                        print(f"Loaded top asset: {p}")
                        break
            except Exception:
                continue
        if self.top_asset is None:
            print("Top asset NOT loaded. Checked paths:")
            for p in top_candidates:
                print(f" - {p} {'(exists)' if os.path.exists(p) else '(missing)'}")

        # Load skirt asset
        skirt_candidates = [SKIRT_ASSET_PATH, os.path.join(script_dir, SKIRT_ASSET_PATH), os.path.join(script_dir, "assets", os.path.basename(SKIRT_ASSET_PATH))]
        for p in skirt_candidates:
            try:
                if os.path.exists(p):
                    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        self.skirt_asset = img
                        print(f"Loaded skirt asset: {p}")
                        break
            except Exception:
                continue

        # Load left arm asset
        arm_left_candidates = [
            ARM_LEFT_ASSET_PATH,
            os.path.join(script_dir, ARM_LEFT_ASSET_PATH),
            os.path.join(script_dir, "assets", os.path.basename(ARM_LEFT_ASSET_PATH))
        ]
        for p in arm_left_candidates:
            try:
                if os.path.exists(p):
                    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        self.arm_left_asset = img
                        print(f"Loaded left arm asset: {p}")
                        break
            except Exception:
                continue

        # Load right arm asset
        arm_right_candidates = [
            ARM_RIGHT_ASSET_PATH,
            os.path.join(script_dir, ARM_RIGHT_ASSET_PATH),
            os.path.join(script_dir, "assets", os.path.basename(ARM_RIGHT_ASSET_PATH))
        ]
        for p in arm_right_candidates:
            try:
                if os.path.exists(p):
                    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        self.arm_right_asset = img
                        print(f"Loaded right arm asset: {p}")
                        break
            except Exception:
                continue

        # Load all background images from folder
        self.background_images = []
        self.current_bg_index = 0
        
        # Search for background images in various locations
        bg_folders = [
            os.path.join(script_dir, "backgrounds"),
            os.path.join(script_dir, "assets", "backgrounds"),
            script_dir
        ]
        
        for folder in bg_folders:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        path = os.path.join(folder, file)
                        try:
                            img = cv2.imread(path)
                            if img is not None:
                                self.background_images.append(img)
                                print(f"Loaded background: {file}")
                        except Exception:
                            continue
        
        # Add gradient background as default
        self.background_images.insert(0, None)  # None = gradient
        print(f"Total backgrounds loaded: {len(self.background_images)} (including gradient)")

    # ===== POSE DETECTION & DRAWING (Original Frame) =====
    def detect_pose(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.append([lm.y, lm.x, lm.visibility])
            return np.array(keypoints)
        return np.zeros((33, 3))

    def draw_pose(self, frame, keypoints, threshold=0.5):
        h, w, _ = frame.shape
        edges = self.mp_pose.POSE_CONNECTIONS
        points = {}

        for i, (y, x, conf) in enumerate(keypoints):
            if conf > threshold:
                points[i] = (int(x * w), int(y * h))

        for start, end in edges:
            if start in points and end in points:
                cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

        for i, pt in points.items():
            conf = keypoints[i, 2]
            color = (147, 112, 219) if conf > 0.7 else (0, 255, 255)
            cv2.circle(frame, pt, 5, color, -1)

    # ===== HANDS DETECTION & DRAWING (Original Frame) =====
    def detect_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        keypoints = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                single_hand = []
                for lm in hand_landmarks.landmark:
                    single_hand.append([lm.x, lm.y])
                keypoints.append(np.array(single_hand))
        return keypoints

    def draw_hands(self, frame, keypoints):
        h, w, _ = frame.shape
        connections = self.mp_hands.HAND_CONNECTIONS

        for hand in keypoints:
            points = {}
            for i, (x, y) in enumerate(hand):
                px, py = int(x * w), int(y * h)
                points[i] = (px, py)

            for start, end in connections:
                if start in points and end in points:
                    cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

            for pt in points.values():
                cv2.circle(frame, pt, 5, (147, 112, 219), -1)

    # ===== FACE MESH DETECTION & DRAWING (Original Frame) =====
    def detect_faces(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        return results.multi_face_landmarks

    def draw_faces(self, frame, faces):
        h, w, _ = frame.shape
        if faces:
            for face_landmarks in faces:
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    # ===== HELPER: Calculate Eye Aspect Ratio (EAR) untuk deteksi kedipan =====
    def calculate_ear(self, eye_points):
        """
        Menghitung Eye Aspect Ratio untuk deteksi kedipan mata
        eye_points: list of 6 points [p1, p2, p3, p4, p5, p6]
        """
        # Vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        # Horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        if C == 0:
            return 0.3
        
        ear = (A + B) / (2.0 * C)
        return ear

    # ===== HELPER: Calculate Mouth Aspect Ratio (MAR) untuk deteksi buka mulut =====
    def calculate_mar(self, mouth_points):
        """
        Menghitung Mouth Aspect Ratio untuk deteksi buka/tutup mulut
        """
        # Vertical distance
        A = np.linalg.norm(mouth_points[1] - mouth_points[7])  # tengah atas-bawah
        B = np.linalg.norm(mouth_points[2] - mouth_points[6])  # kiri atas-bawah
        C = np.linalg.norm(mouth_points[3] - mouth_points[5])  # kanan atas-bawah
        # Horizontal distance
        D = np.linalg.norm(mouth_points[0] - mouth_points[4])  # kiri-kanan
        
        if D == 0:
            return 0.3
        
        mar = (A + B + C) / (3.0 * D)
        return mar

    # ===== ANIMATED AVATAR DRAWING =====
    def draw_animated_avatar(self, blank, pose_keypoints, hand_keypoints, faces, threshold=0.5):
        """
        Menggambar karakter animasi cartoon yang digerakkan oleh tracking data
        dengan ekspresi wajah yang dinamis
        """
        h, w, _ = blank.shape
        points = {}
        # Deferred front-hair strands to draw after legs (so hair overlays legs)
        deferred_front_hair = []
        # Store top overlay bbox to draw later (in front of torso layer)
        top_overlay_params = None

        # Konversi pose keypoints ke pixel coordinates
        for i, (y, x, conf) in enumerate(pose_keypoints):
            if conf > threshold:
                points[i] = (int(x * w), int(y * h))

        # Jika tidak ada pose terdeteksi, return
        if len(points) == 0:
            return

        # ===== EKSTRAK DATA WAJAH DARI FACE MESH =====
        face_data = None
        ear_left = 0.3
        ear_right = 0.3
        mar = 0.3
        
        if faces and len(faces) > 0:
            face_landmarks = faces[0].landmark
            
            # Indeks landmark untuk mata kiri (Eye)
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            # RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            
            # Indeks untuk mulut (Mouth)
            mouth_indices = [61, 291, 0, 17, 269, 405, 314, 17, 84]
            
            # Konversi ke koordinat
            left_eye_points = np.array([[face_landmarks[i].x * w, face_landmarks[i].y * h] 
                                            for i in left_eye_indices])
            right_eye_points = np.array([[face_landmarks[i].x * w, face_landmarks[i].y * h] 
                                             for i in right_eye_indices])
            mouth_points = np.array([[face_landmarks[i].x * w, face_landmarks[i].y * h] 
                                        for i in mouth_indices])
            
            # Hitung EAR dan MAR
            ear_left = self.calculate_ear(left_eye_points)
            ear_right = self.calculate_ear(right_eye_points)
            mar = self.calculate_mar(mouth_points)
            
            face_data = {
                'ear_left': ear_left,
                'ear_right': ear_right,
                'mar': mar
            }

        # ===== GAMBAR BADAN (TORSO) =====
        if 11 in points and 12 in points and 23 in points and 24 in points:
            left_shoulder = points[11]
            right_shoulder = points[12]
            left_hip = points[23]
            right_hip = points[24]
            
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2,
                              (left_shoulder[1] + right_shoulder[1]) // 2)
            hip_center = ((left_hip[0] + right_hip[0]) // 2,
                          (left_hip[1] + right_hip[1]) // 2)
            
            shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
            hip_width = np.linalg.norm(np.array(left_hip) - np.array(right_hip))
            
            torso_pts = np.array([
                [left_shoulder[0] - 15, left_shoulder[1] - 5],
                [right_shoulder[0] + 15, right_shoulder[1] - 5],
                [right_hip[0] + 20, right_hip[1] + 10],
                [left_hip[0] - 20, left_hip[1] + 10]
            ], np.int32)
            
            # Compute bounding box and store for top overlay later; draw base torso fill now
            try:
                x_min = int(np.min(torso_pts[:, 0]))
                y_min = int(np.min(torso_pts[:, 1]))
                x_max = int(np.max(torso_pts[:, 0]))
                y_max = int(np.max(torso_pts[:, 1]))

                pad_x = int(max(5, (x_max - x_min) * 0.08))
                pad_y = int(max(5, (y_max - y_min) * 0.06))
                x1 = max(0, x_min - pad_x)
                # Anchor the top just below the neck by using shoulder center and a small upward offset
                neck_anchor_y = shoulder_center[1] - int(TOP_NECK_OFFSET_RATIO * shoulder_width)
                y1 = max(0, max(y_min - pad_y, neck_anchor_y))
                x2 = min(blank.shape[1], x_max + pad_x)
                y2 = min(blank.shape[0], y_max + pad_y)

                # Apply scaling (widen/taller) and vertical offset (raise)
                orig_w = max(1, x2 - x1)
                orig_h = max(1, y2 - y1)

                center_x = (x1 + x2) // 2
                new_w = max(1, int(orig_w * TOP_SCALE_X))
                new_h = max(1, int(orig_h * TOP_SCALE_Y))

                new_x1 = max(0, center_x - new_w // 2)
                new_y1 = max(0, y1 - int(TOP_VERTICAL_OFFSET_PIX))  # positive offset moves up
                new_x2 = min(blank.shape[1], new_x1 + new_w)
                new_y2 = min(blank.shape[0], new_y1 + new_h)

                final_w = max(1, new_x2 - new_x1)
                final_h = max(1, new_y2 - new_y1)

                top_overlay_params = (new_x1, new_y1, final_w, final_h)
            except Exception:
                pass
            # Base torso fill behind the top asset
            cv2.fillPoly(blank, [torso_pts], (70, 130, 220))
            cv2.line(blank, shoulder_center, hip_center, (50, 100, 180), 3)
            
            collar_width = int(shoulder_width * 0.3)
            collar_pts = np.array([
                [shoulder_center[0] - collar_width, shoulder_center[1] - 10],
                [shoulder_center[0] + collar_width, shoulder_center[1] - 10],
                [shoulder_center[0] + collar_width//2, shoulder_center[1] + 10],
                [shoulder_center[0] - collar_width//2, shoulder_center[1] + 10]
            ], np.int32)
            cv2.fillPoly(blank, [collar_pts], (50, 100, 180))

        # ===== GAMBAR LENGAN Vektor (dasar) =====
        # Digambar dulu (di bawah aset lengan dan baju) agar aset bisa menimpa.
        arm_pairs_vector = [
            (11, 13, 15),  # left shoulder -> elbow -> wrist
            (12, 14, 16),  # right shoulder -> elbow -> wrist
        ]
        for shoulder_idx, elbow_idx, wrist_idx in arm_pairs_vector:
            if shoulder_idx in points and elbow_idx in points:
                shoulder = points[shoulder_idx]
                elbow = points[elbow_idx]
                cv2.line(blank, (shoulder[0] + 2, shoulder[1] + 2),
                         (elbow[0] + 2, elbow[1] + 2), SKIN_COLOR_SHADOW, 24)
                cv2.line(blank, shoulder, elbow, SKIN_COLOR_MAIN, 26)
                cv2.line(blank, shoulder, elbow, SKIN_COLOR_OUTLINE, 3)
            if elbow_idx in points and wrist_idx in points:
                elbow = points[elbow_idx]
                wrist = points[wrist_idx]
                cv2.line(blank, (elbow[0] + 2, elbow[1] + 2),
                         (wrist[0] + 2, wrist[1] + 2), SKIN_COLOR_SHADOW, 20)
                cv2.line(blank, elbow, wrist, SKIN_COLOR_MAIN, 22)
                cv2.line(blank, elbow, wrist, SKIN_COLOR_OUTLINE, 3)
            if elbow_idx in points:
                elbow = points[elbow_idx]
                cv2.circle(blank, (elbow[0] + 1, elbow[1] + 1), 10, SKIN_COLOR_SHADOW, -1)
                cv2.circle(blank, elbow, 10, SKIN_COLOR_MAIN, -1)
                cv2.circle(blank, elbow, 10, SKIN_COLOR_OUTLINE, 2)

        # ===== GAMBAR LENgAN (ARM) DI ATAS TORSO, DI BAWAH TOP =====
        def _prepare_arm(shoulder_idx, elbow_idx, wrist_idx, asset_img):
            try:
                if asset_img is None:
                    return None
                if shoulder_idx not in points or elbow_idx not in points or wrist_idx not in points:
                    return None
                shoulder = np.array(points[shoulder_idx])
                elbow = np.array(points[elbow_idx])
                wrist = np.array(points[wrist_idx])
                # orient by shoulder->wrist vector; scale to segment length
                vec = wrist - shoulder
                length = float(np.linalg.norm(vec))
                if length < 5:
                    return None
                angle = math.degrees(math.atan2(vec[1], vec[0]))
                h, w, ch = asset_img.shape
                scale = max(0.35, min(2.0, length / max(w, h)))
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                resized = cv2.resize(asset_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                center = (new_w // 2, new_h // 2)
                # Tentukan pivot (titik yang harus ditempel ke bahu) berdasarkan orientasi asset
                if new_w >= new_h:
                    pivot = (0, new_h // 2)          # asset horizontal: bahu di sisi kiri tengah
                else:
                    pivot = (new_w // 2, 0)          # asset vertical: bahu di bagian atas tengah
                # Matrix rotasi (di sekitar center)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(resized, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
                # Hitung posisi pivot setelah rotasi (transformasi 2D)
                px, py = pivot
                # Translate pivot ke koordinat relatif pusat
                rel = np.array([px - center[0], py - center[1]])
                rad = math.radians(angle)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                rel_rot = np.array([rel[0] * cos_a - rel[1] * sin_a, rel[0] * sin_a + rel[1] * cos_a])
                pivot_rot = center[0] + rel_rot[0], center[1] + rel_rot[1]
                # Tentukan top-left sehingga pivot_rot jatuh di titik bahu
                top_left_x = int(shoulder[0] - pivot_rot[0])
                top_left_y = int(shoulder[1] - pivot_rot[1])
                return rotated, (top_left_x, top_left_y)
            except Exception:
                return None

        # (Aset lengan dihapus; tidak digambar lagi)

        # ===== TOP OVERLAY (draw above arms) =====
        try:
            if hasattr(self, 'top_asset') and self.top_asset is not None and top_overlay_params is not None:
                x1, y1, tgt_w, tgt_h = top_overlay_params
                if tgt_w > 0 and tgt_h > 0:
                    top_img = cv2.resize(self.top_asset, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
                    if top_img.shape[2] == 4:
                        alpha = top_img[:, :, 3].astype(float) / 255.0
                        color = top_img[:, :, :3].astype(float)
                        roi = blank[y1:y1 + tgt_h, x1:x1 + tgt_w].astype(float)
                        alpha_3 = np.dstack([alpha, alpha, alpha])
                        blended = (alpha_3 * color + (1 - alpha_3) * roi).astype(np.uint8)
                        blank[y1:y1 + tgt_h, x1:x1 + tgt_w] = blended
                    else:
                        blank[y1:y1 + tgt_h, x1:x1 + tgt_w] = top_img
        except Exception:
            pass

        # ===== GAMBAR LEHER =====
        if 0 in points and 11 in points and 12 in points:
            nose = points[0]
            left_shoulder = points[11]
            right_shoulder = points[12]
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2,
                              (left_shoulder[1] + right_shoulder[1]) // 2)
            
            neck_width = 25
            neck_pts = np.array([
                [shoulder_center[0] - neck_width, shoulder_center[1]],
                [shoulder_center[0] + neck_width, shoulder_center[1]],
                [nose[0] + 15, nose[1] + 40],
                [nose[0] - 15, nose[1] + 40]
            ], np.int32)
            cv2.fillPoly(blank, [neck_pts], SKIN_COLOR_MAIN)

        # (moved) TOP OVERLAY will be drawn later, above arms

        # ===== GAMBAR KEPALA SIMETRIS =====
        if 0 in points:
            nose = points[0]
            
            # Hitung head center yang lebih simetris
            head_center = nose
            if 7 in points and 8 in points:  # ear left and right
                ear_left = points[7]
                ear_right = points[8]
                # Ambil center point dari kedua telinga untuk kepala yang simetris
                head_center = ((ear_left[0] + ear_right[0]) // 2, 
                              (ear_left[1] + ear_right[1]) // 2)
            else:
                head_center = (nose[0], nose[1] - 10)
            
            # Hitung ukuran kepala (tetap konstan)
            head_size = 50
            
            # ===== GAMBAR RAMBUT (DI BELAKANG KEPALA) =====
            # Rambut bagian belakang yang menutupi kepala (ukuran tetap)
            hair_width = 60  # 50 * 1.2
            hair_height = 65  # 50 * 1.3
            
            # Shadow rambut
            cv2.ellipse(blank, (head_center[0] + 3, head_center[1] - 3 + 3), 
                        (hair_width, hair_height), 0, 0, 360, HAIR_COLOR_SHADOW, -1) # DIGANTI
            
            # Main hair - bagian belakang
            cv2.ellipse(blank, (head_center[0], head_center[1] - 3), 
                        (hair_width, hair_height), 0, 0, 360, HAIR_COLOR_MAIN, -1) # DIGANTI
            
            # ===== RAMBUT PANJANG LURUS (SEBELUM KEPALA DIGAMBAR) =====
            # Fungsi untuk menggambar helai rambut lurus
            def draw_wavy_hair_strand(start_x, start_y, length, amplitude, direction=1):
                """Menggambar helai rambut lurus"""
                points = []
                num_points = 15
                
                for i in range(num_points):
                    t = i / (num_points - 1)
                    y = start_y + int(length * t)
                    # Rambut lurus - tidak ada wave effect
                    x = start_x  # Tetap lurus tanpa gelombang
                    points.append([x, y])
                
                points = np.array(points, np.int32)
                
                # Gambar dengan thickness yang berkurang di ujung
                for i in range(len(points) - 1):
                    thickness = max(2, int(8 * (1 - i / len(points))))
                    # Shadow
                    cv2.line(blank, tuple(points[i] + [1, 1]), tuple(points[i + 1] + [1, 1]), 
                             HAIR_COLOR_SHADOW, thickness + 1)
                    # Main strand
                    cv2.line(blank, tuple(points[i]), tuple(points[i + 1]), 
                             HAIR_COLOR_MAIN, thickness)
            
            # RAMBUT (Lebih banyak helai - total 28: 14 kiri + 14 kanan)
            left_hair_start = head_center[1] - 30  # 50 * 0.6
            hair_length = 125  # 50 * 2.5

            # Gambar 14 helai kiri, dari luar ke dalam
            for i in range(14):
                t = i / 13.0
                x_offset = int(48 - 25 * t)  # 50 * (0.95 - 0.5 * t)
                x = head_center[0] - x_offset
                start_y = left_hair_start + int(t * 30)
                length = hair_length - int(t * 40)
                amp = max(2, int(17 - t * 1))  # 50 * (0.34 - t * 0.02)
                draw_wavy_hair_strand(x, start_y, length, amplitude=amp, direction=-1)
                # Record a copy of this strand to draw later on top of legs (front hair)
                try:
                    deferred_front_hair.append((x, start_y, length, amp, -1))
                except Exception:
                    pass

            # Gambar 14 helai kanan (mirror dari kiri)
            for i in range(14):
                t = i / 13.0
                x_offset = int(48 - 25 * t)  # 50 * (0.95 - 0.5 * t)
                x = head_center[0] + x_offset
                start_y = left_hair_start + int(t * 30)
                length = hair_length - int(t * 40)
                amp = max(2, int(17 - t * 1))  # 50 * (0.34 - t * 0.02)
                draw_wavy_hair_strand(x, start_y, length, amplitude=amp, direction=1)
                # Record a copy of this strand to draw later on top of legs (front hair)
                try:
                    deferred_front_hair.append((x, start_y, length, amp, 1))
                except Exception:
                    pass
            
            # Shadow kepala
            cv2.ellipse(blank, (head_center[0] + 3, head_center[1] + 3), 
                        (50, 58), 0, 0, 360, SKIN_COLOR_SHADOW, -1)
            
            # Main head - cream skin (covers hair behind)
            cv2.ellipse(blank, head_center, (50, 58), 
                        0, 0, 360, SKIN_COLOR_MAIN, -1)
            cv2.ellipse(blank, head_center, (50, 58), 
                        0, 0, 360, SKIN_COLOR_OUTLINE, 2)
            
            # ===== PONI DEPAN (BAGIAN ATAS) =====
            # Poni bagian kiri dengan curve
            poni_pts_left = []
            for i in range(8):
                t = i / 7
                x = head_center[0] - int(45 - 25 * t)  # 50 * (0.9 - t * 0.5)
                y = head_center[1] - int(50 - 15 * t) + int(math.sin(t * math.pi) * 12)  # 50 * (1.0 - t * 0.3)
                poni_pts_left.append([x, y])
            
            # Tambah titik ke dahi untuk close shape
            poni_pts_left.append([head_center[0] - 20, head_center[1] - 35])  # 0.4 * 50, 0.7 * 50
            poni_pts_left.append([head_center[0] - 40, head_center[1] - 40])  # 0.8 * 50, 0.8 * 50
            
            cv2.fillPoly(blank, [np.array(poni_pts_left, np.int32)], HAIR_COLOR_MAIN) # DIGANTI
            
            # Poni bagian kanan (mirror)
            poni_pts_right = []
            for i in range(8):
                t = i / 7
                x = head_center[0] + int(45 - 25 * t)  # 50 * (0.9 - t * 0.5)
                y = head_center[1] - int(50 - 15 * t) + int(math.sin(t * math.pi) * 12)  # 50 * (1.0 - t * 0.3)
                poni_pts_right.append([x, y])
            
            poni_pts_right.append([head_center[0] + 20, head_center[1] - 35])  # 0.4 * 50, 0.7 * 50
            poni_pts_right.append([head_center[0] + 40, head_center[1] - 40])  # 0.8 * 50, 0.8 * 50
            
            cv2.fillPoly(blank, [np.array(poni_pts_right, np.int32)], HAIR_COLOR_MAIN) # DIGANTI
            
            # Poni tengah (middle bangs)
            poni_center = np.array([
                [head_center[0] - 15, head_center[1] - 48],  # 0.3*50, 0.95*50
                [head_center[0] - 8, head_center[1] - 53],   # 0.15*50, 1.05*50
                [head_center[0], head_center[1] - 54],       # 1.08*50
                [head_center[0] + 8, head_center[1] - 53],   # 0.15*50, 1.05*50
                [head_center[0] + 15, head_center[1] - 48],  # 0.3*50, 0.95*50
                [head_center[0], head_center[1] - 38],       # 0.75*50
            ], np.int32)
            cv2.fillPoly(blank, [poni_center], HAIR_COLOR_MAIN) # DIGANTI
            
            # Detail helai rambut poni (digambar sebelum wajah)
            for i in range(-2, 3):
                x_offset = int(8 * i)  # 50 * 0.15 * i
                start_y = head_center[1] - 49  # 0.98*50
                end_y = head_center[1] - 39    # 0.78*50
                cv2.line(blank, 
                         (head_center[0] + x_offset, start_y),
                         (head_center[0] + x_offset + (5 if i < 0 else -5), end_y),
                         HAIR_COLOR_SHADOW, 2)
            
            # Highlight rambut (kilau di poni)
            cv2.ellipse(blank, (head_center[0] - 10, head_center[1] - 43), 
                        (8, 5), 45, 0, 180, HAIR_COLOR_HIGHLIGHT, -1)  # 0.2*50, 0.85*50, 0.15*50, 0.1*50
            cv2.ellipse(blank, (head_center[0] + 10, head_center[1] - 43), 
                        (8, 5), 135, 0, 180, HAIR_COLOR_HIGHLIGHT, -1)
            
            # ===== POSISI MATA DAN MULUT YANG SIMETRIS =====
            # Gunakan head_center sebagai acuan
            eye_y = head_center[1] - 8  # 0.15*50
            eye_spacing = 18  # 0.35*50
            
            left_eye_pos = (head_center[0] - eye_spacing, eye_y)
            right_eye_pos = (head_center[0] + eye_spacing, eye_y)
            
            # Paksa mata tetap berada di posisi relatif terhadap head_center
            # dan gunakan ukuran konstan (tidak lagi override dari landmarks)
            # Catatan: kedipan masih memakai 'face_data["ear_left/right"]'
            # sehingga mata tetap bisa berkedip tetapi posisinya tidak bergeser.
            
            # ===== GAMBAR ALIS SIMETRIS (ANIME STYLE) =====
            # Alis kiri - lebih tipis dan melengkung
            eyebrow_left = [(left_eye_pos[0] - 18, left_eye_pos[1] - 18),
                            (left_eye_pos[0] + 18, left_eye_pos[1] - 20)]
            cv2.line(blank, eyebrow_left[0], eyebrow_left[1], (0, 0, 0), 3)
            
            # Alis kanan (simetris) - lebih tipis dan melengkung
            eyebrow_right = [(right_eye_pos[0] - 18, right_eye_pos[1] - 20),
                             (right_eye_pos[0] + 18, right_eye_pos[1] - 18)]
            cv2.line(blank, eyebrow_right[0], eyebrow_right[1], (0, 0, 0), 3)
            
            # ===== GAMBAR MATA DENGAN KEDIPAN (STYLE ANIME SEDERHANA) =====
            # Threshold untuk kedipan (EAR < 0.2 = mata tertutup)
            EAR_THRESHOLD = 0.2
            
            # Mata kiri (ukuran dan posisi tetap)
            # Putih mata
            cv2.ellipse(blank, left_eye_pos, (20, 25), 0, 0, 360, (255, 255, 255), -1)
            # Iris biru besar (konstan)
            cv2.circle(blank, (left_eye_pos[0], left_eye_pos[1] + 2), 16, (200, 120, 50), -1)
            # Pupil hitam (konstan)
            cv2.circle(blank, (left_eye_pos[0], left_eye_pos[1] + 3), 10, (0, 0, 0), -1)
            # Highlight putih (kilau mata)
            cv2.circle(blank, (left_eye_pos[0] - 6, left_eye_pos[1] - 2), 5, (255, 255, 255), -1)
            cv2.circle(blank, (left_eye_pos[0] + 7, left_eye_pos[1] + 5), 3, (255, 255, 255), -1)
            # Outline mata
            cv2.ellipse(blank, left_eye_pos, (20, 25), 0, 0, 360, (0, 0, 0), 2)

            # Mata kanan (ukuran dan posisi tetap)
            cv2.ellipse(blank, right_eye_pos, (20, 25), 0, 0, 360, (255, 255, 255), -1)
            cv2.circle(blank, (right_eye_pos[0], right_eye_pos[1] + 2), 16, (200, 120, 50), -1)
            cv2.circle(blank, (right_eye_pos[0], right_eye_pos[1] + 3), 10, (0, 0, 0), -1)
            cv2.circle(blank, (right_eye_pos[0] - 6, right_eye_pos[1] - 2), 5, (255, 255, 255), -1)
            cv2.circle(blank, (right_eye_pos[0] + 7, right_eye_pos[1] + 5), 3, (255, 255, 255), -1)
            cv2.ellipse(blank, right_eye_pos, (20, 25), 0, 0, 360, (0, 0, 0), 2)

            # ===== BLINK (overlay kelopak) =====
            # Jika landmark EAR menunjukkan kedipan, gambar kelopak di atas mata
            if face_data and face_data.get('ear_left', 1.0) < EAR_THRESHOLD:
                # Tutupi mata kiri dengan warna kulit lalu gambar garis kelopak
                cv2.ellipse(blank, left_eye_pos, (20, 25), 0, 0, 360, SKIN_COLOR_MAIN, -1)
                cv2.ellipse(blank, left_eye_pos, (20, 5), 0, 0, 360, (0, 0, 0), -1)

            if face_data and face_data.get('ear_right', 1.0) < EAR_THRESHOLD:
                cv2.ellipse(blank, right_eye_pos, (20, 25), 0, 0, 360, SKIN_COLOR_MAIN, -1)
                cv2.ellipse(blank, right_eye_pos, (20, 5), 0, 0, 360, (0, 0, 0), -1)
            
            # ===== GAMBAR HIDUNG SEDERHANA (ANIME STYLE) =====
            # Hidung kecil sederhana - hanya 2 titik kecil
            nose_y = head_center[1] + 5  # 0.10*50
            cv2.circle(blank, (head_center[0] - 3, nose_y), 2, (200, 150, 130), -1)
            cv2.circle(blank, (head_center[0] + 3, nose_y), 2, (200, 150, 130), -1)
            
            # ===== PIPI MERAH (BLUSH) =====
            cheek_y = head_center[1] + 8  # 0.15*50
            cheek_offset = 28  # 0.55*50
            # Pipi kiri
            cv2.ellipse(blank, (head_center[0] - cheek_offset, cheek_y), 
                        (13, 8), 
                        0, 0, 360, (150, 130, 200), -1)
            # Pipi kanan
            cv2.ellipse(blank, (head_center[0] + cheek_offset, cheek_y), 
                        (13, 8), 
                        0, 0, 360, (150, 130, 200), -1)
            
            # ===== GAMBAR MULUT DENGAN TRACKING (BUKA/TUTUP) - ANIME STYLE =====
            # Posisi mulut yang lebih presisi di tengah
            mouth_center = (head_center[0], head_center[1] + 20)  # 0.40*50
            
            mouth_width = 18  # 0.35*50
            
            # Threshold untuk mulut terbuka (MAR > 0.5 = mulut terbuka lebar)
            MAR_THRESHOLD = 0.5
            
            if face_data and face_data['mar'] > MAR_THRESHOLD:
                # Mulut terbuka lebar (bentuk O sederhana)
                mouth_open_height = int(mouth_width * 0.7 * min(face_data['mar'], 1.2))
                
                # Mulut terbuka - oval sederhana
                cv2.ellipse(blank, mouth_center, (mouth_width // 2, mouth_open_height), 
                            0, 0, 360, (180, 80, 80), -1)
                cv2.ellipse(blank, mouth_center, (mouth_width // 2, mouth_open_height), 
                            0, 0, 360, (0, 0, 0), 2)
            else:
                # Mulut tersenyum sederhana (anime style) - garis melengkung
                smile_points = []
                num_points = 10
                for i in range(num_points):
                    t = i / (num_points - 1)
                    x = int(mouth_center[0] - mouth_width + t * mouth_width * 2)
                    y = int(mouth_center[1] + math.sin(t * math.pi) * 8)
                    smile_points.append((x, y))
                
                # Gambar senyum
                for i in range(len(smile_points) - 1):
                    cv2.line(blank, smile_points[i], smile_points[i + 1], (150, 60, 60), 3)
            
            # ===== PIPI MERONA SIMETRIS =====
            cheek_left = (head_center[0] - 30, head_center[1] + 12)  # 0.6*50
            cheek_right = (head_center[0] + 30, head_center[1] + 12)
            overlay = blank.copy()
            cv2.ellipse(overlay, cheek_left, (20, 12), 0, 0, 360, (150, 130, 200), -1)
            cv2.ellipse(overlay, cheek_right, (20, 12), 0, 0, 360, (150, 130, 200), -1)
            cv2.addWeighted(overlay, 0.5, blank, 0.5, 0, blank)

        # (Vector arm block dipindahkan ke atas; bagian ini dihapus agar tidak menggambar ulang di atas aset.)

        # ===== GAMBAR TANGAN (HANDS) =====
        for hand in hand_keypoints:
            if len(hand) > 0:
                wrist = (int(hand[0][0] * w), int(hand[0][1] * h))
                
                # smaller wrist ellipses for a slimmer look
                cv2.ellipse(blank, (wrist[0] + 2, wrist[1] + 2), (18, 20), 0, 0, 360, SKIN_COLOR_SHADOW, -1)
                cv2.ellipse(blank, wrist, (18, 20), 0, 0, 360, SKIN_COLOR_MAIN, -1)
                cv2.ellipse(blank, wrist, (18, 20), 0, 0, 360, SKIN_COLOR_OUTLINE, 2)
                
                finger_tips = [4, 8, 12, 16, 20]
                finger_base = [3, 6, 10, 14, 18]
                
                for i, tip_idx in enumerate(finger_tips):
                    if tip_idx < len(hand) and finger_base[i] < len(hand):
                        tip_pos = (int(hand[tip_idx][0] * w), int(hand[tip_idx][1] * h))
                        base_pos = (int(hand[finger_base[i]][0] * w), int(hand[finger_base[i]][1] * h))
                        
                        cv2.line(blank, (wrist[0] + 1, wrist[1] + 1), 
                                 (base_pos[0] + 1, base_pos[1] + 1), SKIN_COLOR_SHADOW, 10)
                        cv2.line(blank, (base_pos[0] + 1, base_pos[1] + 1), 
                                 (tip_pos[0] + 1, tip_pos[1] + 1), SKIN_COLOR_SHADOW, 8)
                        
                        cv2.line(blank, wrist, base_pos, SKIN_COLOR_MAIN, 11)
                        cv2.line(blank, base_pos, tip_pos, SKIN_COLOR_MAIN, 9)
                        
                        # smaller fingertip circles
                        cv2.circle(blank, (tip_pos[0] + 1, tip_pos[1] + 1), 4, SKIN_COLOR_SHADOW, -1)
                        cv2.circle(blank, tip_pos, 4, SKIN_COLOR_MAIN, -1)
                        cv2.circle(blank, tip_pos, 4, SKIN_COLOR_OUTLINE, 1)

        # ===== GAMBAR KAKI (LEGS) =====
        leg_pairs = [
            (23, 25, 27, 31),  # left hip -> knee -> ankle -> foot
            (24, 26, 28, 32),  # right hip -> knee -> ankle -> foot
        ]
        
        for hip_idx, knee_idx, ankle_idx, foot_idx in leg_pairs:
            if hip_idx in points and knee_idx in points:
                hip = points[hip_idx]
                knee = points[knee_idx]

                cv2.line(blank, (hip[0] + 2, hip[1] + 2), 
                         (knee[0] + 2, knee[1] + 2), SKIN_COLOR_SHADOW, 26)
                cv2.line(blank, hip, knee, SKIN_COLOR_MAIN, 28)
                cv2.line(blank, hip, knee, SKIN_COLOR_OUTLINE, 3)
            
            if knee_idx in points and ankle_idx in points:
                knee = points[knee_idx]
                ankle = points[ankle_idx]

                cv2.line(blank, (knee[0] + 2, knee[1] + 2), 
                         (ankle[0] + 2, ankle[1] + 2), SKIN_COLOR_SHADOW, 24)
                cv2.line(blank, knee, ankle, SKIN_COLOR_MAIN, 26)
                cv2.line(blank, knee, ankle, SKIN_COLOR_OUTLINE, 3)
            
            if knee_idx in points:
                knee = points[knee_idx]
                cv2.circle(blank, (knee[0] + 1, knee[1] + 1), 16, SKIN_COLOR_SHADOW, -1)
                cv2.circle(blank, knee, 16, SKIN_COLOR_MAIN, -1)
                cv2.circle(blank, knee, 16, SKIN_COLOR_OUTLINE, 2)
            
            if ankle_idx in points:
                ankle = points[ankle_idx]
                
                if foot_idx in points:
                    foot = points[foot_idx]
                    shoe_pts = np.array([
                        [ankle[0] - 15, ankle[1]],
                        [ankle[0] + 15, ankle[1]],
                        [foot[0] + 25, foot[1]],
                        [foot[0] - 10, foot[1]]
                    ], np.int32)
                    cv2.fillPoly(blank, [shoe_pts + [2, 2]], (60, 30, 30))
                    cv2.fillPoly(blank, [shoe_pts], (120, 60, 60))
                    cv2.polylines(blank, [shoe_pts], True, (80, 40, 40), 2)
                    cv2.circle(blank, ankle, 10, (100, 50, 50), -1)
                else:
                    cv2.ellipse(blank, (ankle[0] + 2, ankle[1] + 2), (22, 18), 0, 0, 360, (60, 30, 30), -1)
                    cv2.ellipse(blank, ankle, (22, 18), 0, 0, 360, (120, 60, 60), -1)
                    cv2.ellipse(blank, ankle, (22, 18), 0, 0, 360, (80, 40, 40), 2)

        # ===== SKIRT OVERLAY (attach at waist/hips if asset available) =====
        try:
            if hasattr(self, 'skirt_asset') and self.skirt_asset is not None and 23 in points and 24 in points:
                left_hip = points[23]
                right_hip = points[24]
                hip_center = ((left_hip[0] + right_hip[0]) // 2, (left_hip[1] + right_hip[1]) // 2)

                # Determine skirt width and height
                hip_width = int(np.linalg.norm(np.array(left_hip) - np.array(right_hip)))

                # Prefer knees to size skirt vertically; fallback to hip_width multiplier
                if 25 in points and 26 in points:
                    left_knee = points[25]
                    right_knee = points[26]
                    knee_y = min(left_knee[1], right_knee[1])
                    skirt_h = max(20, knee_y - hip_center[1])
                else:
                    skirt_h = int(max(hip_width * 1.4, hip_width // 1))

                skirt_w = int(max(hip_width * 2.8, hip_width + 20))

                # Position skirt so its top aligns slightly above the hip center
                x_min = hip_center[0] - skirt_w // 2
                y_min = hip_center[1] - int(skirt_h * SKIRT_VERTICAL_OFFSET)

                x1 = max(0, x_min)
                y1 = max(0, y_min)
                x2 = min(blank.shape[1], x_min + skirt_w)
                y2 = min(blank.shape[0], y_min + skirt_h)

                if x2 > x1 and y2 > y1:
                    tgt_w = x2 - x1
                    tgt_h = y2 - y1
                    print(f"Skirt bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}, tgt_w={tgt_w}, tgt_h={tgt_h}")
                    sk = cv2.resize(self.skirt_asset, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
                    if sk.shape[2] == 4:
                        alpha = sk[:, :, 3].astype(float) / 255.0
                        color = sk[:, :, :3].astype(float)
                        roi = blank[y1:y1 + tgt_h, x1:x1 + tgt_w].astype(float)
                        alpha_3 = np.dstack([alpha, alpha, alpha])
                        blended = (alpha_3 * color + (1 - alpha_3) * roi).astype(np.uint8)
                        blank[y1:y1 + tgt_h, x1:x1 + tgt_w] = blended
                    else:
                        blank[y1:y1 + tgt_h, x1:x1 + tgt_w] = sk
        except Exception:
            pass
class TkinterTrackerApp:
    def __init__(self, tracker):
        self.tracker = tracker
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Tidak bisa membuka kamera. Pastikan kamera terhubung.")
        

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        # Setup window
        self.root = tk.Tk()
        self.root.title("MediaPipe Tracker - Animated Avatar with Expression")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        main = tk.Frame(self.root, padx=10, pady=10, bg="#1a1a1a")
        main.pack(expand=True, fill="both")

        # Styling untuk label
        label_style = {"font": ("Arial", 12, "bold"), "bg": "#1a1a1a", "fg": "white"}
        
        tk.Label(main, text="Webcam + Tracking", **label_style).grid(row=0, column=0, pady=6)
        tk.Label(main, text="Animated Avatar (Blink & Mouth Tracking)", **label_style).grid(row=0, column=1, pady=6)

        self.canvas1 = tk.Canvas(main, width=self.width, height=self.height, bg="black", highlightthickness=2, highlightbackground="#444")
        self.canvas1.grid(row=1, column=0, padx=6, pady=6)
        
        # Background gradient untuk avatar canvas
        self.canvas2 = tk.Canvas(main, width=self.width, height=self.height, bg="#0a0a0a", highlightthickness=2, highlightbackground="#444")
        self.canvas2.grid(row=1, column=1, padx=6, pady=6)

        self.photo1 = None
        self.photo2 = None

        # Bind keyboard events
        self.root.bind('<b>', self.change_background)
        self.root.bind('<B>', self.change_background)

        self.delay = 15
        self.update()
        self.root.mainloop()
    
    def change_background(self, event=None):
        """Ganti ke background berikutnya saat tombol 'B' ditekan"""
        total_bgs = len(self.tracker.background_images)
        if total_bgs > 0:
            self.tracker.current_bg_index = (self.tracker.current_bg_index + 1) % total_bgs
            bg_name = "Gradient" if self.tracker.current_bg_index == 0 else f"Image {self.tracker.current_bg_index}"
            print(f"Background changed to: {bg_name} ({self.tracker.current_bg_index + 1}/{total_bgs})")

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            # Detections
            pose_keypoints = self.tracker.detect_pose(frame)
            hand_keypoints = self.tracker.detect_hands(frame)
            faces = self.tracker.detect_faces(frame)

            # Draw on original frame
            self.tracker.draw_pose(frame, pose_keypoints)
            self.tracker.draw_hands(frame, hand_keypoints)
            self.tracker.draw_faces(frame, faces)

            # Prepare animated avatar frame
            blank = np.zeros_like(frame)
            
            # Use current background
            current_bg = self.tracker.background_images[self.tracker.current_bg_index]
            if current_bg is not None:
                # Resize background to match frame size
                bg_resized = cv2.resize(current_bg, (self.width, self.height))
                blank = bg_resized.copy()
            else:
                # Gradient background (dark blue to black)
                for i in range(self.height):
                    color_value = int(20 * (1 - i / self.height))
                    blank[i, :] = [color_value, color_value, color_value + 10]
            
            # Draw animated character with expressions
            self.tracker.draw_animated_avatar(blank, pose_keypoints, hand_keypoints, faces)

            # Convert BGR->RGB and to ImageTk
            img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
            im1 = Image.fromarray(img1)
            im2 = Image.fromarray(img2)
            self.photo1 = ImageTk.PhotoImage(image=im1)
            self.photo2 = ImageTk.PhotoImage(image=im2)

            # Update canvases
            self.canvas1.create_image(0, 0, image=self.photo1, anchor='nw')
            self.canvas2.create_image(0, 0, image=self.photo2, anchor='nw')

        self.root.after(self.delay, self.update)

    def on_close(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    try:
        tracker = PoseHandFaceTracker()
        app = TkinterTrackerApp(tracker)
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Terjadi kesalahan yang tidak terduga: {e}")