import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


class PoseHandFaceTracker:
    def __init__(self):
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
            max_num_faces=2,  # bisa lebih dari 1 wajah
            refine_landmarks=True,  # aktifkan landmark iris mata
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # ===== POSE =====
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

    def draw_pose_on_blank(self, blank, keypoints, threshold=0.5):
        h, w, _ = blank.shape
        edges = self.mp_pose.POSE_CONNECTIONS
        points = {}

        for i, (y, x, conf) in enumerate(keypoints):
            if conf > threshold:
                points[i] = (int(x * w), int(y * h))

        for start, end in edges:
            if start in points and end in points:
                cv2.line(blank, points[start], points[end], (0, 255, 0), 2)

        for i, pt in points.items():
            cv2.circle(blank, pt, 5, (0, 255, 255), -1)

    # ===== HANDS =====
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

    def draw_hands_on_blank(self, blank, keypoints):
        h, w, _ = blank.shape
        connections = self.mp_hands.HAND_CONNECTIONS

        for hand in keypoints:
            points = {}
            for i, (x, y) in enumerate(hand):
                px, py = int(x * w), int(y * h)
                points[i] = (px, py)

            for start, end in connections:
                if start in points and end in points:
                    cv2.line(blank, points[start], points[end], (0, 255, 0), 2)

            for pt in points.values():
                cv2.circle(blank, pt, 5, (0, 255, 255), -1)

    # ===== FACE MESH =====
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

    def draw_faces_on_blank(self, blank, faces):
        h, w, _ = blank.shape
        if faces:
            for face_landmarks in faces:
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(blank, (x, y), 1, (255, 255, 255), -1)

    # ===== MAIN LOOP =====
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Tidak bisa membuka kamera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Pose
            pose_keypoints = self.detect_pose(frame)
            self.draw_pose(frame, pose_keypoints)

            # Hands
            hand_keypoints = self.detect_hands(frame)
            self.draw_hands(frame, hand_keypoints)

            # Faces (Face Mesh)
            faces = self.detect_faces(frame)
            self.draw_faces(frame, faces)

            # === WINDOW 1: webcam + tracking ===
            cv2.imshow("🤖 Webcam + Tracking (Pose + Hands + Face)", frame)

            # === WINDOW 2: hanya skeleton (pose + hands + face) di background hitam ===
            blank = np.zeros_like(frame)
            self.draw_pose_on_blank(blank, pose_keypoints)
            self.draw_hands_on_blank(blank, hand_keypoints)
            self.draw_faces_on_blank(blank, faces)
            cv2.imshow("🎬 Skeleton + Face Mesh Only", blank)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


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
        self.root.title("MediaPipe Tracker - Tkinter")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        main = tk.Frame(self.root, padx=10, pady=10)
        main.pack(expand=True, fill="both")

        tk.Label(main, text="Webcam + Tracking", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=6)
        tk.Label(main, text="Skeleton + Face Mesh", font=("Arial", 12, "bold")).grid(row=0, column=1, pady=6)

        self.canvas1 = tk.Canvas(main, width=self.width, height=self.height, bg="black")
        self.canvas1.grid(row=1, column=0, padx=6, pady=6)
        self.canvas2 = tk.Canvas(main, width=self.width, height=self.height, bg="black")
        self.canvas2.grid(row=1, column=1, padx=6, pady=6)

        self.photo1 = None
        self.photo2 = None

        self.delay = 15
        self.update()  # start update loop
        self.root.mainloop()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            # detections
            pose_keypoints = self.tracker.detect_pose(frame)
            hand_keypoints = self.tracker.detect_hands(frame)
            faces = self.tracker.detect_faces(frame)

            # draw on original frame
            self.tracker.draw_pose(frame, pose_keypoints)
            self.tracker.draw_hands(frame, hand_keypoints)
            self.tracker.draw_faces(frame, faces)

            # prepare skeleton-only blank
            blank = np.zeros_like(frame)
            self.tracker.draw_pose_on_blank(blank, pose_keypoints)
            self.tracker.draw_hands_on_blank(blank, hand_keypoints)
            self.tracker.draw_faces_on_blank(blank, faces)

            # convert BGR->RGB and to ImageTk
            img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
            im1 = Image.fromarray(img1)
            im2 = Image.fromarray(img2)
            self.photo1 = ImageTk.PhotoImage(image=im1)
            self.photo2 = ImageTk.PhotoImage(image=im2)

            # update canvases
            self.canvas1.create_image(0, 0, image=self.photo1, anchor='nw')
            self.canvas2.create_image(0, 0, image=self.photo2, anchor='nw')

        self.root.after(self.delay, self.update)

    def on_close(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    tracker = PoseHandFaceTracker()
    app = TkinterTrackerApp(tracker)
