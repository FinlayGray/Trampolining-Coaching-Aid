import sys
import os
import math
import joblib
import psycopg2
from psycopg2 import pool
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QPushButton, QVBoxLayout, QWidget,
    QTabWidget, QStackedWidget, QFileDialog, QLabel, QScrollArea, QSpacerItem,
    QSizePolicy, QLineEdit, QMessageBox, QComboBox, QListWidget, QListWidgetItem,
    QHBoxLayout
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2
import cv2

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------------------------
# Database Manager
# ------------------------------------------------------------------

class DatabaseManager:
    def __init__(self, host="localhost", database="trampolining", user="postgres", password="postgres", minconn=1, maxconn=10):
        try:
            self.pool = pool.ThreadedConnectionPool(minconn, maxconn,
                                                      host=host,
                                                      database=database,
                                                      user=user,
                                                      password=password)
            if self.pool:
                print("Connection pool created successfully")
        except Exception as e:
            print("Error creating connection pool:", e)
            sys.exit(1)

    def get_connection(self):
        return self.pool.getconn()

    def release_connection(self, conn):
        self.pool.putconn(conn)

    def close_all(self):
        self.pool.closeall()

    def get_user(self, username):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, username, password, account_type FROM users WHERE username = %s", (username,))
                row = cur.fetchone()
                if row:
                    return {"id": row[0], "username": row[1], "password": row[2], "account_type": row[3]}
                return None
        finally:
            self.release_connection(conn)

    def add_user(self, username, password, account_type):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO users (username, password, account_type) VALUES (%s, %s, %s)",
                            (username, password, account_type))
                conn.commit()
        finally:
            self.release_connection(conn)

    def add_gymnast_to_coach(self, coach_id, gymnast_id):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO coach_gymnast (coach_id, gymnast_id) VALUES (%s, %s)",
                            (coach_id, gymnast_id))
                conn.commit()
        finally:
            self.release_connection(conn)

    def get_gymnasts_for_coach(self, coach_id):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT gymnast_id FROM coach_gymnast WHERE coach_id = %s", (coach_id,))
                rows = cur.fetchall()
                return [row[0] for row in rows]
        finally:
            self.release_connection(conn)

    def send_invitation(self, coach_id, gymnast_id):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO group_invites (coach_id, gymnast_id, status) VALUES (%s, %s, %s)",
                            (coach_id, gymnast_id, 'pending'))
                conn.commit()
        finally:
            self.release_connection(conn)

    def get_invitations_for_gymnast(self, gymnast_id):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT coach_id, status FROM group_invites WHERE gymnast_id = %s AND status = 'pending'", (gymnast_id,))
                rows = cur.fetchall()
                return [{"coach_id": row[0], "status": row[1]} for row in rows]
        finally:
            self.release_connection(conn)

    def update_invitation_status(self, coach_id, gymnast_id, new_status):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE group_invites SET status = %s WHERE coach_id = %s AND gymnast_id = %s",
                            (new_status, coach_id, gymnast_id))
                conn.commit()
        finally:
            self.release_connection(conn)

    def get_latest_submission_by_gymnast(self, gymnast_id):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT submission_id, overall_score, submission_date
                    FROM submissions
                    WHERE gymnast_id = %s
                    ORDER BY submission_date DESC
                    LIMIT 1;
                """, (gymnast_id,))
                row = cur.fetchone()
                if row:
                    return {"submission_id": row[0], "overall_score": row[1], "submission_date": row[2]}
                return None
        finally:
            self.release_connection(conn)

    def get_all_gymnasts(self):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, username FROM users WHERE account_type = 'Gymnast'")
                rows = cur.fetchall()
                return [{"id": row[0], "username": row[1]} for row in rows]
        finally:
            self.release_connection(conn)

    def get_user_by_id(self, user_id):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, username, account_type FROM users WHERE id = %s", (user_id,))
                row = cur.fetchone()
                if row:
                    return {"id": row[0], "username": row[1], "account_type": row[2]}
                return None
        finally:
            self.release_connection(conn)

    def get_coaches_for_gymnast(self, gymnast_id):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT coach_id FROM coach_gymnast WHERE gymnast_id = %s", (gymnast_id,))
                rows = cur.fetchall()
                return [row[0] for row in rows]
        finally:
            self.release_connection(conn)

    def update_user_password(self, user_id, new_password):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET password = %s WHERE id = %s", (new_password, user_id))
                conn.commit()
        finally:
            self.release_connection(conn)

    def add_submission(self, gymnast_id, overall_score):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO submissions (gymnast_id, overall_score) 
                    VALUES (%s, %s) RETURNING submission_id
                    """,
                    (gymnast_id, overall_score)
                )
                submission_id = cur.fetchone()[0]
                conn.commit()
                return submission_id
        finally:
            self.release_connection(conn)

    def add_skill_scores(self, submission_id, skill_records):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                for record in skill_records:
                    skill_No, skill_Name, deduction_value = record
                    cur.execute(
                        """
                        INSERT INTO skill_scores (submission_id, skill_No, skill_Name, deduction_value)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (submission_id, skill_No, skill_Name, deduction_value)
                    )
                conn.commit()
        finally:
            self.release_connection(conn)

    def add_deduction_details(self, submission_id, deduction_details):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                for record in deduction_details:
                    skill_No, deduction_reason = record
                    cur.execute(
                        """
                        INSERT INTO deduction_details (submission_id, skill_No, deduction_reason)
                        VALUES (%s, %s, %s)
                        """,
                        (submission_id, skill_No, deduction_reason)
                    )
                conn.commit()
        finally:
            self.release_connection(conn)

    def get_submissions_by_gymnast(self, gymnast_id):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT submission_id, overall_score, submission_date
                    FROM submissions
                    WHERE gymnast_id = %s
                    ORDER BY submission_date;
                """, (gymnast_id,))
                rows = cur.fetchall()
                return [{"submission_id": row[0], "overall_score": row[1], "submission_date": row[2]} for row in rows]
        finally:
            self.release_connection(conn)

    def get_skill_scores_by_submission(self, submission_id):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT skill_No, skill_Name, deduction_value
                    FROM skill_scores
                    WHERE submission_id = %s
                    ORDER BY skill_No;
                """, (submission_id,))
                rows = cur.fetchall()
                return [{"skill_No": row[0], "skill_Name": row[1], "deduction_value": row[2]} for row in rows]
        finally:
            self.release_connection(conn)
    def remove_gymnast_from_coach(self, coach_id, gymnast_id):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM coach_gymnast WHERE coach_id = %s AND gymnast_id = %s", (coach_id, gymnast_id))
                conn.commit()
        finally:
            self.release_connection(conn)


    def get_deduction_details_by_submission(self, submission_id):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT skill_No, deduction_reason
                    FROM deduction_details
                    WHERE submission_id = %s
                    ORDER BY skill_No;
                """, (submission_id,))
                rows = cur.fetchall()
                return [{"skill_No": row[0], "deduction_reason": row[1]} for row in rows]
        finally:
            self.release_connection(conn)

# Global database manager instance
db_manager = DatabaseManager()

# ------------------------------------------------------------------
# Global Variables, Model Paths & Dummy User Database
# ------------------------------------------------------------------

model_path = '/Users/finlaygray/Documents/diss/pose_landmarker_full.task'
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

num_features_per_frame = None

users = {"test": {"password": "test", "account_type": "Gymnast"}}

# ------------------------------------------------------------------
# In-Memory Splitting
# ------------------------------------------------------------------

def watch_and_split_video_in_memory(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return [], 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []
    split_frames = [0]
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Press SPACE to Split, 'q' to Quit", frame)
        all_frames.append(frame.copy())
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord(' '):
            if current_frame != split_frames[-1]:
                split_frames.append(current_frame)
        elif key == ord('q'):
            break
        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()

    if split_frames[-1] != len(all_frames):
        split_frames.append(len(all_frames))

    segments = []
    for i in range(len(split_frames) - 1):
        start_frame = split_frames[i]
        end_frame = split_frames[i + 1]
        segments.append(all_frames[start_frame:end_frame])
    return segments, fps

# ------------------------------------------------------------------
# Convert In-Memory Frames -> Flattened Features
# ------------------------------------------------------------------

def convert_vid(file_path, invert=False):
    landmarker = PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    frame_index = 0
    video_features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if invert:
            frame = cv2.flip(frame, 1)
        frame_timestamp_ms = int((frame_index / fps) * 1000)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame_rgb))
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        annotated = draw_landmarks_on_image(frame_rgb, pose_landmarker_result)
        cv2.imshow('Video Frame with Pose Landmarks', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_features = extract_features_from_frame(pose_landmarker_result)
        if frame_features is not None:
            video_features.append(frame_features)
        frame_index += 1
    return video_features

def convert_vid_in_memory(frames, fps=30, invert=False):
    global num_features_per_frame
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    video_features = []
    for idx, frame in enumerate(frames):
        if invert:
            frame = cv2.flip(frame, 1)
        frame_timestamp_ms = int((idx / fps) * 1000)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame_rgb))
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        frame_features = extract_features_from_frame(pose_landmarker_result)
        if frame_features is not None:
            video_features.append(frame_features)
    return video_features

# ------------------------------------------------------------------
# Extracting landmarks from frame
# ------------------------------------------------------------------

def extract_features_from_frame(pose_landmarker_result):
    global num_features_per_frame
    features = []
    if pose_landmarker_result.pose_landmarks:
        num_landmarks = len(pose_landmarker_result.pose_landmarks[0])
        if num_features_per_frame is None:
            num_features_per_frame = num_landmarks * 5
        for landmark in pose_landmarker_result.pose_landmarks[0]:
            features.extend([
                landmark.x, landmark.y, landmark.z,
                landmark.visibility, landmark.presence
            ])
    else:
        if num_features_per_frame is not None:
            features = [0.0] * num_features_per_frame
        else:
            return None
    return features

def get_landmark(flat_landmarks, i):
    return [
        flat_landmarks[0 + i*5],
        flat_landmarks[1 + i*5],
        flat_landmarks[2 + i*5],
        flat_landmarks[3 + i*5],
        flat_landmarks[4 + i*5]
    ]

# ------------------------------------------------------------------
# Calculating angles from landmarks
# ------------------------------------------------------------------

def angle_2d(landmarkA, landmarkB, landmarkC):
    Ax, Ay = landmarkA[0], landmarkA[1]
    Bx, By = landmarkB[0], landmarkB[1]
    Cx, Cy = landmarkC[0], landmarkC[1]
    BAx = Ax - Bx
    BAy = Ay - By
    BCx = Cx - Bx
    BCy = Cy - By
    dot = (BAx * BCx) + (BAy * BCy)
    magBA = math.sqrt(BAx**2 + BAy**2)
    magBC = math.sqrt(BCx**2 + BCy**2)
    if magBA == 0 or magBC == 0:
        return 0.0
    cos_angle = dot / (magBA * magBC)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    angle = math.degrees(math.acos(cos_angle))
    return angle

def angle_from_vertical(landmarkHip, landmarkAnkle):
    hx, hy = landmarkHip[0], landmarkHip[1]
    ax, ay = landmarkAnkle[0], landmarkAnkle[1]
    vx = ax - hx
    vy = ay - hy
    angle_degs = math.degrees(math.atan2(vy, vx))
    angle_vertical = angle_degs - 90
    while angle_vertical < 0:
        angle_vertical += 360
    while angle_vertical >= 360:
        angle_vertical -= 360
    if angle_vertical > 180:
        angle_vertical = 360 - angle_vertical
    return abs(angle_vertical)

def angle_from_verticaltest(x1, y1, x2, y2):
    dx = x2 - x1    
    dy = y2 - y1
    angle_radians = math.atan2(dx, dy)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def compute_all_angles(flat_landmarks):
    left_shoulder  = get_landmark(flat_landmarks, 11)
    right_shoulder = get_landmark(flat_landmarks, 12)
    left_elbow     = get_landmark(flat_landmarks, 13)
    right_elbow    = get_landmark(flat_landmarks, 14)
    left_wrist     = get_landmark(flat_landmarks, 15)
    right_wrist    = get_landmark(flat_landmarks, 16)
    left_hip       = get_landmark(flat_landmarks, 23)
    right_hip      = get_landmark(flat_landmarks, 24)
    left_knee      = get_landmark(flat_landmarks, 25)
    right_knee     = get_landmark(flat_landmarks, 26)
    left_ankle     = get_landmark(flat_landmarks, 27)
    right_ankle    = get_landmark(flat_landmarks, 28)
    left_heel      = get_landmark(flat_landmarks, 29)
    right_heel     = get_landmark(flat_landmarks, 30)
    left_toe       = get_landmark(flat_landmarks, 31)
    right_toe      = get_landmark(flat_landmarks, 32)
    angles = {}
    angles["left_knee_angle"] = angle_2d(left_hip, left_knee, left_ankle)
    angles["right_knee_angle"] = angle_2d(right_hip, right_knee, right_ankle)
    angles["left_elbow_angle"] = angle_2d(left_shoulder, left_elbow, left_wrist)
    angles["right_elbow_angle"] = angle_2d(right_shoulder, right_elbow, right_wrist)
    angles["left_hip_angle"] = angle_2d(left_shoulder, left_hip, left_knee)
    angles["right_hip_angle"] = angle_2d(right_shoulder, right_hip, right_knee)
    angles["left_leg_vertical_angle"] = angle_from_vertical(left_hip, left_ankle)
    angles["right_leg_vertical_angle"] = angle_from_vertical(right_hip, right_ankle)
    angles["left_knee_vertical_angle"] = angle_from_vertical(left_hip, left_knee)
    angles["right_knee_vertical_angle"] = angle_from_vertical(right_hip, right_knee)
    angles["right_shoulder_hip_knee"] = angle_2d(right_shoulder, right_hip, right_knee)
    angles["left_shoulder_hip_knee"] = angle_2d(left_shoulder, left_hip, left_knee)
    angles["right_pointed_toes"] = angle_2d(right_ankle, right_heel, right_toe)
    angles["left_pointed_toes"] = angle_2d(left_ankle, left_heel, left_toe)
    ankle_diff = abs(left_ankle[0] - right_ankle[0])
    angles['ank_dif'] = ankle_diff
    angles["legs_together"] = ankle_diff < 0.02
    center_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
    center_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
    angles["trunk_angle"] = angle_from_vertical(center_hip, center_shoulder)
    return angles

def compute_video_angles(video_features):
    video_angles = []
    for frame_feats in video_features:
        if not frame_feats:
            video_angles.append({})
            continue
        angle_dict = compute_all_angles(frame_feats)
        video_angles.append(angle_dict)
    return video_angles

# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------

def pad_frame_features(frame_features, target_length):
    current_length = len(frame_features)
    if current_length < target_length:
        frame_features.extend([0.0] * (target_length - current_length))
    elif current_length > target_length:
        frame_features = frame_features[:target_length]
    return frame_features

def predict_video_in_memory(frames, fps):
    global num_features_per_frame
    model = keras.models.load_model('classification_model.h5')
    lb = joblib.load('label_encoder.joblib')
    video_features = convert_vid_in_memory(frames, fps=fps)
    if not video_features:
        return ("unknown_jump", [], [])
    if num_features_per_frame is None:
        num_features_per_frame = len(video_features[0])
    for i in range(len(video_features)):
        video_features[i] = pad_frame_features(video_features[i], num_features_per_frame)
    video_angles = compute_video_angles(video_features)
    max_seq_len = model.input_shape[1]
    video_padded = pad_sequences([video_features], maxlen=max_seq_len,
                                  dtype='float32', padding='post')
    probs = model.predict(video_padded)
    idx = np.argmax(probs, axis=1)
    onehot = np.eye(len(lb.categories_[0]))[idx]
    predicted_label = lb.inverse_transform(onehot)
    jump_label = predicted_label[0][0]
    return jump_label, video_features, video_angles

def angle_from_vertical_hip_ankle(hip_x, hip_y, ankle_x, ankle_y):
    dx = ankle_x - hip_x
    dy = ankle_y - hip_y
    angle_radians = math.atan2(dx, dy)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

# ------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------

def manual_execution(jump, features):
    deduction = 0
    reason = []
    if jump in ["straddle_jump", "pike_jump"]:
        deduction = 0.5
        leg_deduction = 0
        toes_deduction = 0
        start_idx = int(len(features) * 0.4)
        end_idx   = int(len(features) * 0.6)
        for f in features[start_idx:end_idx]:
            angle1 = f["right_leg_vertical_angle"]
            angle2 = f["left_leg_vertical_angle"]
            legs = f["right_knee_angle"]
            legs2 = f["left_knee_angle"]
            toes1 = f["right_pointed_toes"]
            toes2 = f["left_pointed_toes"]
            if legs <= 120 or legs2 <= 120:
                leg_deduction = 0.1
                reason.append('Legs Bent')
            if toes1 < 30 or toes2 < 30:
                toes_deduction = 0.1
                reason.append('Toes Flexed')
            if angle1 >= 145. or angle2 >= 145.0:
                continue
            if angle1 >= 90 and angle2 >= 90:
                deduction = 0
                break
            if angle1 >= 65 and angle1 < 90 and angle2 >= 65 and angle2 < 90:
                new_deduction = 0.1
            elif angle1 >= 45 and angle1 < 65 and angle2 >= 45 and angle2 < 65:
                new_deduction = 0.2
            else:
                new_deduction = 0.5
            deduction = min(deduction, new_deduction)
        if deduction > 0:
            reason.append('Loose shape')
        deduction = min(0.5, deduction + toes_deduction + leg_deduction)
    elif jump == 'tuck_jump':
        deduction = 0.5
        toes_deduction = 0
        start_idx = int(len(features) * 0.3)
        end_idx   = int(len(features) * 0.7)
        for f in features[start_idx:end_idx]:
            angle1 = f["right_knee_vertical_angle"]
            angle2 = f["left_knee_vertical_angle"]
            toes1 = f["right_pointed_toes"]
            toes2 = f["left_pointed_toes"]
            if toes1 < 30 and toes2 < 30:
                toes_deduction = 0.1
                reason.append('Toes Flexed')
            if angle1 >= 135 and angle2 >= 135:
                deduction = 0
                break
            if angle1 >= 90 and angle1 < 135 and angle2 >= 90 and angle2 < 135:
                new_deduction = 0.1
            else:
                new_deduction = 0.2
            deduction = min(deduction, new_deduction)
        if deduction > 0:
            reason.append('Loose shape')
        deduction = min(0.5, deduction + toes_deduction)
    elif jump == 'tuck_back':
        deduction = 0.5
        toes_deduction = 0
        start_idx = int(len(features) * 0.3)
        end_idx   = int(len(features) * 0.7)
        for f in features[start_idx:end_idx]:
            angle1 = f["right_shoulder_hip_knee"]
            angle2 = f["left_shoulder_hip_knee"]
            toes1 = f["right_pointed_toes"]
            toes2 = f["left_pointed_toes"]
            if toes1 < 30 and toes2 < 30:
                toes_deduction = 0.1
                reason.append('Toes Flexed')
            if angle1 == 0 or angle2 == 0:
                continue
            if angle1 <= 45 and angle2 <= 45:
                deduction = 0
                break
            if angle1 >= 45 and angle1 < 90 and angle2 >= 45 and angle2 < 90:
                new_deduction = 0.1
            else:
                new_deduction = 0.2
            deduction = min(deduction, new_deduction)
        if deduction > 0:
            reason.append('Loose shape')
        deduction = min(0.5, deduction + toes_deduction)
    elif jump in ['seat_drop','seat_to_feet','half_to_seat','half_to_feet']:
        leg_deduction = 0
        legs_together_deduction = 0
        toes_deduction = 0
        start_idx = int(len(features) * 0.2)
        end_idx   = int(len(features) * 0.8)
        for f in features[start_idx:end_idx]:
            legs = f["right_knee_angle"]
            toes1 = f["right_pointed_toes"]
            toes2 = f["left_pointed_toes"]
            if not f["legs_together"]:
                legs_together_deduction = 0.1
                reason.append('Legs Apart')
            if legs <= 140:
                leg_deduction = 0.1
                reason.append('Legs Bent')
            if toes1 < 30 and toes2 < 30:
                toes_deduction = 0.1
                reason.append('Toes Flexed')
        deduction = min(0.5, toes_deduction + leg_deduction + legs_together_deduction)
    elif jump in ['half_turn', 'full_turn']:
        leg_deduction = 0
        legs_together_deduction = 0
        toes_deduction = 0
        trunk_penalty = 0
        start_idx = int(len(features) * 0.2)
        end_idx   = int(len(features) * 0.8)
        for f in features[start_idx:end_idx]:
            legs = f["right_knee_angle"]
            toes1 = f["right_pointed_toes"]
            toes2 = f["left_pointed_toes"]
            if not f["legs_together"]:
                legs_together_deduction = 0.1
                reason.append('Legs Apart')
            if legs <= 140:
                leg_deduction = 0.1
                reason.append('Legs Bent')
            if toes1 < 30 and toes2 < 30:
                toes_deduction = 0.1
                reason.append('Toes Flexed')
            if f["trunk_angle"] > 195 or f['trunk_angle'] < 165:
                trunk_penalty = 0.1
                reason.append('Off Balance')
        deduction = min(0.5, toes_deduction + leg_deduction + legs_together_deduction + trunk_penalty)
    return deduction, reason

# ------------------------------------------------------------------
# Preprocessing for tmeporal splitting
# ------------------------------------------------------------------

def evaluate_segments(segments, fps):
    results = []
    for i, seg_frames in enumerate(segments):
        jump, video_features, video_angles = predict_video_in_memory(seg_frames, fps)
        deduction, reason = manual_execution(jump, video_angles)
        results.append((jump, deduction, reason, video_angles))
    return results

def clip_outliers(signal, threshold):
    clipped = signal.copy()
    for i in range(len(clipped)):
        if abs(clipped[i]) > threshold:
            clipped[i] = clipped[i-1] if i > 0 else (threshold if clipped[i] > 0 else -threshold)
    return clipped

def dynamic_center(frame, baseline_torso, min_weight=0.3):
    ls = get_landmark(frame, 11)[1]
    rs = get_landmark(frame, 12)[1]
    avg_shoulder = (ls + rs) / 2
    lh = get_landmark(frame, 23)[1]
    rh = get_landmark(frame, 24)[1]
    avg_hip = (lh + rh) / 2
    torso = abs(avg_shoulder - avg_hip)
    w = max(min_weight, torso / baseline_torso)
    center = w * avg_shoulder + (1 - w) * avg_hip
    return center, avg_hip, torso

def build_submission_string_for_gymnast(gymnast_id):
    submissions = db_manager.get_submissions_by_gymnast(gymnast_id)
    history_data = []
    for s in submissions:
        skills = db_manager.get_skill_scores_by_submission(s["submission_id"])
        ded_details = db_manager.get_deduction_details_by_submission(s["submission_id"])
        history_line = f"Submission {s['submission_id']} on {s['submission_date']}: Overall Score = {s['overall_score']}\n"
        if skills:
            history_line += "  Skill Scores:\n" + "\n".join(
                [f"    Skill {skill['skill_No']} - {skill['skill_Name']}: Deduction = {skill['deduction_value']}" for skill in skills]
            )
        if ded_details:
            ded_group = {}
            for d in ded_details:
                skill_no = d["skill_No"]
                reason = d["deduction_reason"]
                if skill_no in ded_group:
                    ded_group[skill_no].append(reason)
                else:
                    ded_group[skill_no] = [reason]
            ded_lines = []
            for skill_no, reasons in ded_group.items():
                ded_lines.append(f"    Skill {skill_no}: " + ", ".join(reasons))
            history_line += "\n  Deduction Details:\n" + "\n".join(ded_lines)
        history_data.append(history_line)
    return "\n\n".join(history_data) if history_data else "No submission history found."

def build_latest_submission_string_for_gymnast(gymnast_id):
    submission = db_manager.get_latest_submission_by_gymnast(gymnast_id)
    if not submission:
        return "No submission history found."
    skills = db_manager.get_skill_scores_by_submission(submission["submission_id"])
    ded_details = db_manager.get_deduction_details_by_submission(submission["submission_id"])
    history_line = (f"Submission {submission['submission_id']} on {submission['submission_date']}: "
                    f"Overall Score = {submission['overall_score']}\n")
    if skills:
        history_line += "  Skill Scores:\n" + "\n".join(
            [f"    Skill {skill['skill_No']} - {skill['skill_Name']}: Deduction = {skill['deduction_value']}"
             for skill in skills]
        )
    if ded_details:
        ded_group = {}
        for d in ded_details:
            skill_no = d["skill_No"]
            reason = d["deduction_reason"]
            if skill_no in ded_group:
                ded_group[skill_no].append(reason)
            else:
                ded_group[skill_no] = [reason]
        ded_lines = []
        for skill_no, reasons in ded_group.items():
            ded_lines.append(f"    Skill {skill_no}: " + ", ".join(reasons))
        history_line += "\n  Deduction Details:\n" + "\n".join(ded_lines)
    return history_line

# ------------------------------------------------------------------
# Temporal action localisation 
# ------------------------------------------------------------------

def temporal(file, fps=30, min_bounce_gap=20, min_weight=0.2):
    min_bounce_gap_seconds = 1.0
    min_bounce_gap = int(fps * min_bounce_gap_seconds)

    video_features = convert_vid(file)
    if not video_features:
        print("No video features found!")
        return []
    torso_lengths = []
    for frame in video_features:
        ls = get_landmark(frame, 11)[1]
        rs = get_landmark(frame, 12)[1]
        avg_shoulder = (ls + rs) / 2
        lh = get_landmark(frame, 23)[1]
        rh = get_landmark(frame, 24)[1]
        avg_hip = (lh + rh) / 2
        torso_lengths.append(abs(avg_shoulder - avg_hip))
    baseline_torso = np.max(torso_lengths)
    print(f"Baseline torso length: {baseline_torso:.2f}")
    all_hip_positions = []
    for frame in video_features:
        lh = get_landmark(frame, 23)[1]
        rh = get_landmark(frame, 24)[1]
        avg_hip = (lh + rh) / 2
        all_hip_positions.append(avg_hip)
    max_hip = np.percentile(all_hip_positions, 95)
    print(f"Computed landing hip level (95th percentile): {max_hip:.2f}")
    hip_position_threshold = 0.80 * max_hip
    delta = 3
    dynamic_centers = []
    hip_positions = []
    for frame in video_features:
        center, hip, torso = dynamic_center(frame, baseline_torso, min_weight)
        dynamic_centers.append(center)
        hip_positions.append(hip)
    weighted_velocity = []
    for i in range(delta, len(dynamic_centers) - delta):
        vel_weighted = dynamic_centers[i+delta] - dynamic_centers[i-delta]
        weighted_velocity.append(vel_weighted)
    velocity_threshold = 1.0
    kernel = np.ones(3) / 3
    weighted_velocity = np.convolve(weighted_velocity, kernel, mode='same')
    weighted_velocity = clip_outliers(weighted_velocity, velocity_threshold)
    bounces = [0]
    bounce_details = []
    for idx in range(1, len(weighted_velocity) - 1):
        if weighted_velocity[idx] > 0 and weighted_velocity[idx+1] <= 0:
            denom = weighted_velocity[idx] - weighted_velocity[idx+1]
            fraction = weighted_velocity[idx] / denom if denom != 0 else 0.5
            estimated_bounce = idx + fraction
            bounce_frame = int(round(estimated_bounce))
            center, avg_hip, current_torso = dynamic_center(video_features[idx], baseline_torso, min_weight)
            if avg_hip < hip_position_threshold:
                print(f"Frame {idx}: Rejected because hip position ({avg_hip:.2f}) is below threshold (< {hip_position_threshold:.2f}).")
                continue
            print(f"Frame {idx}: Weighted vel = {weighted_velocity[idx]:.2f}, Estimated bounce = {estimated_bounce:.2f} -> Bounce frame {bounce_frame}, Avg hip = {avg_hip:.2f}, Torso = {current_torso:.2f}")
            if bounces and (idx - bounces[-1]) < min_bounce_gap:
                continue
            bounces.append(bounce_frame)
            detail = {
                'frame': bounce_frame,
                'weighted_velocity': weighted_velocity[idx],
                'avg_hip': avg_hip
            }
            bounce_details.append(detail)
    bounces.append(len(video_features) - 1)
    print("Final bounce frame indices:", bounces)
    cap = cv2.VideoCapture(file)
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    segments = []
    for i in range(len(bounces) - 1):
        start, end = bounces[i], bounces[i+1]
        segments.append(all_frames[start:end])
    for idx, segment in enumerate(segments):
        if idx == 0:
            reason_str = "Initial segment (no bounce detection)."
        else:
            detail = bounce_details[idx-1]
            reason_components = [
                f"Momentum: {detail['weighted_velocity']:.2f}",
                f"Avg hip: {detail['avg_hip']:.2f}"
            ]
            reason_str = "; ".join(reason_components)
        print(f"Playing segment {idx+1}/{len(segments)}. Bounce reason: {reason_str}. Press 'q' to skip to next segment.")
        for frame in segment:
            cv2.imshow("Segment Playback", frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return segments

# ------------------------------------------------------------------
# PyQt5 GUI
# ------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trampolining Judging System")
        self.setGeometry(200, 200, 800, 600)
        self.current_user = None
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        self.login_page = LoginPage(self)
        self.register_page = RegisterPage(self)
        self.home_page = HomePage(self)
        self.upload_page = UploadPage(self)
        self.display_score_page = DisplayScorePage(self)
        self.data_page = DataPage(self)
        self.profile_page = ProfilePage(self)
        self.gymnast_search_page = GymnastSearchPage(self)
        self.invitations_page = InvitationsPage(self)
        self.stacked_widget.addWidget(self.login_page)
        self.stacked_widget.addWidget(self.register_page)
        self.stacked_widget.addWidget(self.home_page)
        self.stacked_widget.addWidget(self.upload_page)
        self.stacked_widget.addWidget(self.display_score_page)
        self.stacked_widget.addWidget(self.data_page)
        self.stacked_widget.addWidget(self.profile_page)
        self.stacked_widget.addWidget(self.gymnast_search_page)
        self.stacked_widget.addWidget(self.invitations_page)
        self.create_menu()
        self.stacked_widget.setCurrentWidget(self.login_page)

    def create_menu(self):
        menu_bar = self.menuBar()
        options_menu = menu_bar.addMenu("Options")
        logout_action = QAction("Logout", self)
        logout_action.triggered.connect(self.show_login_page)
        options_menu.addAction(logout_action)

    def show_login_page(self):
        self.stacked_widget.setCurrentWidget(self.login_page)
    def show_register_page(self):
        self.stacked_widget.setCurrentWidget(self.register_page)
    def show_home_page(self):
        self.stacked_widget.setCurrentWidget(self.home_page)
    def show_upload_page(self):
        self.stacked_widget.setCurrentWidget(self.upload_page)
    def show_display_score_page(self):
        self.stacked_widget.setCurrentWidget(self.display_score_page)
    def show_data_page(self):
        self.stacked_widget.setCurrentWidget(self.data_page)
    def show_profile_page(self):
        self.stacked_widget.setCurrentWidget(self.profile_page)
    def show_gymnast_search_page(self):
        self.stacked_widget.setCurrentWidget(self.gymnast_search_page)
    def show_invitations_page(self):
        self.stacked_widget.setCurrentWidget(self.invitations_page)

# ---------------------------
# Login Page
# ---------------------------
class LoginPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setContentsMargins(0, 0, 0, 0)
        container = QWidget()
        container.setObjectName("LoginContainer")
        container.setFixedWidth(400)
        container_layout = QVBoxLayout()
        container_layout.setSpacing(15)
        container_layout.setContentsMargins(30, 30, 30, 30)
        title_label = QLabel("Login")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("Username")
        self.username_edit.setFixedWidth(350)
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Password")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setFixedWidth(350)
        login_button = QPushButton("Login")
        login_button.setFixedSize(200, 45)
        login_button.clicked.connect(self.handle_login)
        create_account_button = QPushButton("Create Account")
        create_account_button.setFixedSize(200, 45)
        create_account_button.clicked.connect(self.go_to_register)
        container_layout.addWidget(title_label, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.username_edit, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.password_edit, alignment=Qt.AlignCenter)
        container_layout.addWidget(login_button, alignment=Qt.AlignCenter)
        container_layout.addWidget(create_account_button, alignment=Qt.AlignCenter)
        container.setLayout(container_layout)
        main_layout.addWidget(container)
        self.setLayout(main_layout)

    def handle_login(self):
        username = self.username_edit.text().strip()
        password = self.password_edit.text().strip()
        user = db_manager.get_user(username)
        if user and user["password"] == password:
            self.parent_window.current_user = user
            QMessageBox.information(self, "Login Success", f"You are now logged in as a {user['account_type']}.")
            self.parent_window.show_home_page()
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid username or password.")

    def go_to_register(self):
        self.parent_window.show_register_page()

# ---------------------------
# Register Page
# ---------------------------
class RegisterPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setContentsMargins(0, 0, 0, 0)
        container = QWidget()
        container.setObjectName("RegisterContainer")
        container.setFixedWidth(400)
        container_layout = QVBoxLayout()
        container_layout.setSpacing(15)
        container_layout.setContentsMargins(30, 30, 30, 30)
        title_label = QLabel("Create Account")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("Choose a username")
        self.username_edit.setFixedWidth(350)
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Choose a password")
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setFixedWidth(350)
        self.confirm_edit = QLineEdit()
        self.confirm_edit.setPlaceholderText("Confirm password")
        self.confirm_edit.setEchoMode(QLineEdit.Password)
        self.confirm_edit.setFixedWidth(350)
        account_type_label = QLabel("Account Type:")
        account_type_label.setFont(QFont("Arial", 14))
        account_type_label.setAlignment(Qt.AlignCenter)
        self.account_type_combo = QComboBox()
        self.account_type_combo.addItems(["Gymnast", "Coach"])
        self.account_type_combo.setFixedWidth(350)
        register_button = QPushButton("Register")
        register_button.setFixedSize(200, 45)
        register_button.clicked.connect(self.handle_register)
        back_button = QPushButton("Back to Login")
        back_button.setFixedSize(200, 45)
        back_button.clicked.connect(self.go_to_login)
        container_layout.addWidget(title_label, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.username_edit, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.password_edit, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.confirm_edit, alignment=Qt.AlignCenter)
        container_layout.addWidget(account_type_label, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.account_type_combo, alignment=Qt.AlignCenter)
        container_layout.addWidget(register_button, alignment=Qt.AlignCenter)
        container_layout.addWidget(back_button, alignment=Qt.AlignCenter)
        container.setLayout(container_layout)
        main_layout.addWidget(container)
        self.setLayout(main_layout)

    def handle_register(self):
        username = self.username_edit.text().strip()
        password = self.password_edit.text().strip()
        confirm = self.confirm_edit.text().strip()
        account_type = self.account_type_combo.currentText()
        if not username or not password:
            QMessageBox.warning(self, "Error", "Please fill in all fields.")
            return
        if password != confirm:
            QMessageBox.warning(self, "Error", "Passwords do not match.")
            return
        if db_manager.get_user(username):
            QMessageBox.warning(self, "Error", "Username already exists.")
            return
        db_manager.add_user(username, password, account_type)
        QMessageBox.information(self, "Success", f"Account created as a {account_type}. You can now log in.")
        self.parent_window.show_login_page()

    def go_to_login(self):
        self.parent_window.show_login_page()

# ---------------------------
# Home Page
# ---------------------------
class HomePage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setContentsMargins(0, 0, 0, 0)
        container = QWidget()
        container.setObjectName("HomeContainer")
        container.setFixedWidth(400)
        container_layout = QVBoxLayout()
        container_layout.setSpacing(15)
        container_layout.setContentsMargins(30, 30, 30, 30)
        welcome_label = QLabel("Welcome to the Application")
        welcome_label.setFont(QFont("Arial", 20, QFont.Bold))
        welcome_label.setAlignment(Qt.AlignCenter)
        upload_button = QPushButton("Upload Video")
        upload_button.setFixedSize(200, 45)
        upload_button.clicked.connect(self.parent_window.show_upload_page)
        data_button = QPushButton("View Data")
        data_button.setFixedSize(200, 45)
        data_button.clicked.connect(self.parent_window.show_data_page)
        profile_button = QPushButton("Profile")
        profile_button.setFixedSize(200, 45)
        profile_button.clicked.connect(self.parent_window.show_profile_page)
        logout_button = QPushButton("Logout")
        logout_button.setFixedSize(200, 45)
        logout_button.clicked.connect(self.parent_window.show_login_page)
        container_layout.addWidget(welcome_label, alignment=Qt.AlignCenter)
        container_layout.addWidget(upload_button, alignment=Qt.AlignCenter)
        container_layout.addWidget(data_button, alignment=Qt.AlignCenter)
        container_layout.addWidget(profile_button, alignment=Qt.AlignCenter)
        container_layout.addWidget(logout_button, alignment=Qt.AlignCenter)
        container.setLayout(container_layout)
        main_layout.addWidget(container)
        self.setLayout(main_layout)

# ---------------------------
# Data Page
# ---------------------------
        
class DataPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent

        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
       
        container = QWidget()
        container.setObjectName("DataContainer")
        container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(15)
        container_layout.setContentsMargins(30, 30, 30, 30)
        
       
        self.gymnast_selector = QComboBox()
        self.gymnast_selector.setFixedWidth(350)
        
        self.gymnast_selector.currentIndexChanged.connect(self.refresh_data)
        
        self.gymnast_selector.hide()
        
  
        self.tab_widget = QTabWidget()
        self.overall_tab = QWidget()
        self.skill_tab = QWidget()
        self.deduction_tab = QWidget()
        self.history_tab = QWidget()
        
        self.tab_widget.addTab(self.overall_tab, "Overall Performance")
        self.tab_widget.addTab(self.skill_tab, "Skill Performance")
        self.tab_widget.addTab(self.deduction_tab, "Deduction Analysis")
        self.tab_widget.addTab(self.history_tab, "Submission History")
        
        self.create_overall_tab()
        self.create_skill_tab()
        self.create_deduction_tab()
        self.create_history_tab()
        
        self.back_button = QPushButton("Back to Home")
        self.back_button.setFixedSize(200, 45)
        self.back_button.clicked.connect(self.parent_window.show_home_page)
        
       
        container_layout.addWidget(self.gymnast_selector, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.tab_widget)
        container_layout.addWidget(self.back_button, alignment=Qt.AlignCenter)
        
        container.setLayout(container_layout)
        main_layout.addWidget(container)
        self.setLayout(main_layout)

    def create_overall_tab(self):
        self.overall_layout = QVBoxLayout(self.overall_tab)

    def create_skill_tab(self):
        self.skill_layout = QVBoxLayout(self.skill_tab)

    def create_deduction_tab(self):
        self.deduction_layout = QVBoxLayout(self.deduction_tab)

    def create_history_tab(self):
        self.history_layout = QVBoxLayout(self.history_tab)
        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_content = QWidget()
        self.history_content_layout = QVBoxLayout(self.history_content)
        self.history_scroll.setWidget(self.history_content)
        self.history_layout.addWidget(self.history_scroll)

    def populate_gymnast_selector(self):
        
        self.gymnast_selector.clear()
        coach_id = self.parent_window.current_user["id"]

        self.gymnast_selector.addItem("My Data", coach_id)

        gymnast_ids = db_manager.get_gymnasts_for_coach(coach_id)
        for gid in gymnast_ids:
            gymnast = db_manager.get_user_by_id(gid)
            if gymnast:
                self.gymnast_selector.addItem(f"Gymnast: {gymnast['username']}", gymnast["id"])

    def refresh_data(self):

        if self.parent_window.current_user["account_type"] == "Coach":
       
            gymnast_id = self.gymnast_selector.currentData() or self.parent_window.current_user["id"]
        else:
            gymnast_id = self.parent_window.current_user["id"]

        submissions = db_manager.get_submissions_by_gymnast(gymnast_id)
        
  
        if self.overall_layout.count() > 0:
            widget = self.overall_layout.itemAt(0).widget()
            if widget:
                widget.setParent(None)
        if submissions:
            submission_nums = list(range(1, len(submissions) + 1))
            overall_scores = [float(s["overall_score"]) for s in submissions]
            fig_overall = Figure(figsize=(5, 4))
            ax_overall = fig_overall.add_subplot(111)
            ax_overall.plot(submission_nums, overall_scores, marker='o', linestyle='-')
            ax_overall.set_title("Overall Execution Score Trend")
            ax_overall.set_xlabel("Submission #")
            ax_overall.set_ylabel("Score")
            fig_overall.tight_layout() 
            canvas_overall = FigureCanvas(fig_overall)
            self.overall_layout.addWidget(canvas_overall)
        else:
            overall_label = QLabel("No submissions found.")
            overall_label.setWordWrap(True)
            self.overall_layout.addWidget(overall_label)
        
      
        skill_aggregate = {}
        for s in submissions:
            skills = db_manager.get_skill_scores_by_submission(s["submission_id"])
            for skill in skills:
                name = skill["skill_Name"]
                val = float(skill["deduction_value"])
                skill_aggregate.setdefault(name, []).append(val)
        if self.skill_layout.count() > 0:
            widget = self.skill_layout.itemAt(0).widget()
            if widget:
                widget.setParent(None)
        if skill_aggregate:
            fig_skill = Figure(figsize=(5, 4))
            canvas_skill = FigureCanvas(fig_skill)
            ax_skill = fig_skill.add_subplot(111)
            skills = list(skill_aggregate.keys())
            avg_values = [np.mean(skill_aggregate[s]) for s in skills]
            ax_skill.bar(skills, avg_values)
            ax_skill.set_title("Average Skill Deduction")
            ax_skill.set_ylabel("Average Deduction")
            ax_skill.tick_params(axis='x', labelrotation=45)
            fig_skill.tight_layout()
            self.skill_layout.addWidget(canvas_skill)
        else:
            skill_label = QLabel("No skill scores found.")
            skill_label.setWordWrap(True)
            self.skill_layout.addWidget(skill_label)
        
        
        ded_counts = {}
        for s in submissions:
            details = db_manager.get_deduction_details_by_submission(s["submission_id"])
            for d in details:
                reason = d["deduction_reason"]
                if reason:
                    ded_counts[reason] = ded_counts.get(reason, 0) + 1
        if self.deduction_layout.count() > 0:
            widget = self.deduction_layout.itemAt(0).widget()
            if widget:
                widget.setParent(None)
        if ded_counts:
            fig_ded = Figure(figsize=(5, 4))
            canvas_ded = FigureCanvas(fig_ded)
            ax_ded = fig_ded.add_subplot(111)
            reasons = list(ded_counts.keys())
            counts = [ded_counts[r] for r in reasons]
            ax_ded.pie(counts, labels=reasons, autopct="%1.1f%%", startangle=140)
            ax_ded.set_title("Deduction Reasons Breakdown")
            self.deduction_layout.addWidget(canvas_ded)
        else:
            ded_label = QLabel("No deduction details found.")
            ded_label.setWordWrap(True)
            self.deduction_layout.addWidget(ded_label)
        
       
        history_data = []
        for s in submissions:
            skills = db_manager.get_skill_scores_by_submission(s["submission_id"])
            ded_details = db_manager.get_deduction_details_by_submission(s["submission_id"])
            history_line = f"Submission {s['submission_id']} on {s['submission_date']}: Overall Score = {s['overall_score']}\n"
            if skills:
                history_line += "  Skill Scores:\n" + "\n".join(
                    [f"    Skill {skill['skill_No']} - {skill['skill_Name']}: Deduction = {skill['deduction_value']}" for skill in skills]
                )
            if ded_details:
                ded_group = {}
                for d in ded_details:
                    ded_group.setdefault(d["skill_No"], []).append(d["deduction_reason"])
                ded_lines = [f"    Skill {no}: " + ", ".join(reasons) for no, reasons in ded_group.items()]
                history_line += "\n  Deduction Details:\n" + "\n".join(ded_lines)
            history_data.append(history_line)
        history_text = "\n\n".join(history_data) if history_data else "No submission history found."
        for i in reversed(range(self.history_content_layout.count())):
            widget = self.history_content_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        history_label = QLabel(history_text)
        history_label.setWordWrap(True)
        self.history_content_layout.addWidget(history_label)

    def showEvent(self, event):
        super().showEvent(event)
        
        if self.parent_window.current_user["account_type"] == "Coach":
            self.gymnast_selector.show()
            self.populate_gymnast_selector()
        else:
            self.gymnast_selector.hide()
        self.refresh_data()

# ---------------------------
# Profile Page
# ---------------------------

class ProfilePage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent


        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setContentsMargins(0, 0, 0, 0)


        container = QWidget()
        container.setObjectName("ProfileContainer")
        container.setFixedWidth(400)
        container_layout = QVBoxLayout()
        container_layout.setSpacing(15)
        container_layout.setContentsMargins(30, 30, 30, 30)

    
        self.profile_label = QLabel("Your Profile")
        self.profile_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.profile_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self.profile_label, alignment=Qt.AlignCenter)

       
        self.coach_section = QWidget()
        coach_layout = QVBoxLayout(self.coach_section)
        coach_layout.setSpacing(5)  
        
       
        self.gymnast_list_label = QLabel("My Gymnasts")
        self.gymnast_list_label.setAlignment(Qt.AlignCenter)
        coach_layout.addWidget(self.gymnast_list_label)
        
       
        self.gymnast_list_widget = QListWidget()
        self.gymnast_list_widget.setFixedHeight(150)
        coach_layout.addWidget(self.gymnast_list_widget)
        
        
        btn_layout = QHBoxLayout()
        self.remove_gymnast_btn = QPushButton("Remove Selected Gymnast")
        self.remove_gymnast_btn.clicked.connect(self.remove_selected_gymnast)
        btn_layout.addWidget(self.remove_gymnast_btn, alignment=Qt.AlignCenter)
        self.add_gymnast_btn = QPushButton("Add Gymnast")
        self.add_gymnast_btn.clicked.connect(lambda: self.parent_window.show_gymnast_search_page())
        btn_layout.addWidget(self.add_gymnast_btn, alignment=Qt.AlignCenter)
        coach_layout.addLayout(btn_layout)
        
        
        self.coach_pass_label = QLabel("Change Password")
        self.coach_pass_label.setAlignment(Qt.AlignCenter)
        coach_layout.addWidget(self.coach_pass_label)
        self.coach_new_password_edit = QLineEdit()
        self.coach_new_password_edit.setPlaceholderText("New Password")
        self.coach_new_password_edit.setEchoMode(QLineEdit.Password)
        coach_layout.addWidget(self.coach_new_password_edit)
        self.coach_confirm_password_edit = QLineEdit()
        self.coach_confirm_password_edit.setPlaceholderText("Confirm New Password")
        self.coach_confirm_password_edit.setEchoMode(QLineEdit.Password)
        coach_layout.addWidget(self.coach_confirm_password_edit)
        self.coach_change_password_btn = QPushButton("Change Password")
        self.coach_change_password_btn.clicked.connect(self.change_password)
        coach_layout.addWidget(self.coach_change_password_btn, alignment=Qt.AlignCenter)

        
        self.gymnast_section = QWidget()
        gymnast_layout = QVBoxLayout(self.gymnast_section)
        gymnast_layout.setAlignment(Qt.AlignCenter)
        
        self.username_label = QLabel("Username: ")
        gymnast_layout.addWidget(self.username_label, alignment=Qt.AlignCenter)
        
        fixed_width = 200
        self.gymnast_new_password_edit = QLineEdit()
        self.gymnast_new_password_edit.setPlaceholderText("New Password")
        self.gymnast_new_password_edit.setEchoMode(QLineEdit.Password)
        self.gymnast_new_password_edit.setFixedWidth(fixed_width)
        gymnast_layout.addWidget(self.gymnast_new_password_edit, alignment=Qt.AlignCenter)
        
        self.gymnast_confirm_password_edit = QLineEdit()
        self.gymnast_confirm_password_edit.setPlaceholderText("Confirm New Password")
        self.gymnast_confirm_password_edit.setEchoMode(QLineEdit.Password)
        self.gymnast_confirm_password_edit.setFixedWidth(fixed_width)
        gymnast_layout.addWidget(self.gymnast_confirm_password_edit, alignment=Qt.AlignCenter)
        
        self.gymnast_change_password_btn = QPushButton("Change Password")
        self.gymnast_change_password_btn.setFixedSize(fixed_width, 45)
        self.gymnast_change_password_btn.clicked.connect(self.change_password)
        gymnast_layout.addWidget(self.gymnast_change_password_btn, alignment=Qt.AlignCenter)
        
        self.coach_list_label = QLabel("Assigned Coach(es):")
        gymnast_layout.addWidget(self.coach_list_label, alignment=Qt.AlignCenter)
        
        self.coach_list_display = QLabel()
        self.coach_list_display.setWordWrap(True)
        gymnast_layout.addWidget(self.coach_list_display, alignment=Qt.AlignCenter)
        
        self.view_invitations_btn = QPushButton("View Group Invitations")
        self.view_invitations_btn.setFixedSize(fixed_width, 45)
        self.view_invitations_btn.clicked.connect(lambda: self.parent_window.show_invitations_page())
        gymnast_layout.addWidget(self.view_invitations_btn, alignment=Qt.AlignCenter)
        
    
        self.back_button = QPushButton("Back to Home")
        self.back_button.setFixedSize(200, 45)
        self.back_button.clicked.connect(self.parent_window.show_home_page)
        
        container_layout.addWidget(self.coach_section, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.gymnast_section, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.back_button, alignment=Qt.AlignCenter)
        
        container.setLayout(container_layout)
        main_layout.addWidget(container)
        self.setLayout(main_layout)

    def get_account_type(self):
        if self.parent_window.current_user:
            return self.parent_window.current_user.get("account_type", "Gymnast")
        return "Gymnast"

   
    def refresh_gymnast_list(self):
        coach_id = self.parent_window.current_user["id"]
        gymnast_ids = db_manager.get_gymnasts_for_coach(coach_id)
        self.gymnast_list_widget.clear()
        if not gymnast_ids:
            item = QListWidgetItem("No gymnasts currently linked.")
            item.setFlags(Qt.NoItemFlags)
            self.gymnast_list_widget.addItem(item)
        else:
            for gid in gymnast_ids:
                gymnast = db_manager.get_user_by_id(gid)
                username = gymnast["username"] if gymnast else f"ID {gid} (Not Found)"
                item = QListWidgetItem(username)
                item.setData(Qt.UserRole, gid)
                self.gymnast_list_widget.addItem(item)

    def remove_selected_gymnast(self):
        selected_item = self.gymnast_list_widget.currentItem()
        if not selected_item or not (selected_item.flags() & Qt.ItemIsSelectable):
            QMessageBox.warning(self, "No Selection", "Please select a gymnast to remove.")
            return
        gymnast_id = selected_item.data(Qt.UserRole)
        coach_id = self.parent_window.current_user["id"]
        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to remove {selected_item.text()} from your team?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            db_manager.remove_gymnast_from_coach(coach_id, gymnast_id)
            QMessageBox.information(self, "Removed", f"{selected_item.text()} has been removed from your team.")
            self.refresh_gymnast_list()

   
    def refresh_gymnast_profile(self):
        username = self.parent_window.current_user.get("username", "Unknown")
        self.username_label.setText(f"Username: {username}")
        gymnast_id = self.parent_window.current_user["id"]
        coach_ids = db_manager.get_coaches_for_gymnast(gymnast_id)
        if not coach_ids:
            self.coach_list_display.setText("No coach assigned.")
        else:
            names = []
            for cid in coach_ids:
                coach = db_manager.get_user_by_id(cid)
                names.append(coach["username"] if coach else f"ID {cid}")
            self.coach_list_display.setText("\n".join(names))

   
    def change_password(self):
        if self.get_account_type() == "Coach":
            new_password = self.coach_new_password_edit.text().strip()
            confirm_password = self.coach_confirm_password_edit.text().strip()
        else:
            new_password = self.gymnast_new_password_edit.text().strip()
            confirm_password = self.gymnast_confirm_password_edit.text().strip()

        if not new_password:
            QMessageBox.warning(self, "Error", "Please enter a new password.")
            return
        if new_password != confirm_password:
            QMessageBox.warning(self, "Error", "Passwords do not match.")
            return

        user_id = self.parent_window.current_user["id"]
        db_manager.update_user_password(user_id, new_password)
        self.parent_window.current_user["password"] = new_password
        QMessageBox.information(self, "Success", "Password changed successfully.")

        if self.get_account_type() == "Coach":
            self.coach_new_password_edit.clear()
            self.coach_confirm_password_edit.clear()
        else:
            self.gymnast_new_password_edit.clear()
            self.gymnast_confirm_password_edit.clear()

    def showEvent(self, event):
        super().showEvent(event)
        if self.get_account_type() == "Coach":
            self.coach_section.show()
            self.gymnast_section.hide()
            self.refresh_gymnast_list()
        else:
            self.coach_section.hide()
            self.gymnast_section.show()
            self.refresh_gymnast_profile()
        

# ---------------------------
# GymnastSearchPage
# ---------------------------
class GymnastSearchPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setContentsMargins(0, 0, 0, 0)
        container = QWidget()
        container.setObjectName("GymnastSearchContainer")
        container.setFixedWidth(400)
        container_layout = QVBoxLayout()
        container_layout.setSpacing(10)
        container_layout.setContentsMargins(30, 30, 30, 30)
        self.title_label = QLabel("Search Gymnasts")
        self.title_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Enter gymnast username")
        self.search_edit.setFixedWidth(350)
        self.search_edit.textChanged.connect(self.update_list)
        self.list_widget = QListWidget()
        self.list_widget.setFixedWidth(350)
        self.invite_btn = QPushButton("Send Invitation")
        self.invite_btn.setFixedSize(200, 45)
        self.invite_btn.clicked.connect(self.send_invitation)
        self.back_btn = QPushButton("Back to Profile")
        self.back_btn.setFixedSize(200, 45)
        self.back_btn.clicked.connect(lambda: self.parent_window.show_profile_page())
        container_layout.addWidget(self.title_label, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.search_edit, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.list_widget, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.invite_btn, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.back_btn, alignment=Qt.AlignCenter)
        container.setLayout(container_layout)
        main_layout.addWidget(container)
        self.setLayout(main_layout)
        self.all_gymnasts = []
        self.load_gymnasts()

    def load_gymnasts(self):
        self.all_gymnasts = db_manager.get_all_gymnasts()
        self.update_list()

    def update_list(self):
        search_text = self.search_edit.text().lower()
        self.list_widget.clear()
        for gymnast in self.all_gymnasts:
            if search_text in gymnast["username"].lower():
                item = QListWidgetItem(gymnast["username"])
                item.setData(Qt.UserRole, gymnast["id"])
                self.list_widget.addItem(item)

    def send_invitation(self):
        selected_item = self.list_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "No Selection", "Please select a gymnast from the list.")
            return
        gymnast_id = selected_item.data(Qt.UserRole)
        coach_id = self.parent_window.current_user["id"]
        db_manager.send_invitation(coach_id, gymnast_id)
        QMessageBox.information(self, "Invitation Sent", f"An invitation has been sent to {selected_item.text()}.")

    def showEvent(self, event):
        super().showEvent(event)
        self.load_gymnasts()

# ---------------------------
# InvitationsPage 
# ---------------------------
class InvitationsPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent_window = parent
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setContentsMargins(0, 0, 0, 0)
        container = QWidget()
        container.setObjectName("InvitationsContainer")
        container.setFixedWidth(400)
        container_layout = QVBoxLayout()
        container_layout.setSpacing(10)
        container_layout.setContentsMargins(30, 30, 30, 30)
        self.title_label = QLabel("Group Invitations")
        self.title_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.invite_list = QListWidget()
        self.invite_list.setFixedWidth(350)
        self.back_btn = QPushButton("Back to Profile")
        self.back_btn.setFixedSize(200, 45)
        self.back_btn.clicked.connect(lambda: self.parent_window.show_profile_page())
        container_layout.addWidget(self.title_label, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.invite_list, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.back_btn, alignment=Qt.AlignCenter)
        container.setLayout(container_layout)
        main_layout.addWidget(container)
        self.setLayout(main_layout)
        self.invite_list.itemClicked.connect(self.handle_invitation_click)

    def load_invitations(self):
        self.invite_list.clear()
        if not self.parent_window.current_user:
            QMessageBox.warning(self, "Error", "No user is logged in. Please log in first.")
            return
        gymnast_id = self.parent_window.current_user["id"]
        invitations = db_manager.get_invitations_for_gymnast(gymnast_id)
        for invite in invitations:
            coach_id = invite["coach_id"]
            coach = db_manager.get_user_by_id(coach_id)
            if coach:
                display_name = coach["username"]
            else:
                display_name = f"ID {coach_id} (Not Found)"
            item = QListWidgetItem(f"Coach: {display_name}")
            item.setData(Qt.UserRole, coach_id)
            self.invite_list.addItem(item)

    def handle_invitation_click(self, item):
        coach_id = item.data(Qt.UserRole)
        response = QMessageBox.question(self, "Invitation Response",
                                        f"Do you want to accept the invitation from Coach ID {coach_id}?",
                                        QMessageBox.Yes | QMessageBox.No)
        gymnast_id = self.parent_window.current_user["id"]
        if response == QMessageBox.Yes:
            db_manager.update_invitation_status(coach_id, gymnast_id, "accepted")
            db_manager.add_gymnast_to_coach(coach_id, gymnast_id)
            QMessageBox.information(self, "Invitation Accepted", f"You have accepted the invitation from Coach ID {coach_id}.")
        else:
            db_manager.update_invitation_status(coach_id, gymnast_id, "rejected")
            QMessageBox.information(self, "Invitation Rejected", f"You have rejected the invitation from Coach ID {coach_id}.")
        self.load_invitations()

    def showEvent(self, event):
        super().showEvent(event)
        self.load_invitations()

# ---------------------------
# Upload Page – Fixed Container with Object Name
# ---------------------------

class UploadPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.file_path = None

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setContentsMargins(0, 0, 0, 0)

        container = QWidget()
        container.setObjectName("UploadContainer")
        container.setFixedWidth(400)
        container_layout = QVBoxLayout()
        container_layout.setSpacing(15)
        container_layout.setContentsMargins(30, 30, 30, 30)

       
        self.target_selector = QComboBox()
        self.target_selector.setFixedWidth(350)
 
        self.target_selector.hide()


        single_jump_button = QPushButton("Upload Single Jump")
        full_routine_button = QPushButton("Upload Full Routine")
        single_jump_button.setFixedSize(200, 45)
        full_routine_button.setFixedSize(200, 45)
        single_jump_button.clicked.connect(lambda: self.upload_and_process_video(full_routine=False))
        full_routine_button.clicked.connect(lambda: self.upload_and_process_video(full_routine=True))
        
        back_button = QPushButton("Back to Home")
        back_button.setFixedSize(200, 45)
        back_button.clicked.connect(lambda: self.parent.stacked_widget.setCurrentWidget(self.parent.home_page))

        container_layout.addWidget(self.target_selector, alignment=Qt.AlignCenter)
        container_layout.addWidget(single_jump_button, alignment=Qt.AlignCenter)
        container_layout.addWidget(full_routine_button, alignment=Qt.AlignCenter)
        container_layout.addWidget(back_button, alignment=Qt.AlignCenter)
        container.setLayout(container_layout)
        main_layout.addWidget(container)
        self.setLayout(main_layout)

    def populate_target_selector(self):
        """Populate the selector with options for coaches."""
        self.target_selector.clear()
        coach_id = self.parent.current_user["id"]
        
        self.target_selector.addItem("My Video", coach_id)
        
        gymnast_ids = db_manager.get_gymnasts_for_coach(coach_id)
        for gid in gymnast_ids:
            gymnast = db_manager.get_user_by_id(gid)
            if gymnast:
                self.target_selector.addItem(f"Gymnast: {gymnast['username']}", gymnast["id"])

    def upload_and_process_video(self, full_routine):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
            options=options
        )
        if file_path:
            self.file_path = file_path
            self.process_video(full_routine)
        else:
            print("No video selected.")

    def process_video(self, full_routine):
        segments = temporal(self.file_path)
        results = evaluate_segments(segments, 30)
        
        if results:
            
            if self.parent.current_user["account_type"] == "Coach":
                target_id = self.target_selector.currentData() or self.parent.current_user["id"]
            else:
                target_id = self.parent.current_user["id"]

            results_text = []
            total_deductions = 0
            skill_records = []         
            deduction_details = []     
            
            for i, (jump, deduction, reasons, angles) in enumerate(results):
                results_text.append(f"Segment {i+1}: Jump={jump}, Deduction={deduction}")
                total_deductions += deduction
                skill_records.append((i+1, jump, deduction))
                for r in set(reasons):
                    deduction_details.append((i+1, r))
                        
            overall_score = (10 - total_deductions) *2

            
            submission_id = db_manager.add_submission(target_id, overall_score)
            db_manager.add_skill_scores(submission_id, skill_records)
            db_manager.add_deduction_details(submission_id, deduction_details)
            
            
            self.parent.display_score_page.set_multiple_results(total_deductions, target_id=target_id)
            self.parent.show_display_score_page()
        else:
            QMessageBox.warning(self, "Error", "No segments created from the video.")


    def showEvent(self, event):
        super().showEvent(event)
        
        if self.parent.current_user["account_type"] == "Coach":
            self.target_selector.show()
            self.populate_target_selector()
        else:
            self.target_selector.hide()

# ---------------------------
# Display Score Page – Fixed Container with Object Name
# ---------------------------

class DisplayScorePage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.target_id = None  
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setContentsMargins(0, 0, 0, 0)
        container = QWidget()
        container.setObjectName("DisplayScoreContainer")
        container.setFixedWidth(400)
        container_layout = QVBoxLayout()
        container_layout.setSpacing(15)
        container_layout.setContentsMargins(30, 30, 30, 30)
        self.total_score_label = QLabel("Total Execution Score: 20.00")
        self.total_score_label.setFont(QFont("Arial", 18, QFont.Bold))
        self.total_score_label.setAlignment(Qt.AlignCenter)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.detail_widget = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_widget)
        self.detail_widget.setLayout(self.detail_layout)
        self.scroll_area.setWidget(self.detail_widget)
        back_button = QPushButton("Back to Home")
        back_button.setFixedSize(200, 45)
        back_button.clicked.connect(lambda: self.parent.stacked_widget.setCurrentWidget(self.parent.home_page))
        container_layout.addWidget(self.total_score_label, alignment=Qt.AlignCenter)
        container_layout.addWidget(self.scroll_area)
        container_layout.addWidget(back_button, alignment=Qt.AlignCenter)
        container.setLayout(container_layout)
        main_layout.addWidget(container)
        self.setLayout(main_layout)

    def set_multiple_results(self, total_deductions, target_id=None):

        self.target_id = target_id if target_id is not None else self.parent.current_user["id"]
        total_score = (10 - total_deductions)*2
        self.total_score_label.setText("Total Execution Score: {:.2f}".format(total_score))
        
        
        detailed_text = build_latest_submission_string_for_gymnast(self.target_id)
        
        for i in reversed(range(self.detail_layout.count())):
            widget = self.detail_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        detail_label = QLabel(detailed_text)
        detail_label.setWordWrap(True)
        self.detail_layout.addWidget(detail_label)


# ------------------------------------------------------------------
# Main Entry – Global Stylesheet & Application Run
# ------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
    QMainWindow {
    background-color: #F5F5F5;  /* Light gray */
}
QWidget {
    font-family: 'Segoe UI', sans-serif;
    font-size: 12pt;
    color: #2C3E50;
}
QPushButton {
    background-color: #4169E1;  /* Royal Blue */
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
}
QPushButton:hover {
    background-color: #27408B;  /* Darker Royal Blue */
}
QLineEdit {
    border: 1px solid #BDC3C7;
    border-radius: 5px;
    padding: 8px;
    background-color: white;
}
QLabel {
    font-size: 14pt;
}
QTabWidget::pane {
    border: 1px solid #BDC3C7;
    border-radius: 5px;
    background-color: white;
}
QTabBar::tab {
    background: #BDC3C7;
    padding: 10px;
    margin: 2px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}
QTabBar::tab:selected {
    background: #4169E1;
    color: white;
}
QScrollArea {
    border: none;
}
/* Global container styles */
QWidget#LoginContainer,
QWidget#RegisterContainer,
QWidget#HomeContainer,
QWidget#DataContainer,
QWidget#ProfileContainer,
QWidget#GymnastSearchContainer,
QWidget#InvitationsContainer,
QWidget#UploadContainer,
QWidget#DisplayScoreContainer {
    background-color: white;
    border: 1px solid #BDC3C7;
    border-radius: 8px;
}


    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
