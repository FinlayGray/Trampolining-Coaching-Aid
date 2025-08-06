import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QPushButton, QVBoxLayout,
    QWidget, QStackedWidget, QFileDialog, QLabel, QSpacerItem, QSizePolicy
)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping  # Import EarlyStopping callback
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight  # For computing class weights
from sklearn.model_selection import KFold  # For k-fold cross validation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib
import os
import pandas as pd  # For creating the testing table

# Path to the model
model_path = '/Users/finlaygray/Documents/diss/pose_landmarker_full.task'

# Importing necessary classes from MediaPipe tasks
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)


num_features_per_frame = None  # Will be set dynamically

# --- Feature Extraction Functions ---

def extract_features_from_frame(pose_landmarker_result):
    global num_features_per_frame 
    features = []
    if pose_landmarker_result.pose_landmarks:

        num_landmarks = len(pose_landmarker_result.pose_landmarks[0])
        if num_features_per_frame is None:
            num_features_per_frame = num_landmarks * 5
        for landmark in pose_landmarker_result.pose_landmarks[0]:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility,
                landmark.presence
            ])
    else:
        # If no landmarks are detected, append zeros if we know the expected length
        if num_features_per_frame is not None:
            features = [0.0] * num_features_per_frame
        else:
            print("No landmarks detected and num_features_per_frame is not set. Skipping frame.")
            return None  # Return None to indicate skipping
    return features

def convert_vid(file_path, invert=False):
    landmarker = PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(file_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second (FPS): {fps}")

    frame_index = 0
    video_features = []  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally if invert is True
        if invert:
            frame = cv2.flip(frame, 1)  # Flip along the y-axis

        frame_timestamp_ms = int((frame_index / fps) * 1000)
        print(f"Timestamp for frame {frame_index}: {frame_timestamp_ms} ms")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        numpy_frame_from_opencv = np.array(frame_rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)

        # Call the pose landmarker detect function
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        print(pose_landmarker_result)

        # Extract features from the pose_landmarker_result
        frame_features = extract_features_from_frame(pose_landmarker_result)
        if frame_features is not None:
            video_features.append(frame_features)

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    return video_features  # Return the collected features for the video

# --- Collect Features and Labels ---

all_videos_features = [] 
video_labels = []         


video_filepaths = []  
video_labels_original = []  # Corresponding labels

directory = 'training_data'

for filename in os.listdir(directory):
    
    if filename in ['tuck_jump', 'seat_drop', 'seat_to_feet', 'straddle_jump', 'full_turn',
                    'tuck_back', 'half_turn', 'pike_jump', 'half_to_seat', 'half_to_feet']:
        file_path1 = os.path.join(directory, filename)
        for file_train in os.listdir(file_path1):
            file_path2 = os.path.join(file_path1, file_train)
            if os.path.isfile(file_path2):
                print(f"Processing file: {file_path2}")
                video_filepaths.append(file_path2)
                video_labels_original.append(filename)

for file_path, label in zip(video_filepaths, video_labels_original):
    print(f"Processing video: {file_path}")
    
    video_features = convert_vid(file_path, invert=False)
    all_videos_features.append(video_features)
    video_labels.append(label)
   
    inverted_video_features = convert_vid(file_path, invert=True)
    all_videos_features.append(inverted_video_features)
    video_labels.append(label)  

# --- Prepare Data for Training ---

# If num_features_per_frame is still None, set it based on the collected data
if num_features_per_frame is None:
    for video in all_videos_features:
        if video:  # Ensure the video is not empty
            num_features_per_frame = len(video[0])
            print(f"Number of features per frame set to: {num_features_per_frame}")
            break
    else:
        raise ValueError("No frames with landmarks detected in any video.")

# Ensure that all frame features have length num_features_per_frame
def pad_frame_features(frame_features, target_length):
    current_length = len(frame_features)
    if current_length < target_length:
        frame_features.extend([0.0] * (target_length - current_length))
    elif current_length > target_length:
        frame_features = frame_features[:target_length]
    return frame_features

for video in all_videos_features:
    for i in range(len(video)):
        video[i] = pad_frame_features(video[i], num_features_per_frame)

# Pad sequences so that all videos have the same number of frames
max_sequence_length = max(len(video) for video in all_videos_features)
print(f"Maximum sequence length: {max_sequence_length}")

X = pad_sequences(
    all_videos_features,
    maxlen=max_sequence_length,
    dtype='float32',
    padding='post'
)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(np.array(video_labels).reshape(-1, 1))

input_shape = (X.shape[1], num_features_per_frame)  # (sequence_length, num_features_per_frame)
print(f"Input shape for the model: {input_shape}")

# --- Define the Model Creation Function ---

def create_sequence_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Masking(mask_value=0.0, input_shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Define EarlyStopping callback to monitor val_accuracy
early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, min_delta=0.001, mode='max', restore_best_weights=True)

# --- k-Fold Cross Validation with Extended Metrics ---

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []  # List to store results for each fold
fold = 1

for train_index, val_index in kf.split(X):
    print(f"\nTraining fold {fold}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    
    y_train_int = np.argmax(y_train, axis=1)
    classes = np.unique(y_train_int)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_int)
    class_weight_dict = {i: weight for i, weight in zip(classes, class_weights)}
    
    
    model = create_sequence_model(input_shape, num_classes=y.shape[1])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_data=(X_val, y_val),
        verbose=1,
        class_weight=class_weight_dict,
        callbacks=[early_stopping]
    )
    
    
    score = model.evaluate(X_val, y_val, verbose=0)
    val_loss = score[0]
    
    
    y_val_pred_prob = model.predict(X_val)
    y_val_pred_class = np.argmax(y_val_pred_prob, axis=1)
    y_val_true_class = np.argmax(y_val, axis=1)
    
    
    val_accuracy = accuracy_score(y_val_true_class, y_val_pred_class)
    precision = precision_score(y_val_true_class, y_val_pred_class, average='weighted', zero_division=0)
    recall = recall_score(y_val_true_class, y_val_pred_class, average='weighted', zero_division=0)
    f1 = f1_score(y_val_true_class, y_val_pred_class, average='weighted', zero_division=0)
    cm = confusion_matrix(y_val_true_class, y_val_pred_class)
    
    print(f"Classification Report for Fold {fold}:\n",
          classification_report(y_val_true_class, y_val_pred_class, zero_division=0))
    
    fold_results.append({
         "Fold": fold,
         "Validation Loss": val_loss,
         "Validation Accuracy": val_accuracy,
         "Precision": precision,
         "Recall": recall,
         "F1 Score": f1,
         "Confusion Matrix": cm
    })
    fold += 1


results_df = pd.DataFrame([
    {k: v for k, v in fold_result.items() if k != "Confusion Matrix"}
    for fold_result in fold_results
])
print("\nExtended Testing Table for Dissertation Report:")
print(results_df)


results_df.to_csv("kfold_extended_testing_results.csv", index=False)

print("\nMean validation accuracy over folds:", np.mean(results_df["Validation Accuracy"]))

# --- Final Model Training ---


y_int_full = np.argmax(y, axis=1)
classes_full = np.unique(y_int_full)
class_weights_full = compute_class_weight('balanced', classes=classes_full, y=y_int_full)
class_weight_dict_full = {i: weight for i, weight in zip(classes_full, class_weights_full)}

final_model = create_sequence_model(input_shape, num_classes=y.shape[1])
final_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
final_model.fit(
    X, y,
    epochs=30,
    batch_size=16,
    verbose=1,
    class_weight=class_weight_dict_full,
    callbacks=[early_stopping]
)
final_model.save('classification_model.h5')
joblib.dump(encoder, 'label_encoder.joblib')