import os
import cv2
import numpy as np
from pathlib import Path

# PATH CONFIGURATION

DATASET_DIR = Path(r"C:\Users\Neil\Downloads\backex\Cut Vids Final")

OUTPUT_DIR = Path(r"C:\Users\Neil\Downloads\backex\back_exercise_form_detector")
FEATURE_DIR = OUTPUT_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# Folder name mapping
exercise_name_map = {
    "ex1": "lat_pulldown",
    "ex2": "seated_cable_row",
    "ex3": "cable_pullover"
}

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]

# Sequence length
SEQ_LEN = 60

# Frame size for faster processing
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Shi-Tomasi keypoint detection parameters
feature_params = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Lucas-Kanade optical flow parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10,
        0.03
    )
)

# FEATURE COMPUTATION

def compute_motion_features(old_points, new_points, status):
    """
    Extracts motion features from tracked optical flow points.

    Feature order:
    1. Mean x-displacement
    2. Mean y-displacement
    3. Mean motion magnitude
    4. Mean motion direction
    5. Number of tracked points
    6. Standard deviation of x-displacement
    7. Standard deviation of y-displacement
    8. Standard deviation of motion magnitude
    """

    if old_points is None or new_points is None or status is None:
        return np.zeros(8, dtype=np.float32)

    good_new = new_points[status == 1].reshape(-1, 2)
    good_old = old_points[status == 1].reshape(-1, 2)

    if len(good_new) == 0:
        return np.zeros(8, dtype=np.float32)

    displacement = good_new - good_old

    dx = displacement[:, 0]
    dy = displacement[:, 1]

    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    direction = np.arctan2(dy, dx)

    features = np.array([
        np.mean(dx),
        np.mean(dy),
        np.mean(magnitude),
        np.mean(direction),
        len(good_new),
        np.std(dx),
        np.std(dy),
        np.std(magnitude)
    ], dtype=np.float32)

    return features


def pad_or_trim(sequence, seq_len=SEQ_LEN):
    """
    Makes all video feature sequences the same length.
    """

    sequence = np.array(sequence, dtype=np.float32)

    if len(sequence) == 0:
        return np.zeros((seq_len, 8), dtype=np.float32)

    if len(sequence) < seq_len:
        padding = np.zeros((seq_len - len(sequence), 8), dtype=np.float32)
        sequence = np.vstack([sequence, padding])

    elif len(sequence) > seq_len:
        indices = np.linspace(0, len(sequence) - 1, seq_len).astype(int)
        sequence = sequence[indices]

    return sequence


def extract_features_from_video(video_path):
    """
    Extracts Lucas-Kanade optical flow features from one video.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None

    ret, old_frame = cap.read()

    if not ret:
        print(f"Could not read first frame: {video_path}")
        cap.release()
        return None

    old_frame = cv2.resize(old_frame, (FRAME_WIDTH, FRAME_HEIGHT))
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    old_points = cv2.goodFeaturesToTrack(
        old_gray,
        mask=None,
        **feature_params
    )

    sequence_features = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if old_points is None or len(old_points) < 10:
            old_points = cv2.goodFeaturesToTrack(
                old_gray,
                mask=None,
                **feature_params
            )

            if old_points is None:
                sequence_features.append(np.zeros(8, dtype=np.float32))
                old_gray = frame_gray.copy()
                continue

        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray,
            frame_gray,
            old_points,
            None,
            **lk_params
        )

        features = compute_motion_features(old_points, new_points, status)
        sequence_features.append(features)

        if new_points is not None and status is not None:
            good_new = new_points[status == 1]

            if len(good_new) > 0:
                old_points = good_new.reshape(-1, 1, 2)
            else:
                old_points = cv2.goodFeaturesToTrack(
                    frame_gray,
                    mask=None,
                    **feature_params
                )
        else:
            old_points = cv2.goodFeaturesToTrack(
                frame_gray,
                mask=None,
                **feature_params
            )

        old_gray = frame_gray.copy()

    cap.release()

    return pad_or_trim(sequence_features, SEQ_LEN)


# PROCESS DATASET

def process_dataset():
    X = []
    exercise_labels = []
    form_labels = []

    for exercise_folder in DATASET_DIR.iterdir():
        if not exercise_folder.is_dir():
            continue

        folder_name = exercise_folder.name

        if folder_name not in exercise_name_map:
            print(f"Skipping unknown folder: {folder_name}")
            continue

        exercise_label = exercise_name_map[folder_name]

        for form_folder in exercise_folder.iterdir():
            if not form_folder.is_dir():
                continue

            form_label = form_folder.name.lower()

            if form_label not in ["good", "bad"]:
                print(f"Skipping unknown form folder: {form_folder}")
                continue

            for video_file in form_folder.iterdir():
                if video_file.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue

                print(f"Processing: {video_file}")
                print(f"Exercise: {exercise_label}")
                print(f"Form: {form_label}")

                features = extract_features_from_video(video_file)

                if features is not None:
                    X.append(features)
                    exercise_labels.append(exercise_label)
                    form_labels.append(form_label)

    X = np.array(X, dtype=np.float32)
    exercise_labels = np.array(exercise_labels)
    form_labels = np.array(form_labels)

    np.save(FEATURE_DIR / "X.npy", X)
    np.save(FEATURE_DIR / "exercise_labels.npy", exercise_labels)
    np.save(FEATURE_DIR / "form_labels.npy", form_labels)

    print("\nFeature extraction finished.")
    print("Saved to:", FEATURE_DIR)
    print("X shape:", X.shape)
    print("Exercise labels:", exercise_labels.shape)
    print("Form labels:", form_labels.shape)


if __name__ == "__main__":
    process_dataset()