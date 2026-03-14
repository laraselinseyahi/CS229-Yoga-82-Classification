# feature_engineering.py
# computes joint angles, pairwise distances, and symmetry features from raw mediapipe landmarks
# then reruns the same classifiers from phase 1 to see if engineered features help
# output goes to ./outputs/phase2/

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings("ignore")

output_dir = "./results/phase2"

# mediapipe landmark indices we care about
nose_idx = 0
left_shoulder, right_shoulder = 11, 12
left_elbow, right_elbow = 13, 14
left_wrist, right_wrist = 15, 16
left_hip, right_hip = 23, 24
left_knee, right_knee = 25, 26
left_ankle, right_ankle = 27, 28


# each landmark takes up 4 values (x, y, z, visibility), we only need x,y,z
def get_landmark(row, idx):
    start = idx * 4 # example: landmark 2 starts at index 8 (2*4) 
    return row[start:start + 3]


def angle_between(a, b, c):
    # angle at point b
    v1 = a - b
    v2 = c - b
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

# 1) angles calculator
def compute_angles(row):
    nose = get_landmark(row, nose_idx)
    l_shoulder, r_shoulder = get_landmark(row, left_shoulder), get_landmark(row, right_shoulder)
    l_elbow, r_elbow = get_landmark(row, left_elbow), get_landmark(row, right_elbow)
    l_wrist, r_wrist = get_landmark(row, left_wrist), get_landmark(row, right_wrist)
    l_hip, r_hip = get_landmark(row, left_hip), get_landmark(row, right_hip)
    l_knee, r_knee = get_landmark(row, left_knee), get_landmark(row, right_knee)
    l_ankle, r_ankle = get_landmark(row, left_ankle), get_landmark(row, right_ankle)

    # array of 10 angles 
    return np.array([
        angle_between(l_shoulder, l_elbow, l_wrist),   # left elbow
        angle_between(r_shoulder, r_elbow, r_wrist),   # right elbow
        angle_between(l_elbow, l_shoulder, l_hip),     # left shoulder
        angle_between(r_elbow, r_shoulder, r_hip),     # right shoulder
        angle_between(l_shoulder, l_hip, l_knee),      # left hip
        angle_between(r_shoulder, r_hip, r_knee),      # right hip
        angle_between(l_hip, l_knee, l_ankle),         # left knee
        angle_between(r_hip, r_knee, r_ankle),         # right knee
        angle_between(nose, (l_shoulder + r_shoulder) / 2, (l_hip + r_hip) / 2),  # neck
        angle_between(l_shoulder, (l_hip + r_hip) / 2, r_shoulder),               # torso lean
    ], dtype=np.float32)

# 2) distances calculator
def compute_distances(row):
    l_shoulder, r_shoulder = get_landmark(row, left_shoulder), get_landmark(row, right_shoulder)
    l_hip, r_hip = get_landmark(row, left_hip), get_landmark(row, right_hip)
    l_wrist, r_wrist = get_landmark(row, left_wrist), get_landmark(row, right_wrist)
    l_ankle, r_ankle = get_landmark(row, left_ankle), get_landmark(row, right_ankle)
    l_knee, r_knee = get_landmark(row, left_knee), get_landmark(row, right_knee)
    nose = get_landmark(row, nose_idx)

    mid_shoulder = (l_shoulder + r_shoulder) / 2
    mid_hip = (l_hip + r_hip) / 2
    torso_len = np.linalg.norm(mid_shoulder - mid_hip) + 1e-8

    def dist(a, b):
        # normalize by torso length 
        return np.linalg.norm(a - b) / torso_len

    # array of 10 distances between body parts normalized by torso length
    return np.array([
        dist(l_wrist, r_wrist),        # hands apart
        dist(l_ankle, r_ankle),        # feet apart
        dist(l_wrist, l_ankle),        # left hand to left foot
        dist(r_wrist, r_ankle),        # right hand to right foot
        dist(l_wrist, r_ankle),        # cross body
        dist(r_wrist, l_ankle),        # cross body other way
        dist(nose, mid_hip),           # how upright
        dist(l_shoulder, r_shoulder),  # shoulder width
        dist(l_hip, r_hip),            # hip width
        dist(l_knee, r_knee),          # knee spread
    ], dtype=np.float32)

# 3) symmetry features
def compute_symmetry(row):
    # difference between left and right side body parts and direction of difference 
    pairs = [
        (left_shoulder, right_shoulder),
        (left_elbow, right_elbow),
        (left_wrist, right_wrist),
        (left_hip, right_hip),
        (left_knee, right_knee),
        (left_ankle, right_ankle),
    ]
    l_shoulder, r_shoulder = get_landmark(row, left_shoulder), get_landmark(row, right_shoulder)
    l_hip, r_hip = get_landmark(row, left_hip), get_landmark(row, right_hip)
    mid_shoulder = (l_shoulder + r_shoulder) / 2
    mid_hip = (l_hip + r_hip) / 2
    torso_len = np.linalg.norm(mid_shoulder - mid_hip) + 1e-8
    
    diffs = []
    # 12 values
    for l_idx, r_idx in pairs:
        left = get_landmark(row, l_idx) # the 3 landmark values 
        right = get_landmark(row, r_idx)
        # idx 1 is the y value so vertical diff
        diffs.append(left[1] - right[1])   # direction of diff (positive means left side is higher)
        diffs.append(np.linalg.norm(left - right)/ torso_len) # diff in 3D, normalize by torso length 
    return np.array(diffs, dtype=np.float32)


def engineer_features(raw_features):
    # angles 
    angles = np.array([compute_angles(row) for row in raw_features])
    # distances
    distances = np.array([compute_distances(row) for row in raw_features])
    # symmetry features
    symmetry = np.array([compute_symmetry(row) for row in raw_features])
    # horizontal stack of all engineered features: 10 angles + 10 distances + 12 symmetry = 32 features
    engineered = np.hstack([angles, distances, symmetry])
    print(f"engineered features shape: {engineered.shape}  (10 angles + 10 distances + 12 symmetry)")
    return engineered


def run_model(name, model, X_train, X_test, y_train, y_test):
    t0 = time()
    model.fit(X_train, y_train)
    t = time() - t0
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"  {name}: acc={acc:.4f}  f1={f1:.4f}  ({t:.1f}s)")
    return y_pred, acc, f1, t


def save_confusion_matrix(y_test, y_pred, label_names, title, path):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    n = len(label_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.4), max(6, n * 0.4)))
    sns.heatmap(cm_norm, annot=(n <= 20), fmt=".2f" if n <= 20 else "",
                xticklabels=label_names, yticklabels=label_names,
                cmap="Blues", ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=max(5, 10 - n // 10))
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(output_dir, exist_ok=True)

    train_features = np.load("./train_outputs/features.npy")
    train_labels = pd.read_csv("./train_outputs/filtered_rows.csv")
    test_features = np.load("./test_outputs/features.npy")
    test_labels = pd.read_csv("./test_outputs/filtered_rows.csv")
    print(f"raw features: {train_features.shape}")

    # 32 engineered features: angles, distances, symmetry
    train_engineered = engineer_features(train_features)
    test_engineered = engineer_features(test_features)

    # 132 + 32 = 164 features total when combined
    train_combined = np.hstack([train_features, train_engineered])
    test_combined = np.hstack([test_features, test_engineered])
    print(f"combined (raw + engineered): {train_combined.shape[1]} features")

    levels = {"6-class": "label_6", "20-class": "label_20", "82-class": "label_82"}
    feature_sets = {
        "engineered": (train_engineered, test_engineered),
        "combined": (train_combined, test_combined),
    }
    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42, n_jobs=-1)),
        ("SVM", SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
        ("KNN", KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, early_stopping=True, random_state=42)),
    ]

    all_results = []
    # loop through engineered features only and combined features
    for feat_name, (train_feat, test_feat) in feature_sets.items():
        print(f"{feat_name} ({train_feat.shape[1]} features):")
        # loop through 6-class, 20-class, and 82-class labels
        for level_name, col in levels.items():
            y_train = train_labels[col].values
            y_test = test_labels[col].values
            label_names = sorted(train_labels[col].unique())

            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_feat)
            X_test = scaler.transform(test_feat)

            for name, model in models:
                y_pred, acc, f1, t = run_model(name, model, X_train, X_test, y_train, y_test)
                cm_path = os.path.join(output_dir, f"cm_{name.lower().replace(' ','_')}_{level_name.replace('-','_')}_{feat_name}.png")
                save_confusion_matrix(y_test, y_pred, label_names, f"{name} {level_name} {feat_name}", cm_path)
                all_results.append({"features": feat_name, "model": name, "level": level_name, "accuracy": round(acc, 4), "f1": round(f1, 4), "time": round(t, 2)})

    results_df = pd.DataFrame(all_results)
    print("\n", results_df.to_string())
    results_df.to_csv(os.path.join(output_dir, "results_phase2.csv"), index=False)

if __name__ == "__main__":
    main()
