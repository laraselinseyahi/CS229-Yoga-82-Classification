# extract_features.py
# runs mediapipe on every image in yoga_prepared.csv and saves the landmarks
# each image gives 33 landmarks * 4 values (x, y, z, visibility) = 132 features
# output: features.npy and labels.csv in ./outputs/

import os
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from tqdm import tqdm

# MediaPipe Pose
mp_pose = mp.solutions.pose

# change these paths for the test set if needed
data_csv = "./test_outputs/yoga_prepared_test.csv"
output_dir = "./test_outputs"


def extract_landmarks(image_path, pose_model):
    # reads img into a numpy array 
    img = cv2.imread(image_path)
    if img is None:
        return None
    # convert BGR to RGB for mediapipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # run mediapipe pose model on the image to get the 33 landmarks
    results = pose_model.process(img_rgb)

    if results.pose_landmarks is None:
        return None

    landmarks = []
    # loop through the 33 landmarks and add their x, y, z, visibility to the landmarks list
    for lm in results.pose_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(landmarks, dtype=np.float32)


def main():
    df = pd.read_csv(data_csv)
    print(f"loaded {len(df)} images")

    all_features = []
    valid_indices = []

    # creates the mediapipe pose model, static_image_mode=True to not treat the images as a video stream 
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
        # loop through the rows in the df 
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # get the landmarks for the image (array of 33 * 4 numbers)
            landmarks = extract_landmarks(row["full_path"], pose)
            if landmarks is None or np.any(np.isnan(landmarks)):
                continue
            # if valid, add the landmarks to the all_features list and the index to valid_indices
            all_features.append(landmarks)
            valid_indices.append(idx)

    # convert the list of features to a numpy array
    features = np.array(all_features, dtype=np.float32)
    # create a new labels_df with only the valid indices (those that had valid landmarks)
    labels_df = df.loc[valid_indices].reset_index(drop=True)

    # save the features and labels to the output directory
    os.makedirs(output_dir, exist_ok=True)
    # numpy's file format for saving arrays is .npy
    np.save(os.path.join(output_dir, "features.npy"), features)
    labels_df.to_csv(os.path.join(output_dir, "filtered_rows.csv"), index=False)

    print(f"done! got {len(valid_indices)}/{len(df)} images, feature shape: {features.shape}")

if __name__ == "__main__":
    main()