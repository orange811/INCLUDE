import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

# Function to calculate movement using Euclidean distance
def calculate_movement_euclidean(frame1, frame2):
    return np.sqrt(np.sum((frame1 - frame2) ** 2))

# Function to compute movements for all frames in a video
def compute_movements(frames):
    movements = []
    for i in range(1, len(frames)):
        movement = calculate_movement_euclidean(frames[i], frames[i - 1])
        movements.append(movement)
    return movements

# Function to apply exponential falloff weights
def apply_exponential_falloff(movement_scores, alpha=0.05):
    n = len(movement_scores)
    weights = np.exp(-alpha * np.abs(np.arange(n) - n / 2))
    weighted_movements = movement_scores * weights
    return weighted_movements

# Select the best consecutive frames based on movement
def select_consecutive_frames(movement_scores, n_frames, window_size=60):
    if n_frames < window_size:
        # If the number of frames is less than the window size, select all frames
        window_size = n_frames
    
    max_sum = -1
    best_start = 0

    for i in range(n_frames - window_size + 1):
        current_sum = sum(movement_scores[i : i + window_size])
        if current_sum > max_sum:
            max_sum = current_sum
            best_start = i

    return best_start, best_start + window_size

# Directory containing JSON files
input_directory = "D:\\Neha\\BE\\final year project\\INCLUDE_git_pew\\keypoints_dir\\include_test_keypoints"
results = []

# Process each JSON file in the directory
for filename in tqdm(os.listdir(input_directory), desc="Processing Videos"):
    if filename.endswith(".json"):
        file_path = os.path.join(input_directory, filename)

        # Load JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Check for missing data
        if "pose_x" not in data or "pose_y" not in data or "n_frames" not in data:
            print(f"Missing data in {filename}, skipping.")
            continue

        # Extract information from JSON
        video_class = data["label"]
        video_uid = data["uid"]
        pose_x = np.array(data["pose_x"])
        pose_y = np.array(data["pose_y"])
        n_frames = data["n_frames"]
        
        # Combine x and y to form 25 landmarks per frame
        frames = np.stack((pose_x, pose_y), axis=-1).reshape(pose_x.shape[0], -1)
        
        # Compute movement scores for the video
        movement_scores = compute_movements(frames)
        
        # Apply exponential falloff to movement scores
        weighted_movements = apply_exponential_falloff(np.array(movement_scores))
        
        # Select the best 60 consecutive frames
        start, end = select_consecutive_frames(weighted_movements, n_frames, window_size=60)
        
        # Store the result
        results.append([video_class, video_uid, start, end])

# Convert the results into a DataFrame
output_df = pd.DataFrame(results, columns=["class", "video_uid", "start", "end"])

# Save to a new CSV file
output_df.to_csv("output_file.csv", index=False)

print("Processing complete. Results saved to 'output_file.csv'.")
