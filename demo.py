import collections
import numpy as np
import cv2
import time
import mediapipe as mp
import tensorflow as tf
import joblib
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load the trained model
model = tf.keras.models.load_model("demo/models/train_1.h5")

# Load the label encoder
label_encoder = joblib.load("demo/label_encoder.pkl")  # Load the saved label encoder

# Initialize the video capture from the webcam
video = cv2.VideoCapture(0)

# Allow camera to warm up for 2 seconds
time.sleep(2.0)

# To store the last 45 frames of pose landmarks
landmark_buffer = collections.deque(maxlen=45)  # Store the most recent 45 frames


# Function to draw landmarks on the image
def draw_landmarks_on_image(rgb_image, pose_detection_result, hand_detection_result):
    annotated_image = np.copy(rgb_image)
    hand_landmarks_list = hand_detection_result.hand_landmarks
    pose_landmarks_list = pose_detection_result.pose_landmarks

    # Check if pose landmarks are detected
    if pose_landmarks_list:
        for pose_landmarks in pose_landmarks_list:
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in pose_landmarks  # Directly loop over pose_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )

    # Check if hand landmarks are detected
    if hand_landmarks_list:
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in hand_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style(),
            )
    return annotated_image


# Create a PoseLandmarker object
pose_base_options = python.BaseOptions(model_asset_path="demo/pose_landmarker.task")
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options  # ,output_segmentation_masks=True #to draw segmentation mask
)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

# Creating a HandLandmarker object
hand_base_options = python.BaseOptions(model_asset_path="demo/hand_landmarker.task")
hand_options = vision.HandLandmarkerOptions(base_options=hand_base_options, num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)


while True:
    # Capture a frame from the webcam
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to RGB format (required by MediaPipe)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the image to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    annotated_image = np.copy(rgb_image)
    # Detect pose landmarks from the input frame
    pose_detection_result = pose_detector.detect(mp_image)
    hand_detection_result = hand_detector.detect(mp_image)

    # If there are any detected poses, draw them on the frame
    annotated_image = draw_landmarks_on_image(
        rgb_image, pose_detection_result, hand_detection_result
    )
    # Convert the annotated image back to BGR for display
    if pose_detection_result.pose_landmarks:
        frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        # Extract landmark data for prediction
        """
        pose_landmarks_data = []
        for pose_landmarks in pose_detection_result.pose_landmarks:
            # Access only the first 24 landmarks for each frame (if your model expects this)
            for landmark in pose_landmarks[:24]:  # Extract only the first 24 landmarks
                pose_landmarks_data.append([landmark.x, landmark.y, landmark.z])

        # Flatten the pose landmarks and add to the buffer
        pose_landmarks_data = np.array(pose_landmarks_data).flatten()
        landmark_buffer.append(
            pose_landmarks_data
        )  # Add the current frame to the buffer

        # If we have accumulated 45 frames, make a prediction
        if len(landmark_buffer) == 45:
            # Prepare the input for the model (45 frames, 72 features per frame)
            model_input = np.array(landmark_buffer).reshape(1, 45, 72)

            # Predict the class
            predictions = model.predict(model_input)
            predicted_class = np.argmax(predictions, axis=1)[
                0
            ]  # Get the index of the highest score

            # Decode the predicted class index to the actual label
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]

            # Display the prediction on the frame
            cv2.putText(
                frame,
                f"Predicted: {predicted_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
    """
    # Display the frame with or without annotations
    cv2.imshow("Pose Detection", frame)

    # Check if the user presses 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
