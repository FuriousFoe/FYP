# import cv2
# import numpy as np
# import mediapipe as mp
# import tensorflow as tf
# from collections import Counter
#
# # Load trained LSTM model
# model = tf.keras.models.load_model("model.h5")
#
# # Initialize MediaPipe holistic model
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils
#
# # Define action labels
# actions = ["Hello", "I love you", "Thanks"]
#
# # Function to extract keypoints from MediaPipe results
# def extract_keypoints(results):
#     pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
#                      results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
#     face = np.array([[res.x, res.y, res.z] for res in
#                      results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
#     lh = np.array([[res.x, res.y, res.z] for res in
#                    results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
#     rh = np.array([[res.x, res.y, res.z] for res in
#                    results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
#
#     keypoints = np.concatenate([pose, face, lh, rh])  # Ensures 1662 features
#     return keypoints
#
# # Initialize video capture
# cap = cv2.VideoCapture(0)
#
# sequence = []
# predictions = []
# displayed_word = ""
# frames_required = 5  # Number of frames required for a stable word display
# threshold = 0.8
#
# # OpenCV loop for real-time keypoint extraction and prediction
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Convert frame color (BGR to RGB)
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#
#         # Make detections
#         results = holistic.process(image)
#
#         # Convert back to BGR for OpenCV rendering
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#         # Draw landmarks
#         mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#         mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#         mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#
#         # Extract keypoints & store in sequence
#         keypoints = extract_keypoints(results)
#         sequence.append(keypoints)
#         sequence = sequence[-30:]  # Keep last 30 frames
#
#         # Only predict if we have 30 frames
#         if len(sequence) == 30:
#             res = model.predict(np.expand_dims(sequence, axis=0))[0]
#
#             if np.max(res) > threshold:
#                 predicted_word = actions[np.argmax(res)]
#                 predictions.append(predicted_word)
#                 predictions = predictions[-frames_required:]  # Keep last few frames
#
#                 # Only update the displayed word if it appears at least 80% of the time in the last `frames_required` frames
#                 most_common_word, count = Counter(predictions).most_common(1)[0]
#                 if count >= 0.8 * frames_required:
#                     displayed_word = most_common_word
#
#         # Display word on screen
#         cv2.putText(image, f'Prediction: {displayed_word}', (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#
#         # Show the output
#         cv2.imshow("Live Feed", image)
#
#         if cv2.waitKey(10) & 0xFF == ord("q"):
#             break
#
# # Release resources
# cap.release()
# cv2.destroyAllWindows()
#

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import Counter

# Load trained LSTM model
model = tf.keras.models.load_model("model.h5")

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define action labels (modify this based on your model's classes)
actions = ["Hello", "I love you", "Thanks"]

# Function to extract keypoints from MediaPipe results
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)

    keypoints = np.concatenate([pose, face, lh, rh])  # Ensures 1662 features
    return keypoints

# Initialize video capture
cap = cv2.VideoCapture(0)

sequence = []
predictions = []
sentence = []  # To store sentence formation
displayed_word = ""
frames_required = 5  # Number of frames required for a stable word display
threshold = 0.8

# OpenCV loop for real-time keypoint extraction and prediction
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame color (BGR to RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detections
        results = holistic.process(image)

        # Convert back to BGR for OpenCV rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.face_mesh.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract keypoints & store in sequence
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep last 30 frames

        # Only predict if we have 30 frames
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

            if np.max(res) > threshold:
                predicted_word = actions[np.argmax(res)]
                predictions.append(predicted_word)
                predictions = predictions[-frames_required:]  # Keep last few frames

                # Only update displayed_word if it appears at least 80% of the time in the last `frames_required` frames
                most_common_word, count = Counter(predictions).most_common(1)[0]
                if count >= 0.8 * frames_required:
                    if displayed_word != most_common_word:  # Avoid duplicates in the sentence
                        displayed_word = most_common_word
                        sentence.append(displayed_word)  # Add to sentence
                        sentence = sentence[-10:]  # Keep only last 10 words for better readability

        # Display word and sentence on screen
        cv2.putText(image, f'Prediction: {displayed_word}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(image, f'Sentence: {" ".join(sentence)}', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the output
        cv2.imshow("Live Feed", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
