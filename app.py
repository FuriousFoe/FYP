# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import numpy as np
# import cv2
# import mediapipe as mp
# import tensorflow as tf
# from collections import Counter
#
# # Load trained LSTM model
# model = tf.keras.models.load_model("model.h5")
#
# # Initialize FastAPI
# app = FastAPI()
#
# # Define action labels (Modify as per your model)
# actions = ["Hello", "I love you", "Thanks"]
#
# # Initialize MediaPipe holistic model
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils
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
#     return keypoints.tolist()  # Convert to list for JSON compatibility
#
# # Define input schema
# class ImageData(BaseModel):
#     image: list  # Expecting an image frame as a list of pixel values
#
# # Define sequence storage
# sequence = []
# predictions = []
# sentence = []
# displayed_word = ""
# frames_required = 5  # Number of frames required for a stable word display
# threshold = 0.8
#
# @app.post("/predict")
# def predict_keypoints(data: ImageData):
#     global sequence, predictions, sentence, displayed_word
#
#     # Convert image data to OpenCV format
#     np_image = np.array(data.image, dtype=np.uint8)
#     np_image = np_image.reshape((480, 640, 3))  # Assuming 640x480 resolution
#     np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
#
#     # Process the image with MediaPipe
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         results = holistic.process(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB))
#
#     # Extract keypoints
#     keypoints = extract_keypoints(results)
#     sequence.append(keypoints)
#     sequence = sequence[-30:]  # Keep last 30 frames
#
#     # Only predict if we have 30 frames
#     if len(sequence) == 30:
#         res = model.predict(np.expand_dims(sequence, axis=0))[0]
#
#         if np.max(res) > threshold:
#             predicted_word = actions[np.argmax(res)]
#             predictions.append(predicted_word)
#             predictions = predictions[-frames_required:]
#
#             # Ensure the word remains stable for at least 80% of frames
#             most_common_word, count = Counter(predictions).most_common(1)[0]
#             if count >= 0.8 * frames_required:
#                 if displayed_word != most_common_word:
#                     displayed_word = most_common_word
#                     sentence.append(displayed_word)
#                     sentence = sentence[-10:]  # Keep last 10 words for readability
#
#     return {"prediction": displayed_word, "sentence": " ".join(sentence)}
#
# # Run using: uvicorn app:app --host 0.0.0.0 --port 8000

#########################################################################################################################################################


# # Base 64 encoding wala code which works
# Thissss fuckingggg worked, on the test.py while using live screen
# it has a latency of 35-46 seconds. Imma really happy


# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import numpy as np
# import cv2
# import mediapipe as mp
# import tensorflow as tf
# import base64
# from collections import Counter
#
# # Load trained LSTM model
# model = tf.keras.models.load_model("model.h5")
#
# # Initialize FastAPI
# app = FastAPI()
#
# # Define action labels (Modify as per your model)
# actions = ["Hello", "I love you", "Thanks"]
#
# # Initialize MediaPipe holistic model
# mp_holistic = mp.solutions.holistic
#
#
# # Define input schema
# class ImageData(BaseModel):
#     image: str  # Base64-encoded image string
#
#
# # Store previous frames for sequence prediction
# sequence = []
# predictions = []
# sentence = []
# displayed_word = ""
# frames_required = 5  # Number of frames required for a stable word display
# threshold = 0.8  # Confidence threshold
#
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
#
# @app.post("/predict")
# async def predict_keypoints(data: ImageData):
#     global sequence, predictions, sentence, displayed_word
#
#     try:
#         # Decode Base64 string to OpenCV image
#         image_data = base64.b64decode(data.image)
#         np_image = np.frombuffer(image_data, np.uint8)
#         frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
#
#         if frame is None:
#             raise HTTPException(status_code=400, detail="Invalid image data")
#
#         # Debugging step: Save the decoded image
#         cv2.imwrite("debug_image.jpg", frame)
#         print("Debug: Image saved as debug_image.jpg")
#
#         # Convert frame to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Process image with MediaPipe
#         with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#             results = holistic.process(image)
#
#         # Debugging step: Check if keypoints were detected
#         keypoints = extract_keypoints(results)
#         print("Extracted Keypoints (first 10):", keypoints[:10])
#
#         sequence.append(keypoints)
#         sequence = sequence[-30:]  # Keep last 30 frames
#
#         # Only predict if we have 30 frames
#         if len(sequence) == 30:
#             res = model.predict(np.expand_dims(sequence, axis=0))[0]
#
#             # Debugging step: Print raw model predictions
#             print("Model Prediction Probabilities:", res)
#
#             if np.max(res) > threshold:
#                 predicted_word = actions[np.argmax(res)]
#                 predictions.append(predicted_word)
#                 predictions = predictions[-frames_required:]
#
#                 # Ensure word remains stable in the last 5 frames
#                 most_common_word, count = Counter(predictions).most_common(1)[0]
#                 if count >= 0.8 * frames_required:
#                     if displayed_word != most_common_word:
#                         displayed_word = most_common_word
#                         sentence.append(displayed_word)
#                         sentence = sentence[-10:]  # Keep last 10 words
#
#         return {"prediction": displayed_word, "sentence": " ".join(sentence)}
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


#########################################################################################################################################################
# # Above code but optimized
# it has a latency of 5-6 seconds. Imma really happy

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import base64
from collections import Counter

# Load trained LSTM model once (avoid reloading per request)
model = tf.keras.models.load_model("model.h5")

# Initialize FastAPI
app = FastAPI()

# Define action labels (Modify as per your model)
actions = ["Hello", "I love you", "Thanks"]

# Initialize MediaPipe holistic model once (persistent instance)
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Define input schema
class ImageData(BaseModel):
    image: str  # Base64-encoded image string

# Store previous frames for sequence prediction
sequence = []
predictions = []
sentence = []
displayed_word = ""
frames_required = 5  # Number of frames required for a stable word display
threshold = 0.8  # Confidence threshold

# Function to extract all 1662 keypoints (unchanged)
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

@app.post("/predict")
async def predict_keypoints(data: ImageData):
    global sequence, predictions, sentence, displayed_word

    try:
        # Decode Base64 string to OpenCV image
        image_data = base64.b64decode(data.image)
        np_image = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process image with persistent MediaPipe model
        results = holistic_model.process(image)

        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep last 30 frames

        # Predict only when we have 30 frames
        if len(sequence) == 30:
            sequence_array = np.expand_dims(sequence, axis=0)
            res = model.predict(sequence_array, verbose=0)[0]  # Silent prediction

            if np.max(res) > threshold:
                predicted_word = actions[np.argmax(res)]
                predictions.append(predicted_word)
                predictions = predictions[-frames_required:]

                # Ensure word remains stable in the last 5 frames
                most_common_word, count = Counter(predictions).most_common(1)[0]
                if count >= 0.8 * frames_required:
                    if displayed_word != most_common_word:
                        displayed_word = most_common_word
                        sentence.append(displayed_word)
                        sentence = sentence[-10:]  # Keep last 10 words

        return {"prediction": displayed_word, "sentence": " ".join(sentence)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

#########################################################################################################################################################
