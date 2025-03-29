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
# # Define input schema
# class ImageData(BaseModel):
#     image: str  # Base64-encoded image string
#
# # Store previous frames for sequence prediction
# sequence = []
# predictions = []
# sentence = []
# displayed_word = ""
# frames_required = 5  # Number of frames required for a stable word display
# threshold = 0.8  # Confidence threshold
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
#         # Convert frame to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Process image with MediaPipe
#         with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#             results = holistic.process(image)
#
#         # Extract keypoints
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

# experimenting with Batch Frame Processing and Optimized MediaPipe Processing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import base64
from collections import Counter

# Load trained LSTM model
model = tf.keras.models.load_model("model.h5")

# Initialize FastAPI
app = FastAPI()

# Define action labels
actions = ["Hello", "I love you", "Thanks"]

# Initialize MediaPipe holistic **once** (to avoid re-initializing in every request)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define input schema for multiple frames
class ImageData(BaseModel):
    frames: list[str]  # List of Base64-encoded images

# Store previous frames for sequence prediction
sequence = []
predictions = []
sentence = []
displayed_word = ""
frames_required = 5
threshold = 0.8

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

@app.post("/predict")
async def predict_keypoints(data: ImageData):
    global sequence, predictions, sentence, displayed_word

    try:
        for image_base64 in data.frames:
            # Decode Base64 string to OpenCV image
            image_data = base64.b64decode(image_base64)
            np_image = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

            if frame is None:
                raise HTTPException(status_code=400, detail="Invalid image data")

            # Convert frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process image with MediaPipe (reusing global holistic model)
            results = holistic.process(image)

            # Extract keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Keep last 30 frames

        # Only predict if we have 30 frames
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]

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
