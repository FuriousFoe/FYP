import cv2
import base64
import requests
import json

# Local API URL
API_URL = "http://127.0.0.1:8000/predict"

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Encode the frame as Base64
    _, buffer = cv2.imencode(".jpg", frame)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    # Prepare JSON payload
    payload = json.dumps({"image": base64_image})

    # Send POST request to API
    response = requests.post(API_URL, data=payload, headers={"Content-Type": "application/json"})

    # Parse and display response
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']}, Sentence: {result['sentence']}")
    else:
        print(f"Error: {response.status_code}, Response: {response.text}")

    # Display the webcam feed
    cv2.imshow("Live Feed", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
