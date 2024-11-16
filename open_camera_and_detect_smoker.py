import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load the trained model (assuming it's saved as .h5)
model = load_model('smoker_classification_model.h5')

# Define a function to preprocess the frame before prediction
def preprocess_frame(frame):
    # Resize the image to 250x250 (as your model expects this input size)
    img = cv2.resize(frame, (250, 250))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype('float32') / 255  # Normalize the image
    return img

# Start the video capture (0 is the default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)


    # Make a prediction
    prediction = model.predict(processed_frame)
    if prediction[0] > 0.5:
        label = "Smoking"
        color = (0, 0, 255)  # Red
    else:
        label = "Not Smoking"
        color = (0, 255, 0)  # Green

    # Put the label and rectangle on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (10, 40), (250, 70), color, 2)

    # Display the resulting frame
    cv2.imshow('Smoking Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
