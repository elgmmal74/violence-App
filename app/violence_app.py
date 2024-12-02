import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Load the model
model = keras.models.load_model('violence_detection_model.h5')

# Define the classes
classes = ['Normal', 'Violence']

frame_buffer = []
sequence_length = 15

# Open the video file
cap = cv2.VideoCapture('vid.mp4')  # Replace with your video file path

# Initialize the figure for displaying the video
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()

while cap.isOpened():
    ret, frame = cap.read()
    
    # Check if frame is read properly
    if not ret:
        print("End of video or failed to read frame.")
        break

    # Resize the frame and normalize
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame / 255.0
    frame_buffer.append(normalized_frame)

    # Make a prediction if we have 15 frames
    if len(frame_buffer) == sequence_length:
        input_batch = np.expand_dims(frame_buffer, axis=0)
        prediction = model.predict(input_batch)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index]
        label = f"{classes[class_index]} ({confidence:.2f})"
        
        # Display the prediction on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_buffer.pop(0)

    # Convert BGR to RGB for Matplotlib display
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame using Matplotlib
    ax.imshow(rgb_frame)
    ax.axis('off')
    plt.draw()
    plt.pause(0.001)
    ax.clear()  # Clear the previous frame to display the next one

cap.release()  # Release the video capture object
plt.close()  # Close the plotting window
print("Video processing completed.")
