import tkinter as tk
from tkinter import messagebox
import cv2
from deepface import DeepFace

# Function to start emotion detection
def start_emotion_detection():
    def run_detection():
        # Initialize webcam capture
        cap = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            # Analyze the current frame for emotion
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

                # Debugging: Print the result structure
                print("Result:", result)

                # Check if the result is a list and has elements
                if isinstance(result, list) and len(result) > 0:
                    # Extract the first result (if multiple faces are detected)
                    first_result = result[0]

                    # Get the predicted emotion and its confidence score
                    dominant_emotion = first_result['dominant_emotion']
                    emotion_scores = first_result['emotion']

                    # Filter emotions to include only happy, neutral, and surprise
                    filtered_emotions = {
                        key: emotion_scores[key]
                        for key in ['happy', 'neutral', 'surprise']
                        if key in emotion_scores
                    }

                    # Display the dominant emotion on the frame
                    cv2.putText(
                        frame,
                        f"Emotion: {dominant_emotion}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    # Optionally, display the filtered emotions with their confidence scores
                    for i, (emotion, score) in enumerate(filtered_emotions.items()):
                        text = f"{emotion}: {score:.2f}%"
                        cv2.putText(
                            frame,
                            text,
                            (50, 100 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

                else:
                    print("No faces detected.")

            except Exception as e:
                print(f"Error in DeepFace analysis: {e}")

            # Display the frame with emotion predictions
            cv2.imshow('Emotion Detector', frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

    # Run the detection in a non-blocking manner
    root.after(100, run_detection)


# Create the Tkinter GUI
root = tk.Tk()
root.title("Emotion Detector")
root.geometry("400x200")

# Add a label
label = tk.Label(
    root,
    text="Click Start to Detect Your Emotions!",
    font=("Arial", 16),
    pady=20
)
label.pack()

# Add a button to start detection
start_button = tk.Button(
    root,
    text="Start Detection",
    font=("Arial", 14),
    command=start_emotion_detection
)
start_button.pack(pady=20)

# Add a quit button to close the application
quit_button = tk.Button(
    root,
    text="Quit",
    font=("Arial", 14),
    command=root.quit
)
quit_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
