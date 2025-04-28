import os
import cv2
from deepface import DeepFace

# -------------------------
# Paths
dataset_path = "dataset"
test_images_path = "test_images"

# DeepFace models to compare
models_to_test = ["ArcFace", "Facenet512", "VGG-Face", "Facenet"]

# Mapping folder names (user1, user2, etc.) to real names
user_mapping = {
    "user1": "Riya",
    "user2": "Suhas",
    "user3": "Nikita",
    "user4": "Dinesh",
    "user5": "suraj",
    "user6": "vinit",
    "user7": "Akanksha",
}

# Distance threshold for recognizing known vs unknown faces
distance_threshold = 0.4
# -------------------------

# Initialize accuracy counters
accuracy_results = {}

# Prepare list of test images
test_data = []

for user_folder in os.listdir(test_images_path):
    user_folder_path = os.path.join(test_images_path, user_folder)
    if os.path.isdir(user_folder_path):
        for img_file in os.listdir(user_folder_path):
            img_path = os.path.join(user_folder_path, img_file)
            test_data.append((user_folder, img_path))

# Total samples
total_samples = len(test_data)
print(f"[INFO] Found {total_samples} test images.")

# Test each model
for model in models_to_test:
    print(f"\n[INFO] Testing model: {model}")

    correct = 0

    for true_user, img_path in test_data:
        try:
            result = DeepFace.find(
                img_path=img_path,
                db_path=dataset_path,
                model_name=model,
                enforce_detection=False,
                silent=True
            )

            if result and not result[0].empty:
                identity_path = result[0].loc[0, 'identity']
                predicted_folder = os.path.basename(os.path.dirname(identity_path))
                predicted_user = user_mapping.get(predicted_folder, predicted_folder)

                true_user_name = user_mapping.get(true_user, true_user)

                if predicted_user == true_user_name:
                    correct += 1
            else:
                # Face not matched
                pass

        except Exception as e:
            print(f"[ERROR] {img_path} -> {str(e)}")

    accuracy = (correct / total_samples) * 100
    accuracy_results[model] = accuracy
    print(f"[RESULT] {model} Accuracy: {accuracy:.2f}%")

# Final Summary
print("\n[FINAL ACCURACY RESULTS]:")
for model, acc in accuracy_results.items():
    print(f"Model: {model} --> Accuracy: {acc:.2f}%")

# -------------------------
# Real-time Webcam Detection
# -------------------------

# Select best model based on accuracy
best_model = max(accuracy_results, key=accuracy_results.get)
print(f"\n[INFO] Best model selected for webcam: {best_model}")

# Start webcam
cap = cv2.VideoCapture(0)

# Load a face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face
        face_crop = frame[y:y+h, x:x+w]

        try:
            # Save the cropped face temporarily
            temp_face_path = "temp_face.jpg"
            cv2.imwrite(temp_face_path, face_crop)

            # Predict using DeepFace
            result = DeepFace.find(
                img_path=temp_face_path,
                db_path=dataset_path,
                model_name=best_model,
                enforce_detection=False,
                silent=True
            )

            if result and not result[0].empty:
                identity_path = result[0].loc[0, 'identity']
                distance = result[0].loc[0, 'distance']

                predicted_folder = os.path.basename(os.path.dirname(identity_path))
                predicted_user = user_mapping.get(predicted_folder, predicted_folder)

                if distance <= distance_threshold:
                    label = f"{predicted_user}"
                else:
                    label = "Unknown"
            else:
                label = "Unknown"

        except Exception as e:
            label = "Unknown"

        # Put label above the face rectangle
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow("Webcam Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy
cap.release()
cv2.destroyAllWindows()

# Clean up temporary file if exists
if os.path.exists("temp_face.jpg"):
    os.remove("temp_face.jpg")
