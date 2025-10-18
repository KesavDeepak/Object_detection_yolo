import cv2
from ultralytics import YOLO

model_path = r'C:\Users\admin\Desktop\full classification\yolo11n.pt'
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
else:
    print("Accessing webcam. Press 'q' to quit.")

threshold = 0.5  

while True:
    ret, frame = cap.read()  
    if not ret:
        print("Error: Couldn't capture a frame.")
        break

    
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            padding = 5
            class_name = results.names[int(class_id)]
            print(f"Detected: {class_name} with confidence {score:.2f}")

            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(frame.shape[1], int(x2) + padding)
            y2 = min(frame.shape[0], int(y2) + padding)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

            text = results.names[int(class_id)].upper()
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

   
    cv2.imshow('Webcam Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
