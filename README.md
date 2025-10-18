
**Object Detection using YOLO (You Only Look Once)**

---

### **AIM:**

To implement an object detection model using the **YOLO (You Only Look Once)** algorithm that can identify and locate multiple objects in an image or video in real-time.

---

### **APPARATUS REQUIRED:**

* **Hardware:**

  * Laptop/PC with minimum 8 GB RAM
  * NVIDIA GPU (optional but recommended for faster processing)
  * Webcam (for live detection)
* **Software:**

  * Operating System: Windows / Linux / macOS
  * Python (≥ 3.8)
  * IDE: Jupyter Notebook / VS Code / PyCharm
  * Libraries:

    * `opencv-python`
    * `numpy`
    * `torch`, `torchvision`
    * `ultralytics` (for YOLOv5/YOLOv8)
    * `matplotlib`

---

### **THEORY:**

**YOLO (You Only Look Once)** is a real-time object detection algorithm that treats detection as a **single regression problem**.
Instead of running a classifier on sliding windows (like older methods), YOLO divides the image into a grid and predicts **bounding boxes** and **class probabilities** directly from full images in one evaluation.

#### **Key Concepts:**

1. **Single-Stage Detection:**
   YOLO performs detection and classification in one step, making it very fast.

2. **Grid Division:**
   The image is divided into *S × S* grid cells. Each cell predicts:

   * Bounding box coordinates (x, y, width, height)
   * Objectness score
   * Class probabilities

3. **Advantages:**

   * Real-time performance
   * High accuracy
   * End-to-end trainable

4. **YOLO Versions:**

   * **YOLOv3:** Darknet-53 backbone
   * **YOLOv5:** PyTorch implementation, faster and more efficient
   * **YOLOv8:** Latest version from Ultralytics with improved performance and auto-shape optimization

---

### **PROCEDURE:**

1. **Step 1:** Install required Python libraries

   ```bash
   pip install opencv-python numpy torch torchvision ultralytics
   ```

2. **Step 2:** Import necessary libraries in your Python script.

3. **Step 3:** Load a pre-trained YOLO model (e.g., YOLOv8s).

4. **Step 4:** Load the input image or video file.

5. **Step 5:** Perform object detection using the model.

6. **Step 6:** Display or save the detected output with bounding boxes and labels.

7. **Step 7:** Analyze the detected results and record performance metrics.

---

### **PROGRAM:**

```python
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

```

---

### **OUTPUT:**

**Input Image:**
An image containing multiple objects such as cars, people, or animals.

**Detected Output:**
Objects detected with:

* Bounding boxes
* Class labels (e.g., *person, car, dog*)
* Confidence scores

Example (conceptual):

| Input Image                                                    | Output with Detections                                               |
| -------------------------------------------------------------- | -------------------------------------------------------------------- |
| ![Input](https://via.placeholder.com/400x250?text=Input+Image) | ![Output](https://via.placeholder.com/400x250?text=Detected+Objects) |

---

### **RESULT:**

The **YOLO-based Object Detection model** was successfully implemented.
It accurately detected and localized multiple objects in real-time with high precision and speed.
This demonstrates YOLO’s efficiency in performing end-to-end object detection suitable for real-world applications such as surveillance, autonomous vehicles, and smart cameras.

---


---

Would you like me to tailor this version specifically for **YOLOv5** (since it’s more common in labs and academic demos than YOLOv8)?
I can rewrite the *program* and *procedure* accordingly.
