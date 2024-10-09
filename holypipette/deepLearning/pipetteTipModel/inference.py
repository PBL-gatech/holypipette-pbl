import time
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
# Timing the overall process
# 1. Measure time to load the model
start_time = time.time()
model = YOLO('yolov5/yolov5s.pt')  # You can replace 'yolov8n.pt' with other model variants like 'yolov8s.pt'
model_load_time = time.time() - start_time
print(f"Model loaded in {model_load_time:.4f} seconds")
# 2. Measure time to load the image
image_path = ''
start_time = time.time()
image = cv2.imread(image_path)
image_load_time = time.time() - start_time
print(f"Image loaded in {image_load_time:.4f} seconds")
# 3. Measure time for inference (prediction)
start_time = time.time()
results = model.predict(source=image, show=True)  # show=True will display the image with the detected boxes
inference_time = time.time() - start_time
print(f"Inference completed in {inference_time:.4f} seconds")
# Optional: Save and visualize the annotated image
annotated_image = results[0].plot()
# Display the result using matplotlib
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Hide axis
plt.show()
# Save the result image if needed
cv2.imwrite('annotated_image.jpg', annotated_image)
# Optional: Access detection details (labels, boxes, scores)
for result in results:
    print(result.boxes)  # Bounding box coordinates
    print(result.scores)  # Confidence scores
    print(result.labels)  # Detected class labels
# Total time taken for the entire process
total_time = model_load_time + image_load_time + inference_time
print(f"Total time taken: {total_time:.4f} seconds")