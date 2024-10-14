import time
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def load_model(model_path):
    """Loads the YOLO model and measures the time taken."""
    start_time = time.time()
    model = YOLO(model_path)
    model_load_time = time.time() - start_time
    print(f"Model loaded in {model_load_time:.4f} seconds")
    return model, model_load_time

def load_image(image_path):
    """Loads an image from the given path and measures the time taken."""
    start_time = time.time()
    image = cv2.imread(image_path)
    image_load_time = time.time() - start_time
    print(f"Image loaded in {image_load_time:.4f} seconds")
    return image, image_load_time

def run_inference(model, image):
    """Runs inference on the given image using the loaded model and measures the time taken."""
    start_time = time.time()
    results = model.predict(source=image, show=True)
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.4f} seconds")
    return results, inference_time

def display_and_save_results(results, save_path='annotated_image.jpg'):
    """Displays and saves the annotated image."""
    annotated_image = results[0].plot()

    # Display the result using matplotlib
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axis
    plt.show()

    # Save the result image if needed
    cv2.imwrite(save_path, annotated_image)

    # Access and print detection details (labels, boxes, scores)
    for result in results:
        print(result.boxes)  # Bounding box coordinates
        print(result.scores)  # Confidence scores
        print(result.labels)  # Detected class labels

if __name__ == '__main__':
    # Define the paths
    model_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\runs\detect\train18\weights\best.pt"
    image_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteModelTrainer\Dataset\SplitDataset\test\images\34383_1725465243.164887_original.webp"
    # image_path = r"C:\Users\sa-forest\GaTech Dropbox\Benjamin Magondu\YOLOretrainingdata\Pipette CNN Training Data\20191016\3654099075.png"
    # image_path = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteTipModel\training\dataset\test\3654098892.png"

    # Load the model
    model, model_load_time = load_model(model_path)

    # Load the image
    image, image_load_time = load_image(image_path)

    # Run inference
    results, inference_time = run_inference(model, image)

    # Display and save results
    display_and_save_results(results)

    # Calculate and print total time taken
    total_time = model_load_time + image_load_time + inference_time
    print(f"Total time taken: {total_time:.4f} seconds")

