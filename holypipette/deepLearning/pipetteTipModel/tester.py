import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_recall_curve, average_precision_score

# ================================
# Configuration
# ================================

MODEL_PATH = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteTipModel\results2\train\weights\best.pt"  # Path to your trained YOLO model
IMAGES_DIR = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteModelTrainer\finder_training_set\SplitDatasetnoaug\test\images"  # Directory containing test images
LABELS_DIR = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteModelTrainer\finder_training_set\SplitDatasetnoaug\test\labels"  # Directory containing YOLO-formatted label files
RESULTS_DIR = r"C:\Users\sa-forest\Documents\GitHub\holypipette-pbl\holypipette\deepLearning\pipetteTipModel\results2\test"  # Directory to save all results

# Create necessary directories
annotated_dir = os.path.join(RESULTS_DIR, "annotated_images")
histograms_dir = os.path.join(RESULTS_DIR, "histograms")
plots_dir = os.path.join(RESULTS_DIR, "plots")
os.makedirs(annotated_dir, exist_ok=True)
os.makedirs(histograms_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "metrics.csv")

# ================================
# Helper Functions
# ================================

def load_ground_truth(label_path, img_width, img_height):
    """Load ground truth boxes from a YOLO-formatted label file."""
    ground_truths = []
    if not os.path.exists(label_path):
        return ground_truths
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls, x_c, y_c, w, h = parts[:5]
            x_c, y_c, w, h = map(float, [x_c, y_c, w, h])
            x1 = (x_c - w / 2) * img_width
            y1 = (y_c - h / 2) * img_height
            x2 = (x_c + w / 2) * img_width
            y2 = (y_c + h / 2) * img_height
            ground_truths.append({
                'class': cls,
                'bbox': (x1, y1, x2, y2),
                'centroid': ((x1 + x2)/2, (y1 + y2)/2)
            })
    return ground_truths

def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = (boxA[2]-boxA[0]+1) * (boxA[3]-boxA[1]+1)
    boxB_area = (boxB[2]-boxB[0]+1) * (boxB[3]-boxB[1]+1)
    iou = inter_area / float(boxA_area + boxB_area - inter_area) if (boxA_area + boxB_area - inter_area) else 0
    return iou

def match_predictions(preds, gts, iou_threshold=0.5):
    """Match predictions to ground truths based on IoU."""
    if not preds and not gts:
        return [], [], []
    elif not preds:
        # No predictions, all ground truths are unmatched
        return [], [], list(range(len(gts)))
    elif not gts:
        # No ground truths, all predictions are unmatched
        return [], list(range(len(preds))), []
    
    cost_matrix = np.array([[1 - calculate_iou(p['bbox'], g['bbox']) for g in gts] for p in preds])
    row, col = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_preds = set(range(len(preds)))
    unmatched_gts = set(range(len(gts)))
    for r, c in zip(row, col):
        if calculate_iou(preds[r]['bbox'], gts[c]['bbox']) >= iou_threshold:
            matches.append((r, c))
            unmatched_preds.discard(r)
            unmatched_gts.discard(c)
    return matches, list(unmatched_preds), list(unmatched_gts)

# ================================
# Initialize Model
# ================================

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

# ================================
# Initialize Metrics Dictionary
# ================================

metrics = {}
# Structure: {class_id: {'errors': [], 'y_true': [], 'y_scores': []}}

# To collect all predictions and ground truths for precision-recall calculations
all_predictions = {}
all_ground_truths = {}

# ================================
# Processing Test Images
# ================================

image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Found {len(image_files)} images in {IMAGES_DIR}.")

for idx, image_file in enumerate(image_files, 1):
    img_path = os.path.join(IMAGES_DIR, image_file)
    label_path = os.path.join(LABELS_DIR, os.path.splitext(image_file)[0] + ".txt")
    image = cv2.imread(img_path)
    if image is None:
        print(f"[{idx}/{len(image_files)}] Failed to read {img_path}. Skipping.")
        continue
    h, w = image.shape[:2]
    
    # Run inference
    results = model(image)[0]
    preds = []
    for box in results.boxes:
        cls = str(int(box.cls[0]))
        conf = box.conf[0]
        bbox = box.xyxy[0].tolist()
        preds.append({'class': cls, 'conf': conf, 'bbox': bbox})
    
    # Load ground truth
    gts = load_ground_truth(label_path, w, h)
    
    # Organize by class
    pred_by_class = {}
    for pred in preds:
        pred_by_class.setdefault(pred['class'], []).append(pred)
    gt_by_class = {}
    for gt in gts:
        gt_by_class.setdefault(gt['class'], []).append(gt)
    
    # Match predictions to ground truths per class
    for cls in set(list(pred_by_class.keys()) + list(gt_by_class.keys())):
        preds_cls = pred_by_class.get(cls, [])
        gts_cls = gt_by_class.get(cls, [])
        matches, unmatched_preds, unmatched_gts = match_predictions(preds_cls, gts_cls, iou_threshold=0.5)
        
        # Initialize metrics
        metrics.setdefault(cls, {'errors': [], 'y_true': [], 'y_scores': []})
        all_ground_truths.setdefault(cls, 0)
        all_ground_truths[cls] += len(gts_cls)
        
        # Label predictions: 1 for TP, 0 for FP
        for i, pred in enumerate(preds_cls):
            if i in unmatched_preds:
                metrics[cls]['y_true'].append(0)
                metrics[cls]['y_scores'].append(pred['conf'].cpu().item())  # Move to CPU and convert to float
            else:
                metrics[cls]['y_true'].append(1)
                metrics[cls]['y_scores'].append(pred['conf'].cpu().item())  # Move to CPU and convert to float
        
        # Calculate centroid errors for matched predictions
        for p_idx, g_idx in matches:
            p_bbox = preds_cls[p_idx]['bbox']
            p_centroid = ((p_bbox[0] + p_bbox[2]) / 2, (p_bbox[1] + p_bbox[3]) / 2)
            g_centroid = gts_cls[g_idx]['centroid']
            error = np.linalg.norm(np.array(p_centroid) - np.array(g_centroid))
            metrics[cls]['errors'].append(error)
    
    # Annotate and save image
    annotated = image.copy()
    for gt in gts:
        x1, y1, x2, y2 = map(int, gt['bbox'])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"GT: {gt['class']}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for pred in preds:
        x1, y1, x2, y2 = map(int, pred['bbox'])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated, f"Pred: {pred['class']} {pred['conf']:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(annotated_dir, image_file), annotated)
    print(f"[{idx}/{len(image_files)}] Processed {image_file}")

# ================================
# Calculate Metrics
# ================================

print("\nCalculating Metrics...")

csv_data = []
for cls, data in metrics.items():
    y_true = data['y_true']
    y_scores = data['y_scores']
    
    if len(y_true) == 0:
        precision = recall = f1 = average_precision = 0
    else:
        # Ensure y_true and y_scores are numpy arrays
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Calculate Precision, Recall
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
        
        # Calculate Average Precision (AP)
        average_precision = average_precision_score(y_true, y_scores)
        
        # Calculate F1 Score as the maximum F1 Score
        f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-6)
        f1 = np.max(f1_vals)
        
        # Calculate mean precision and recall
        precision = np.mean(precision_vals)
        recall = np.mean(recall_vals)
    
    mean_error = np.mean(data['errors']) if data['errors'] else 0
    std_error = np.std(data['errors']) if data['errors'] else 0
    
    csv_data.append({
        'Class': cls,
        'Mean Centroid Error (px)': round(mean_error, 2),
        'Std Centroid Error (px)': round(std_error, 2),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1 Score': round(f1, 4),
        'Average Precision (AP@0.5)': round(average_precision, 4)
    })
    
    # Plot Centroid Error Histogram
    if data['errors']:
        plt.figure(figsize=(8, 5))
        sns.histplot(data['errors'], bins=30, kde=True, color='skyblue')
        plt.title(f'Centroid Distance Error Histogram for Class {cls}')
        plt.xlabel('Error (pixels)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(histograms_dir, f'class_{cls}_error_histogram.jpg'))
        plt.close()
    
    # Plot Precision-Recall Curve
    if len(y_true) > 0 and len(y_scores) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(recall_vals, precision_vals, marker='.', label=f'Class {cls} AP@0.5={average_precision:.2f}')
        plt.title(f'Precision-Recall Curve for Class {cls}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, f'class_{cls}_precision_recall.jpg'))
        plt.close()

# Save metrics to CSV
df = pd.DataFrame(csv_data)
df.to_csv(CSV_PATH, index=False)
print(f"\nMetrics saved to {CSV_PATH}")

# ================================
# Compute and Print mAP@0.5
# ================================

# Calculate mAP@0.5 as the mean of APs across all classes
map_05 = df['Average Precision (AP@0.5)'].mean()
print(f"\nMean Average Precision at IoU=0.5 (mAP@0.5): {map_05:.4f}")

# Optionally, add mAP@0.5 to the CSV
map_row = pd.DataFrame({
    'Class': ['mAP@0.5'],
    'Mean Centroid Error (px)': ['-'],
    'Std Centroid Error (px)': ['-'],
    'Precision': ['-'],
    'Recall': ['-'],
    'F1 Score': ['-'],
    'Average Precision (AP@0.5)': [round(map_05, 4)]
})
df = pd.concat([df, map_row], ignore_index=True)
df.to_csv(CSV_PATH, index=False)
print("mAP@0.5 added to the CSV file.")

# ================================
# Plot Aggregate Precision-Recall Curve
# ================================

plt.figure(figsize=(8, 5))
for cls in metrics.keys():
    y_true = metrics[cls]['y_true']
    y_scores = metrics[cls]['y_scores']
    if len(y_true) == 0 or len(y_scores) == 0:
        continue
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    plt.plot(recall_vals, precision_vals, marker='.', label=f'Class {cls} AP@0.5={average_precision:.2f}')
plt.title('Precision-Recall Curve per Class')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'precision_recall_curve_per_class.jpg'))
plt.close()

print("All histograms and plots have been saved.")

# ================================
# Summary
# ================================

print("\n=== Summary of Metrics ===")
print(df)

# ================================
# Optional: Model Conversion
# ================================

# Define overall accuracy threshold (optional)
OVERALL_ACCURACY_THRESHOLD = 0.60  # 60% F1 Score

# Check if all classes meet the accuracy threshold
all_passed = True
for _, row in df.iterrows():
    if row['Class'] == 'mAP@0.5':
        continue  # Skip mAP row
    if row['F1 Score'] < OVERALL_ACCURACY_THRESHOLD:
        print(f"Class {row['Class']} failed to meet the F1 Score threshold with {row['F1 Score']:.2f}.")
        all_passed = False

if all_passed:
    print("\nAll classes passed the F1 Score threshold.")
    # Example: Export to ONNX
    
    model.export(format="onnx")
    print(f"Model exported to ONNX at {MODEL_PATH}")
else:
    print("\nSome classes did not pass the F1 Score threshold. Model conversion aborted.")
