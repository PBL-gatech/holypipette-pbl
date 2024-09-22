import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(image_folder, label_folder, output_folder, split_ratio=0.8):
    """
    Split the dataset of images and YOLO annotations into training and validation sets.
    
    :param image_folder: Path to the folder containing images.
    :param label_folder: Path to the folder containing YOLO label files (.txt).
    :param output_folder: Path where the split dataset will be saved.
    :param split_ratio: Proportion of the dataset to include in the training set (e.g., 0.8 for 80% training, 20% validation).
    """
    
    # Ensure the output folders for images and labels exist
    train_img_dir = os.path.join(output_folder, 'train', 'images')
    val_img_dir = os.path.join(output_folder, 'val', 'images')
    train_label_dir = os.path.join(output_folder, 'train', 'labels')
    val_label_dir = os.path.join(output_folder, 'val', 'labels')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # Get all image files in the folder (without extensions)
    images = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    # Corresponding label files should have the same filename (with .txt extension)
    labels = [f.replace(os.path.splitext(f)[1], '.txt') for f in images]  # Replace image extension with .txt

    # Split the dataset into train and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, train_size=split_ratio, random_state=42)

    # Copy the images and labels to the appropriate directories
    for img_file, lbl_file in zip(train_images, train_labels):
        shutil.copy(os.path.join(image_folder, img_file), os.path.join(train_img_dir, img_file))  # Copy image to train
        shutil.copy(os.path.join(label_folder, lbl_file), os.path.join(train_label_dir, lbl_file))  # Copy label to train

    for img_file, lbl_file in zip(val_images, val_labels):
        shutil.copy(os.path.join(image_folder, img_file), os.path.join(val_img_dir, img_file))  # Copy image to val
        shutil.copy(os.path.join(label_folder, lbl_file), os.path.join(val_label_dir, lbl_file))  # Copy label to val

    print(f"Dataset split complete! Train set: {len(train_images)} images, Validation set: {len(val_images)} images")


if __name__ == '__main__':
    # Paths to the folders
    image_folder = 'convert/P_DET_IMAGES'  # Folder with images
    label_folder = 'convert/output'  # Folder with YOLO .txt labels
    output_folder = 'split/'  # Folder where the split data will be saved
    
    # Run the split (adjust the split ratio if necessary)
    split_dataset(image_folder, label_folder, output_folder, split_ratio=0.8)
