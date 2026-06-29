import os
import shutil
from sklearn.model_selection import train_test_split

# Source dataset path
dataset_dir = r"D:\Desktop\Image classification\Animals"

# Destination split dataset path
base_dir = r"D:\Desktop\Image classification\dataset_split"

# Create folders
for folder in ["train", "validation", "test"]:
    for class_name in ["cats", "dogs", "snakes"]:
        os.makedirs(os.path.join(base_dir, folder, class_name), exist_ok=True)

# Function to split and copy images
def split_and_copy(class_name, train_size=0.7, val_size=0.2, test_size=0.1):
    class_dir = os.path.join(dataset_dir, class_name)
    images = os.listdir(class_dir)

    # Train / temp split
    train_files, temp_files = train_test_split(images, test_size=(1 - train_size), random_state=42)
    # Validation / Test split
    val_files, test_files = train_test_split(temp_files, test_size=(test_size / (test_size + val_size)), random_state=42)

    # Copy files
    for img in train_files:
        shutil.copy(os.path.join(class_dir, img), os.path.join(base_dir, "train", class_name, img))
    for img in val_files:
        shutil.copy(os.path.join(class_dir, img), os.path.join(base_dir, "validation", class_name, img))
    for img in test_files:
        shutil.copy(os.path.join(class_dir, img), os.path.join(base_dir, "test", class_name, img))

# Run for each class
for cls in ["cats", "dogs", "snakes"]:
    split_and_copy(cls)

print("✅ Dataset successfully split into train/validation/test!")
