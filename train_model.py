import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Settings
DATA_DIR = "data"
MODEL_DIR = "sign project/sign project/Data/model"
IMG_SIZE = 224 # Standard for many models

def train():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} folder not found!")
        return

    images = []
    labels = []
    class_names = sorted(os.listdir(DATA_DIR))
    
    print(f"Found classes: {class_names}")

    for idx, class_name in enumerate(class_names):
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(idx)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    if not images:
        print("No images found to train on!")
        return

    X = np.array(images) / 255.0
    y = to_categorical(np.array(labels), num_classes=len(class_names))

    # Simple Model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Starting training...")
    model.fit(X, y, epochs=5, batch_size=32)

    # Save Model and Labels
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "keras_model.h5"))
    
    with open(os.path.join(MODEL_DIR, "labels.txt"), "w") as f:
        for i, name in enumerate(class_names):
            f.write(f"{i} {name}\n")

    print(f"\nSuccess! Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    train()
