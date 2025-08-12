# Import libraries
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, TimeDistributed, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time


# ===============================
# CONFIGURATION
# ===============================
main_source_dir = "/content/drive/MyDrive/SL8H"        # original dataset folder
masked_output_dir = "/content/drive/MyDrive/SL8H_masked"  # output folder for masked images

classes = ['Control', 'Medium', 'High']
time_points = ['T1', 'T2']

# Enhanced masking process

def mask_rice_plant_hybrid(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # -----------------------------
    # Improve contrast using CLAHE
    # -----------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # -----------------------------
    # HSV Mask (tuned range)
    # -----------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower_green, upper_green)

    # -----------------------------
    # ExG - ExR Index
    # -----------------------------
    img_float = img.astype(np.float32)
    B, G, R = cv2.split(img_float)
    Rn = R / 255.0
    Gn = G / 255.0
    Bn = B / 255.0

    ExG = 2 * Gn - Rn - Bn
    ExR = 1.4 * Rn - Gn
    ExGR = ExG - ExR  # This improves separation from soil/background

    # Normalize to 0–255
    ExGR_scaled = ((ExGR - ExGR.min()) / (ExGR.max() - ExGR.min()) * 255).astype(np.uint8)

    _, mask_exgr = cv2.threshold(ExGR_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # -----------------------------
    # Combine Masks (for better coverage)
    # -----------------------------
    combined_mask = cv2.bitwise_or(mask_hsv, mask_exgr)

    # -----------------------------
    # Morphological Cleaning (larger kernel)
    # -----------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # -----------------------------
    # Apply Mask
    # -----------------------------
    masked_img = cv2.bitwise_and(img, img, mask=combined_mask)

    return masked_img

# ===============================
# APPLY HYBRID MASKING TO DATASET
# ===============================
def process_dataset_hybrid(src_dir, out_dir):
    for cls in classes:
        for tp in time_points:
            src_path = os.path.join(src_dir, cls, tp)
            dst_path = os.path.join(out_dir, cls, tp)
            os.makedirs(dst_path, exist_ok=True)

            image_files = [f for f in os.listdir(src_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            for img_file in tqdm(image_files, desc=f"{cls}/{tp}"):
                src_img_path = os.path.join(src_path, img_file)
                masked_img = mask_rice_plant_hybrid(src_img_path)
                if masked_img is not None:
                    save_path = os.path.join(dst_path, img_file)
                    cv2.imwrite(save_path, masked_img)

# ===============================
# RUN PROCESSING
# ===============================
print("Applying Hybrid HSV + ExG vegetation masking...")
process_dataset_hybrid(main_source_dir, masked_output_dir)
print(f"Hybrid masking complete. Saved to: {masked_output_dir}")

# ===============================
# LOAD MASKED DATA INTO SEQUENCES
# ===============================
def load_sequences(base_path, img_size=(224, 224)):
    X, y = [], []
    for label, cls in enumerate(classes):
        for tp in time_points:
            img_dir = os.path.join(base_path, cls, tp)
            image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])
            for img_file in image_files:
                img_path = os.path.join(img_dir, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                X.append(img / 255.0)  # normalize
                y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

print("Loading masked dataset...")
X, y = load_sequences(masked_output_dir)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Loaded: {X.shape[0]} images of shape {X.shape[1:]}")
print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print(f"y_train distribution: {np.bincount(y_train)}")


# ===============================
# MODEL DEFINITION
# ===============================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, TimeDistributed, GlobalAveragePooling2D, LSTM, Dense, Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import time

base_cnn = DenseNet121(weights='imagenet', include_top=False)
base_cnn.trainable = False

model = Sequential([
    Input(shape=(2, 224, 224, 3)),  # For 2 time points
    TimeDistributed(base_cnn),
    TimeDistributed(GlobalAveragePooling2D()),
    LSTM(128),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # Adjust number of classes as needed
])


model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
checkpoint =  ModelCheckpoint("model_epoch_{epoch:02d}.h5", save_weights_only=False)

# ---------------------------
# MODEL TRAINING
# ---------------------------
# Start the timer
start_time = time.time()

callbacks = [
    #EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("/content/best_model.keras", monitor='val_accuracy', save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=8,
    callbacks=callbacks
)

# ---------------------------
# MODEL EVALUATION
# ---------------------------

val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc:.2f}")

# Calculate the elapsed time
elapsed_time = time.time() - start_time
print(f"Time taken to train the CNN-LSTM hybrid model: {elapsed_time:.2f} seconds")


# Save full-sequence report
def save_report_and_plot(y_true, y_pred, title, filename_prefix):
    report_txt = classification_report(y_true, y_pred, target_names=classes)
    with open(f"/content/drive/MyDrive/Reports/{filename_prefix}_report.txt", "w") as f:
        f.write(report_txt)
    print(report_txt)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(title)
    plt.grid(False)
    plt.savefig(f"/content/drive/MyDrive/FIGURES/{filename_prefix}_confusion_matrix_SL8H.png")
    plt.show()


# Time-point specific classification
def evaluate_per_timepoint(X_val_seq, y_val_onehot, time_index, label="T1"):
    print(f"Evaluating for {label} only")

    # Extract data for the given time point
    X_single = X_val_seq[:, time_index, :, :, :]  
    X_single = np.expand_dims(X_single, axis=1)  
    y_true = np.argmax(y_val_onehot, axis=1)

    # Define the model for a single frame
    model_single_frame = Sequential([
        TimeDistributed(DenseNet121(weights='imagenet', include_top=False), input_shape=(2, 224, 224, 3)),  # Use pre-trained VGG16 base
        TimeDistributed(GlobalAveragePooling2D()),
        LSTM(128),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])

    model_single_frame.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Make predictions for this time point
    y_pred_probs = model_single_frame.predict(X_single)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    # Save the confusion matrix and report
    save_report_and_plot(y_true, y_pred_classes, f"Confusion Matrix ({label})", f"{label}")


# EVALUATE PER TIME POINT

import os

os.makedirs("/content/drive/MyDrive/Reports", exist_ok=True)

#evaluate_per_timepoint(X_val, y_val, 0, "T0")
evaluate_per_timepoint(X_val, y_val, 1, "T1")
evaluate_per_timepoint(X_val, y_val, 2, "T2")

# Define History Info
def print_history_info(history):
    """Prints training and validation metrics from a Keras history object."""
    print("\nTraining History:")
    for epoch in range(len(history.history['loss'])):
        print(f"Epoch {epoch+1}:")
        print(f"  Loss: {history.history['loss'][epoch]:.4f}")
        print(f"  Accuracy: {history.history['accuracy'][epoch]:.4f}")
        if 'val_loss' in history.history:
            print(f"  Val Loss: {history.history['val_loss'][epoch]:.4f}")
        if 'val_accuracy' in history.history:
            print(f"  Val Accuracy: {history.history['val_accuracy'][epoch]:.4f}")


# Plot Training & Validation Accuracy

plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='x')
plt.title('Model Accuracy on BLB Detection for SL8H')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='x')
plt.title('Model Loss on BLB Detection for SL8H')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save and show
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/Reports/Training_Validation_Curves_SL8H")
plt.show()


# Save the Model as a Reusable File
#model.save(OUTPUT_MODEL_NAME)

# Save model and architecture to a single file
save_model_dir = "/content/drive/MyDrive/Reports/Models/"
#model.save(save_model_dir + "CNNR_SavedModel", save_format='tf')
model.save(save_model_dir + "BLB_Detection_SL8H.keras") # Save model with .keras extension
print("Saved model to disk", save_model_dir)


