# Pose Classification Using Fine-Tuned MobileNetV3

This project implements a **human pose classification model** using **transfer learning** with a **fine-tuned MobileNetV3-Large** backbone.  
It was developed as part of an Artificial Neural Networks course and demonstrates a full deep-learning workflow: data preprocessing, model design, fine-tuning, training, and evaluation.

---

## Overview

The goal of this project is to classify human poses into multiple categories using a lightweight neural network.  
The model leverages **MobileNetV3-Large** pre-trained on ImageNet and adapts it for multi-class pose recognition through fine-tuning and additional dense layers.

This notebook showcases:
- End-to-end model creation using **TensorFlow/Keras**
- **Transfer learning** for efficient feature reuse
- Data preprocessing and **augmentation**
- Use of training **callbacks** (EarlyStopping, ModelCheckpoint, TensorBoard)
- **Evaluation** with classification reports and sample predictions

---

## Key Features

✅ Fine-tuned MobileNetV3-Large backbone  
✅ In-graph preprocessing and random data augmentation  
✅ Early stopping and checkpoint saving  
✅ TensorBoard logging for training visualization  
✅ Classification report and prediction visualization  

---

## Methodology

### **1. Data Preparation**
Images are stored in folders named after their class labels:

data/anndataset/archive
├── pose_name_1/
├── pose_name_2/
└── ...

A Pandas DataFrame is created containing image filepaths and corresponding labels.  
The dataset is split into **training, validation, and test sets** using `ImageDataGenerator` with an 80/20 train-test split and 20% validation subset.

### **2. Preprocessing and Augmentation**
Each image is resized to `224×224` and normalized.  
Random horizontal/vertical flips, rotations, zooms, and shifts are applied to improve generalization.

### **3. Model Architecture**
- **Backbone:** `MobileNetV3-Large` (ImageNet weights, `include_top=False`, `pooling='avg'`)
- **Head:**
  ```python
  Dense(256, activation='relu')
  Dropout(0.2)
  Dense(256, activation='relu')
  Dropout(0.2)
  Dense(43, activation='softmax')
  '''
- **Optimizer:** Adam (learning rate = 1e-5)
- **Loss:** Categorical cross-entropy
- **Metrics:** Accuracy

### 4. Fine-Tuning

The model uses **partial fine-tuning**: all layers before index **100** in the MobileNetV3 backbone are frozen, while the remaining top layers are unfrozen and trainable.  
This approach allows the model to adapt high-level feature representations to the pose dataset while preserving general visual features learned from ImageNet.

### 5. Callbacks

- **EarlyStopping:** Stops training when validation loss stops improving.  
- **ModelCheckpoint:** Saves the best model weights automatically.  
- **TensorBoard:** Logs metrics for visualization.

---

## Evaluation
After training, the model is evaluated on the test set.  
Results include:

- **Accuracy and F1-scores per class**
- **Classification report**
- **Confusion matrix visualization** (optional)
- **Sample predictions** with correct/incorrect labels color-coded.

---

## Results Summary

Exact metrics may vary depending on dataset composition and training epochs.
The notebook prints detailed per-class performance via classification_report() and demonstrates qualitative performance using randomly selected test samples.
