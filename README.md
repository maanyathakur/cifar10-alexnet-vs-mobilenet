# 🧠 CIFAR-10 Classification: AlexNet vs MobileNet-like CNNs

This project compares two deep learning models — **AlexNet** and a custom **MobileNet-like architecture** — built from scratch to classify images in the CIFAR-10 dataset using Convolutional Neural Networks (CNNs).

---

## 📌 Table of Contents

- [📖 About the Project](#-about-the-project)
- [📊 Dataset](#-dataset)
- [🔍 What is a CNN?](#-what-is-a-cnn)
- [🏗️ Architectures Used](#-architectures-used)
  - [AlexNet](#alexnet)
  - [MobileNet-like](#mobilenet-like)
- [⚙️ Training Details](#-training-details)
- [📈 Results](#-results)
- [🧪 Sample Predictions](#-sample-predictions)
- [📂 Repository Structure](#-repository-structure)
- [🚀 How to Run](#-how-to-run)
- [📝 License](#-license)

---

## 📖 About the Project

This project aims to implement and compare **two CNN models** on the CIFAR-10 dataset:
- **AlexNet** (simplified for 32×32 input)
- **MobileNet-like CNN** (depthwise separable convolution-based)

We evaluate both models on:
- Accuracy
- Precision, Recall, F1-Score
- Inference Time

---

## 📊 Dataset

**CIFAR-10** is a labeled subset of the 80 million tiny images dataset.  
- 60,000 images of 32x32 resolution in 10 classes.
- 50,000 images for training, 10,000 for testing.
- Classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

---

## 🔍 What is a CNN?

A **Convolutional Neural Network (CNN)** is a deep learning model especially effective for image-related tasks.  
It consists of layers that automatically detect patterns such as edges, textures, and object parts from pixel data.

---

## 🏗️ Architectures Used

### 🟦 AlexNet
- Based on the original AlexNet (2012) but scaled down for 32x32 inputs.
- **Architecture Summary**:
  - Multiple Conv + BatchNorm + ReLU layers
  - MaxPooling for downsampling
  - Fully Connected Layers (1024 → 512 → 10)
  - Dropout for regularization
Input → Conv2D → MaxPool → Conv2D → MaxPool → Conv2D → Conv2D → MaxPool → FC → Dropout → FC → Dropout → Output

---

### 🟩 MobileNet-like
- Inspired by **MobileNet v1**, optimized for speed.
- Uses **Depthwise Separable Convolutions** for fewer parameters and faster inference.
- Ends with **GlobalAveragePooling** before classification.
Input → Conv2D → [Depthwise + Pointwise] → MaxPool → [Depthwise + Pointwise] → GAP → Output

---

## ⚙️ Training Details

- **Epochs**: 50  
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  
- **Batch Size**: 128  
- **Early Stopping**: Yes (patience=5)

---

## 🚀 How to Run

1. Open [`AlexNet_vs_MobileNet.ipynb`](./AlexNet_vs_MobileNet.ipynb) in [Google Colab](https://colab.research.google.com/).
2. Run all cells sequentially.
3. Observe training, metrics, plots, and sample predictions.



