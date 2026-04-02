# Chest X-Ray Pneumonia Detection using EfficientNetB0

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

An end-to-end **binary classification** project for detecting **Pneumonia** from chest X-ray images using **Transfer Learning** with **EfficientNetB0** in TensorFlow/Keras.

##  Project Overview

This project implements a deep learning pipeline to classify chest X-ray images into two categories:
- **NORMAL**
- **PNEUMONIA**

The model uses **EfficientNetB0** pretrained on ImageNet with a two-phase training strategy (frozen backbone → fine-tuning) and data augmentation.

**Dataset**: Kaggle Chest X-Ray Pneumonia dataset  
**Framework**: TensorFlow 2.19 + Keras

##  Dataset Statistics

| Split   |Total Images  | NORMAL |PNEUMONIA  |Pneumonia %  |
|---------|--------------|--------|-----------|-------------|
| Train   | 5,216        | 1,341  | 3,875     | 74.3%       |
| Val     | 16           | 8      | 8         | 50.0%       |
| Test    | 624          | 234    | 390       | 62.5%       |

> **Note**: The training set is significantly imbalanced (more Pneumonia cases).

##  Key Features

- Pretrained **EfficientNetB0** backbone
- Aggressive data augmentation (rotation, shift, shear, zoom, flip)
- Two-stage training:
  - Phase 1: Frozen backbone (10 epochs, LR = 1e-3)
  - Phase 2: Full fine-tuning (15 epochs, LR = 1e-5)
- Reproducible training with fixed seed (`SEED = 42`)
- Google Colab + Google Drive ready
- Sample image visualization

##  Technologies Used

- Python 3
- TensorFlow 2.19
- Keras
- EfficientNetB0
- Matplotlib & Seaborn
- Google Colab (GPU)

