# Surface-Crack-Detection-using-Deep-Learning-CNN-Transfer-Learning-in-PyTorch-
Surface Crack Detection using Deep Learning (PyTorch)

*Project Overview*

This project implements a complete deep learning pipeline for automatic surface crack detection using convolutional neural networks in PyTorch.
The goal was not only to reach high accuracy, but to build a clean and structured experimentation workflow: proper train/test split, validation strategy, threshold analysis, regularization techniques, transfer learning, and fine-tuning.
The dataset is relatively well-structured and balanced, which makes it a good case study to understand model behavior when the classification boundary is visually clear.

*Dataset*

The dataset used in this project is publicly available on Kaggle:
Surface Crack Detection Dataset
https://www.kaggle.com/datasets/arunrk7/surface-crack-detection

40,000 RGB images
20,000 Positive (crack)
20,000 Negative (no crack)
Balanced binary classification
Images were resized to 224×224 and normalized using ImageNet statistics to ensure compatibility with transfer learning models.

*What Was Implemented*

*1- Baseline CNN*

- A custom convolutional neural network was built from scratch:
- 3 convolutional blocks (Conv → ReLU → MaxPool)
- Fully connected classifier
- BCEWithLogitsLoss for binary classification
- Threshold analysis (0.3, 0.5, 0.7)

Even in the first epoch, the baseline achieved very high performance (~99% accuracy), which suggests that the dataset is relatively easy to separate. Crack images typically show large and visually distinct defect patterns, while negative samples are clean surfaces with no visible damage.
This allowed the model to converge very quickly.

*2️- Regularized CNN (Dropout + Weight Decay)*

To study generalization behavior, I added:

- Dropout in the classifier
- Weight decay (L2 regularization)
- Validation split
- Early stopping

The regularized model slightly improved generalization and reduced the training/validation gap.

*3️- Transfer Learning (ResNet18)*

A pretrained ResNet18 model was used:

- Backbone initially frozen
- Final fully connected layer replaced
- Then partial fine-tuning (unfreezing layer4)
- Transfer learning converged very quickly and provided marginal performance improvements.

Given the already strong baseline, the transfer learning and fine-tuning experiments were mainly conducted to practice and understand standard deep learning engineering workflows, rather than because the task strictly required additional model complexity.

*Engineering Takeaways*

- Proper threshold selection affects the FP/FN trade-off.
- Even simple CNNs can perform extremely well on clearly separable datasets.
- Regularization helps stabilize training.
- Transfer learning is powerful but not always necessary when the dataset is simple.
- Structured experimentation (validation split + early stopping) is essential for reliable evaluation.

*Repository Structure*

project_root/
│
├── src/
│   ├── train.py
│   ├── model.py
│   ├── evaluation.py
│   └── utils.py
│
└── results/
    └── results.pdf


src/ → full training and evaluation pipeline
results/ → PDF report with metrics and analysis

AUTHOR
Rodrigo Driemeier dos Santos
EESC – University of São Paulo (USP), São Carlos, Brazil — Mechatronics Engineering
École Centrale de Lille, France — Generalist Engineering

📧 rodrigodriemeier@usp.br

🔗 https://www.linkedin.com/in/rodrigo-driemeier-dos-santos-a7698633b/

Thanks for checking out the project :)
