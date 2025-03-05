# MNIST Classification using Naive Bayes, ANN, and CNN

This repository presents an **MNIST digit classification** project, where various machine learning and deep learning techniques are applied. The goal is to classify handwritten digits from the MNIST dataset using **Naive Bayes**, **Artificial Neural Networks (ANN)**, and **Convolutional Neural Networks (CNN)**. Additionally, a **comparative analysis** is performed to evaluate the performance of each method based on different parameters.

## Project Overview

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9), with a total of 70,000 images split into 60,000 training images and 10,000 test images. The objective of this project is to explore how different models and parameter choices impact classification performance.

### Models Implemented:
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem. NB classifiers used BernouliNB, MultinomialNB, GuassianNB
- **Artificial Neural Network (ANN)**: A simple feed-forward neural network for classification. Optimizers, early stopping, Activation functions, Weight Initializers
- **Convolutional Neural Network (CNN)**: A deep learning model specifically designed to process image data.

### Comparative Analysis:
- **Activation Functions**: Comparison of various activation functions such as ReLU, Sigmoid, and Tanh.
- **Weight Initializers**: Impact of different weight initialization methods like Random, Xavier, and He initialization.
- **Hyperparameter Tuning**: Analysis of how varying learning rates, batch sizes, and network architectures influence model performance.

## Project Setup

### Prerequisites:
- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `tensorflow` or `keras`
