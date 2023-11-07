# Machine-Learning-project-useing-t-SNE
# MNIST Dataset Exploration with t-SNE and Random Forest

This Python script is designed to explore the MNIST dataset, a well-known collection of hand-written digit images, using dimensionality reduction techniques like Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE). It also demonstrates how to apply k-Nearest Neighbors (k-NN) sampling and train a Random Forest classifier on the reduced and sampled data.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Code Description](#code-description)
- [Results](#results)
- [License](#license)

## Introduction

The MNIST dataset is a popular dataset for practicing machine learning and image classification tasks. This code aims to provide insights into the dataset by performing the following tasks:

1. Load a subset of the MNIST dataset in CSV format.
2. Reduce the dataset's dimensionality using PCA.
3. Apply t-SNE to further reduce dimensionality and visualize the data in 2D.
4. Perform k-NN sampling on the t-SNE-transformed data.
5. Train a Random Forest classifier on the k-NN sampled data and evaluate its accuracy.

## Prerequisites

Before running the code, you should have the following prerequisites in place:

- Python 3.x
- Required Python libraries: numpy, pandas, scikit-learn (sklearn), matplotlib

## Getting Started

1. Clone or download this repository to your local machine.
2. Open a terminal or command prompt.
3. Navigate to the directory containing the script.
4. Run the following command to execute the code:
   
   ```
   python mnist_tSNE_RandomForest.py
   ```

   Make sure to adjust the `subset_size` variable to specify the number of samples you want to load from the MNIST dataset.

## Code Description

The code file (`mnist_tSNE_RandomForest.py`) is organized as follows:

- Import necessary Python libraries for data manipulation, dimensionality reduction, and machine learning.
- Load a subset of the MNIST dataset and preprocess it.
- Perform PCA to reduce dimensionality.
- Implement t-SNE for further dimensionality reduction and visualization.
- Define functions for k-NN sampling and data visualization.
- Split the data into training and testing sets.
- Train a Random Forest classifier on the k-NN sampled data.
- Evaluate the model's accuracy.

## Results

Upon running the code, it will display a scatter plot of the t-SNE-transformed MNIST data, color-coded by their labels. Additionally, it will print the accuracy of the Random Forest classifier on the test data.

## License

This code is provided under the MIT License. You are free to use, modify, and distribute it as needed. Please review the `LICENSE` file in this repository for more details.
