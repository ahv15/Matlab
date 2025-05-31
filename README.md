# MATLAB Machine Learning Algorithms

## Overview

This repository contains a comprehensive collection of machine learning algorithms implemented from scratch in MATLAB. All algorithms are modular, well-documented, and organized into logical categories for easy navigation and use.

## Repository Structure

```
├── algorithms/
│   ├── optimization/           # Gradient descent optimization algorithms
│   │   ├── adamOptimizer.m
│   │   ├── adamDemo.m
│   │   ├── rmsPropOptimizer.m
│   │   ├── momentumOptimizer.m
│   │   ├── batchGradientDescent.m
│   │   ├── stochasticGradientDescent.m
│   │   └── miniBatchGradientDescent.m
│   ├── regression/             # Regression algorithms
│   │   ├── linearRegression.m
│   │   └── logisticRegression.m
│   ├── clustering/             # Clustering algorithms
│   │   ├── kMeansClustering.m
│   │   └── fuzzyCMeansClustering.m
│   └── classification/         # Classification algorithms
│       └── naiveBayesClassifier.m
├── utils/                      # Utility functions
│   ├── featureNormalize.m      # Feature normalization
│   └── dataUtils.m             # Data processing utilities
└── README.md                   # This file
```

## Algorithm Descriptions

### Optimization Algorithms

#### Gradient Descent Variants
- **Batch Gradient Descent**: Standard gradient descent using entire dataset
- **Stochastic Gradient Descent (SGD)**: Uses one sample at a time for faster convergence
- **Mini-batch Gradient Descent**: Balances batch and stochastic approaches
- **Momentum**: Accelerates gradients in relevant direction and dampens oscillations
- **RMSProp**: Adapts learning rate by dividing gradient by running average of magnitudes
- **Adam**: Combines advantages of AdaGrad and RMSProp with bias correction

### Regression Algorithms

#### Linear Regression
- Implements linear regression with gradient descent optimization
- Supports polynomial features and automatic feature normalization
- Provides training and testing performance metrics

#### Logistic Regression
- Binary classification using logistic regression
- Sigmoid activation function with gradient descent training
- Returns accuracy and cost metrics for evaluation

### Clustering Algorithms

#### K-Means Clustering
- Partitions data into k clusters using iterative centroid updates
- Supports 2D visualization with different colors for each cluster
- Returns final centroids and cluster assignments

#### Fuzzy C-Means Clustering
- Soft clustering where points can belong to multiple clusters
- Uses fuzzy membership values rather than hard assignments
- Supports customizable fuzziness parameter and convergence criteria

### Classification Algorithms

#### Naive Bayes Classifier
- Gaussian Naive Bayes for continuous features
- Supports multi-class classification problems
- Assumes feature independence and normal distribution

## Usage Instructions

### Prerequisites
- MATLAB R2018b or later
- Statistics and Machine Learning Toolbox (recommended)

### Basic Usage

1. **Add paths to your MATLAB workspace:**
```matlab
addpath('algorithms/optimization');
addpath('algorithms/regression');
addpath('algorithms/clustering');
addpath('algorithms/classification');
addpath('utils');
```

2. **Example: Using Adam Optimizer**
```matlab
% Load and preprocess data
data = readtable('your_data.csv');
[X, Y] = preprocessData(data);

% Normalize features
[XNorm, mu, sigma] = featureNormalize(X);
XNorm = [XNorm, ones(size(XNorm, 1), 1)]; % Add bias term

% Train using Adam optimizer
[theta, costHistory] = adamOptimizer(XNorm, Y, 0.002, 100);

% Make predictions
predictions = XNorm * theta;
```

3. **Example: K-Means Clustering**
```matlab
% Generate or load 2D data
data = randn(100, 2);

% Perform clustering
[centroids, assignments, cost] = kMeansClustering(data, 3, 20, true);
```

4. **Example: Naive Bayes Classification**
```matlab
% Load iris dataset
load fisheriris;

% Split data
trainRatio = 0.7;
[trainX, testX, trainY, testY] = splitData(meas, species, trainRatio);

% Train classifier
[trainAcc, testAcc, model] = naiveBayesClassifier(trainX, trainY, testX, testY);
```

### Running Demos

The repository includes demonstration scripts:

```matlab
% Run Adam optimizer demo
run('algorithms/optimization/adamDemo.m');
```

## Utility Functions

### Feature Normalization
```matlab
[XNorm, mu, sigma] = featureNormalize(X);
```
Normalizes features by subtracting mean and dividing by standard deviation.

### Data Splitting
```matlab
[trainX, testX, trainY, testY] = splitData(X, Y, 0.7);
```
Randomly splits data into training and testing sets.

### Mean Squared Error
```matlab
mse = calculateMSE(predictions, actual);
```
Calculates mean squared error for regression problems.

## Algorithm Parameters

### Optimization Algorithms
- **Learning Rate (alpha)**: Controls step size (typical range: 0.001 - 0.1)
- **Iterations**: Number of training iterations (typical range: 100 - 5000)
- **Momentum (gamma)**: Momentum parameter for momentum-based methods (typical: 0.9)
- **Beta parameters**: For Adam and RMSProp (beta1=0.9, beta2=0.999)

### Clustering Algorithms
- **k**: Number of clusters for K-means
- **Fuzziness**: Fuzziness parameter for Fuzzy C-means (typical: 2.0)
- **Tolerance**: Convergence criteria (typical: 1e-5)
