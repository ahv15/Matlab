% ADAM_DEMO Demonstrates Adam optimization algorithm
%
% Purpose:
%   This script demonstrates the Adam optimization algorithm on a 
%   life expectancy dataset. It loads data, preprocesses it, trains
%   a linear regression model using Adam optimizer, and evaluates
%   the results.
%
% Dependencies:
%   - utils/featureNormalize.m
%   - utils/dataUtils.m
%   - algorithms/optimization/adamOptimizer.m
%
% Dataset:
%   Life Expectancy Data.csv (should be in the root directory)

%% Clear workspace and add paths
clear;
clc;
addpath('utils');
addpath('algorithms/optimization');

%% Load and preprocess data
fprintf('Loading Life Expectancy dataset...\n');

% Load data (assumes CSV file is in root directory)
try
    data = readtable('Life Expectancy Data.csv');
    fprintf('Dataset loaded successfully.\n');
catch
    error('Could not load Life Expectancy Data.csv. Please ensure file exists.');
end

% Remove non-numeric columns
data = removevars(data, 'Country');
data = removevars(data, 'Year');
data = removevars(data, 'Status');

% Remove missing values
data = rmmissing(data);
dataArray = table2array(data);

fprintf('Data preprocessing completed.\n');
fprintf('Dataset size: %d samples, %d features\n', size(dataArray));

%% Split data into training and testing sets
Y = dataArray(:, 1);  % Life expectancy (target)
X = dataArray(:, 2:end);  % Features

% Add polynomial features (squared terms)
X = [X, X.^2];

% Split data (70% training, 30% testing)
trainRatio = 0.7;
trainSize = round(size(X, 1) * trainRatio);
idx = randperm(size(X, 1), trainSize);

trainX = X(idx, :);
trainY = Y(idx, :);
testX = X;
testY = Y;
testX(idx, :) = [];
testY(idx, :) = [];

fprintf('Training set: %d samples\n', size(trainX, 1));
fprintf('Test set: %d samples\n', size(testX, 1));

%% Normalize features
[trainXNorm, mu, sigma] = featureNormalize(trainX);

% Add bias term
trainXNorm = [trainXNorm, ones(size(trainXNorm, 1), 1)];

%% Train model using Adam optimizer
fprintf('\nTraining linear regression model with Adam optimizer...\n');

alpha = 0.002;
numIterations = 30;

[theta, costHistory] = adamOptimizer(trainXNorm, trainY, alpha, numIterations);

fprintf('Training completed.\n');
fprintf('Final cost: %.4f\n', costHistory(end));

%% Normalize test data and make predictions
% Apply same normalization as training data
testXNorm = (testX - mu) ./ sigma;
testXNorm = [testXNorm, ones(size(testXNorm, 1), 1)];

% Make predictions
predictions = testXNorm * theta;

%% Evaluate model performance
testMSE = calculateMSE(predictions, testY);
trainMSE = calculateMSE(trainXNorm * theta, trainY);

fprintf('\nModel Performance:\n');
fprintf('Training MSE: %.4f\n', trainMSE);
fprintf('Test MSE: %.4f\n', testMSE);

%% Plot cost history
figure;
plot(1:numIterations, costHistory, 'b-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Cost');
title('Adam Optimizer - Cost vs Iteration');
grid on;

%% Plot predictions vs actual
figure;
subplot(1, 2, 1);
scatter(testY, predictions, 'filled');
hold on;
plot([min(testY), max(testY)], [min(testY), max(testY)], 'r--', 'LineWidth', 2);
xlabel('Actual Life Expectancy');
ylabel('Predicted Life Expectancy');
title('Test Set: Predictions vs Actual');
grid on;

subplot(1, 2, 2);
trainPredictions = trainXNorm * theta;
scatter(trainY, trainPredictions, 'filled');
hold on;
plot([min(trainY), max(trainY)], [min(trainY), max(trainY)], 'r--', 'LineWidth', 2);
xlabel('Actual Life Expectancy');
ylabel('Predicted Life Expectancy');
title('Training Set: Predictions vs Actual');
grid on;

fprintf('\nDemo completed successfully!\n');
