function mse = calculateMSE(predictions, actual)
    % CALCULATEMSE Calculates Mean Squared Error
    %
    % Purpose:
    %   Computes the mean squared error between predictions and actual values.
    %   This is a common evaluation metric for regression algorithms.
    %
    % Input:
    %   predictions - Vector of predicted values
    %   actual      - Vector of actual/target values
    %
    % Output:
    %   mse - Mean squared error value
    %
    % Example:
    %   mse = calculateMSE(modelPredictions, testLabels);
    
    m = length(actual);
    mse = (1 / m) * sum((predictions - actual).^2);
end

function [trainX, testX, trainY, testY] = splitData(X, Y, trainRatio)
    % SPLITDATA Randomly splits data into training and test sets
    %
    % Purpose:
    %   Randomly divides the dataset into training and testing portions
    %   while maintaining the same ratio across features and labels.
    %
    % Input:
    %   X          - Feature matrix
    %   Y          - Label vector
    %   trainRatio - Fraction of data to use for training (0 < trainRatio < 1)
    %
    % Output:
    %   trainX - Training feature matrix
    %   testX  - Testing feature matrix
    %   trainY - Training label vector
    %   testY  - Testing label vector
    %
    % Example:
    %   [trainX, testX, trainY, testY] = splitData(features, labels, 0.7);
    
    m = size(X, 1);
    trainSize = round(m * trainRatio);
    
    idx = randperm(m, trainSize);
    trainX = X(idx, :);
    trainY = Y(idx, :);
    
    testX = X;
    testY = Y;
    testX(idx, :) = [];
    testY(idx, :) = [];
end
