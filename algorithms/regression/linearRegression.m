function [theta, trainMSE, testMSE] = linearRegression(trainX, trainY, testX, testY, alpha, numIterations)
    % LINEARREGRESSION Linear regression with gradient descent
    %
    % Purpose:
    %   Implements linear regression using gradient descent optimization.
    %   Automatically handles feature normalization and provides training
    %   and testing performance metrics.
    %
    % Input:
    %   trainX        - Training feature matrix (m x n)
    %   trainY        - Training target values (m x 1)
    %   testX         - Test feature matrix (k x n)
    %   testY         - Test target values (k x 1)
    %   alpha         - Learning rate (default: 0.3)
    %   numIterations - Number of iterations (default: 3000)
    %
    % Output:
    %   theta    - Learned parameter vector
    %   trainMSE - Mean squared error on training set
    %   testMSE  - Mean squared error on test set
    %
    % Example:
    %   [theta, trainErr, testErr] = linearRegression(Xtrain, Ytrain, Xtest, Ytest);
    
    % Set default parameters
    if nargin < 5
        alpha = 0.3;
    end
    if nargin < 6
        numIterations = 3000;
    end
    
    % Add polynomial features (squared terms)
    trainXPoly = [trainX, trainX.^2];
    testXPoly = [testX, testX.^2];
    
    % Normalize features
    [trainXNorm, mu, sigma] = featureNormalize(trainXPoly);
    
    % Add bias term
    trainXNorm = [trainXNorm, ones(size(trainXNorm, 1), 1)];
    
    % Initialize parameters
    theta = zeros(size(trainXNorm, 2), 1);
    m = size(trainXNorm, 1);
    
    % Train using gradient descent
    for i = 1:numIterations
        gradJ = trainXNorm' * (trainXNorm * theta - trainY);
        theta = theta - (alpha / m) * gradJ;
    end
    
    % Normalize test data using training statistics
    testXNorm = (testXPoly - mu) ./ sigma;
    testXNorm = [testXNorm, ones(size(testXNorm, 1), 1)];
    
    % Calculate performance metrics
    trainPredictions = trainXNorm * theta;
    testPredictions = testXNorm * theta;
    
    trainMSE = calculateMSE(trainPredictions, trainY);
    testMSE = calculateMSE(testPredictions, testY);
end
