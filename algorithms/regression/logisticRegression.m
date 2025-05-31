function [theta, trainAccuracy, testAccuracy, trainCost, testCost] = logisticRegression(trainX, trainY, testX, testY, alpha, numIterations)
    % LOGISTICREGRESSION Logistic regression classifier
    %
    % Purpose:
    %   Implements logistic regression for binary classification using
    %   gradient descent optimization. Suitable for classification tasks
    %   with two classes (0 and 1).
    %
    % Input:
    %   trainX        - Training feature matrix (m x n)
    %   trainY        - Training labels (m x 1) - should be 0 or 1
    %   testX         - Test feature matrix (k x n) 
    %   testY         - Test labels (k x 1) - should be 0 or 1
    %   alpha         - Learning rate (default: 0.3)
    %   numIterations - Number of iterations (default: 3000)
    %
    % Output:
    %   theta        - Learned parameter vector
    %   trainAccuracy - Training accuracy percentage
    %   testAccuracy  - Test accuracy percentage  
    %   trainCost    - Final training cost
    %   testCost     - Final test cost
    %
    % Example:
    %   [theta, trainAcc, testAcc] = logisticRegression(Xtrain, Ytrain, Xtest, Ytest);
    
    % Set default parameters
    if nargin < 5
        alpha = 0.3;
    end
    if nargin < 6
        numIterations = 3000;
    end
    
    % Add bias term to training data
    trainXBias = [trainX, ones(size(trainX, 1), 1)];
    
    % Initialize parameters
    theta = zeros(size(trainXBias, 2), 1);
    m = size(trainXBias, 1);
    
    % Train using gradient descent
    for i = 1:numIterations
        % Sigmoid function
        g = 1 ./ (1 + exp(-(trainXBias * theta)));
        
        % Update parameters
        theta = theta - (alpha / m) * trainXBias' * (g - trainY);
    end
    
    % Add bias term to test data
    testXBias = [testX, ones(size(testX, 1), 1)];
    
    % Calculate predictions and costs
    trainProbabilities = 1 ./ (1 + exp(-(trainXBias * theta)));
    testProbabilities = 1 ./ (1 + exp(-(testXBias * theta)));
    
    % Calculate cost (logistic loss)
    trainCost = -(1/m) * sum(trainY .* log(trainProbabilities + eps) + ...
                             (1 - trainY) .* log(1 - trainProbabilities + eps));
    
    mTest = size(testXBias, 1);
    testCost = -(1/mTest) * sum(testY .* log(testProbabilities + eps) + ...
                               (1 - testY) .* log(1 - testProbabilities + eps));
    
    % Calculate accuracy
    trainPredictions = trainProbabilities >= 0.5;
    testPredictions = testProbabilities >= 0.5;
    
    trainAccuracy = (sum(trainPredictions == trainY) / m) * 100;
    testAccuracy = (sum(testPredictions == testY) / mTest) * 100;
end

function accuracy = calculateClassificationAccuracy(predictions, actual, className)
    % CALCULATECLASSIFICATIONACCURACY Calculate accuracy for specific class
    %
    % Purpose:
    %   Helper function to calculate classification accuracy for a specific
    %   class in binary classification problems.
    %
    % Input:
    %   predictions - Predicted class labels
    %   actual      - Actual class labels  
    %   className   - Class to calculate accuracy for (0 or 1)
    %
    % Output:
    %   accuracy - Accuracy percentage for the specified class
    
    classIndices = actual == className;
    if sum(classIndices) == 0
        accuracy = 0;
        return;
    end
    
    correctPredictions = sum(predictions(classIndices) == actual(classIndices));
    totalClassSamples = sum(classIndices);
    accuracy = (correctPredictions / totalClassSamples) * 100;
end
