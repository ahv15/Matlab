function [trainAccuracy, testAccuracy, model] = naiveBayesClassifier(trainX, trainY, testX, testY, plotResults)
    % NAIVEBAYESCLASSIFIER Naive Bayes classifier implementation
    %
    % Purpose:
    %   Implements a Naive Bayes classifier assuming Gaussian distribution
    %   for continuous features. Works with multi-class classification problems
    %   and provides both univariate and multivariate implementations.
    %
    % Input:
    %   trainX      - Training feature matrix (m x n)
    %   trainY      - Training labels (m x 1) - categorical or cell array
    %   testX       - Test feature matrix (k x n) 
    %   testY       - Test labels (k x 1) - categorical or cell array
    %   plotResults - Whether to plot distributions (default: false)
    %
    % Output:
    %   trainAccuracy - Training accuracy percentage
    %   testAccuracy  - Test accuracy percentage
    %   model         - Trained model structure containing parameters
    %
    % Example:
    %   [trainAcc, testAcc, model] = naiveBayesClassifier(Xtrain, Ytrain, Xtest, Ytest);
    
    % Set default parameters
    if nargin < 5
        plotResults = false;
    end
    
    % Get unique classes
    if iscell(trainY)
        classes = unique(trainY);
    else
        classes = unique(trainY);
    end
    
    numClasses = length(classes);
    numFeatures = size(trainX, 2);
    
    % Initialize model structure
    model.classes = classes;
    model.priorProbs = zeros(numClasses, 1);
    model.means = zeros(numClasses, numFeatures);
    model.variances = zeros(numClasses, numFeatures);
    model.covMatrices = cell(numClasses, 1);
    
    % Calculate class statistics
    for i = 1:numClasses
        if iscell(trainY)
            classIndices = strcmp(trainY, classes{i});
        else
            classIndices = trainY == classes(i);
        end
        
        classData = trainX(classIndices, :);
        
        % Prior probabilities
        model.priorProbs(i) = sum(classIndices) / length(trainY);
        
        % Mean and variance for each feature
        model.means(i, :) = mean(classData, 1);
        model.variances(i, :) = var(classData, 1);
        
        % Covariance matrix for multivariate approach
        if size(classData, 1) > 1
            model.covMatrices{i} = cov(classData);
        else
            model.covMatrices{i} = eye(numFeatures);
        end
    end
    
    % Make predictions on training set
    trainPredictions = predictNaiveBayes(trainX, model);
    trainAccuracy = calculateAccuracy(trainPredictions, trainY);
    
    % Make predictions on test set
    testPredictions = predictNaiveBayes(testX, model);
    testAccuracy = calculateAccuracy(testPredictions, testY);
    
    % Plot distributions if requested
    if plotResults && numFeatures >= 2
        plotClassDistributions(trainX, trainY, model);
    end
    
    fprintf('Naive Bayes Classifier Results:\n');
    fprintf('Training Accuracy: %.2f%%\n', trainAccuracy);
    fprintf('Test Accuracy: %.2f%%\n', testAccuracy);
end

function predictions = predictNaiveBayes(X, model)
    % PREDICTNAIVEBAYES Make predictions using trained Naive Bayes model
    
    numSamples = size(X, 1);
    numClasses = length(model.classes);
    posteriorProbs = zeros(numSamples, numClasses);
    
    % Calculate posterior probabilities for each class
    for i = 1:numClasses
        % Calculate likelihood using Gaussian assumption
        likelihood = 1;
        for j = 1:size(X, 2)
            likelihood = likelihood .* normpdf(X(:, j), model.means(i, j), sqrt(model.variances(i, j)));
        end
        
        % Posterior = Prior * Likelihood (ignoring evidence as it's constant)
        posteriorProbs(:, i) = model.priorProbs(i) * likelihood;
    end
    
    % Find class with maximum posterior probability
    [~, maxIndices] = max(posteriorProbs, [], 2);
    
    % Convert indices to class labels
    if iscell(model.classes)
        predictions = model.classes(maxIndices);
    else
        predictions = model.classes(maxIndices);
    end
end

function accuracy = calculateAccuracy(predictions, actual)
    % CALCULATEACCURACY Calculate classification accuracy
    
    if iscell(predictions) && iscell(actual)
        correct = strcmp(predictions, actual);
    else
        correct = predictions == actual;
    end
    
    accuracy = (sum(correct) / length(actual)) * 100;
end

function plotClassDistributions(X, Y, model)
    % PLOTCLASSDISTRIBUTIONS Plot class distributions for visualization
    
    numFeatures = size(X, 2);
    numClasses = length(model.classes);
    
    if numFeatures >= 2
        figure;
        
        % Plot first two features
        colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k'];
        
        for i = 1:numClasses
            if iscell(Y)
                classIndices = strcmp(Y, model.classes{i});
            else
                classIndices = Y == model.classes(i);
            end
            
            classData = X(classIndices, :);
            
            if ~isempty(classData)
                colorIndex = mod(i - 1, length(colors)) + 1;
                scatter(classData(:, 1), classData(:, 2), 50, colors(colorIndex), 'filled');
                hold on;
            end
        end
        
        xlabel('Feature 1');
        ylabel('Feature 2');
        title('Class Distributions');
        
        % Create legend
        if iscell(model.classes)
            legend(model.classes, 'Location', 'best');
        else
            legendLabels = arrayfun(@(x) sprintf('Class %d', x), model.classes, 'UniformOutput', false);
            legend(legendLabels, 'Location', 'best');
        end
        
        grid on;
    end
end
