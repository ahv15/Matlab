function [theta, costHistory] = miniBatchGradientDescent(X, Y, alpha, numIterations, batchSize)
    % MINIBATCHGRADIENTDESCENT Mini-batch gradient descent optimization
    %
    % Purpose:
    %   Implements mini-batch gradient descent for linear regression.
    %   Uses small batches of training examples to balance the stability
    %   of batch GD and the speed of stochastic GD.
    %
    % Input:
    %   X             - Feature matrix (m x n) where m = samples, n = features
    %   Y             - Target values (m x 1)
    %   alpha         - Learning rate (default: 0.04)
    %   numIterations - Number of iterations (default: 3000)
    %   batchSize     - Size of mini-batches (default: 50)
    %
    % Output:
    %   theta       - Optimized parameter vector
    %   costHistory - Cost function values during optimization (sampled)
    %
    % Example:
    %   [theta, cost] = miniBatchGradientDescent(features, labels, 0.04, 1000, 32);
    
    % Set default parameters if not provided
    if nargin < 3
        alpha = 0.04;
    end
    if nargin < 4
        numIterations = 3000;
    end
    if nargin < 5
        batchSize = 50;
    end
    
    % Initialize parameters
    m = size(X, 1);
    theta = zeros(size(X, 2), 1);
    
    % Cost history for monitoring convergence (sample every 10 iterations)
    costSampleInterval = 10;
    costHistory = zeros(1, floor(numIterations / costSampleInterval));
    costIndex = 1;
    
    % Calculate number of batches
    numBatches = floor(m / batchSize);
    
    % Mini-batch gradient descent loop
    for i = 1:numIterations
        % Randomly shuffle the training examples
        idx = randperm(m);
        XShuffled = X(idx, :);
        YShuffled = Y(idx, :);
        
        % Process each mini-batch
        for j = 1:numBatches
            % Extract mini-batch
            startIdx = (j - 1) * batchSize + 1;
            endIdx = j * batchSize;
            
            XBatch = XShuffled(startIdx:endIdx, :);
            YBatch = YShuffled(startIdx:endIdx, :);
            
            % Compute gradient for mini-batch
            gradJ = XBatch' * (XBatch * theta - YBatch);
            
            % Update parameters
            theta = theta - (alpha / batchSize) * gradJ;
        end
        
        % Sample cost periodically for monitoring
        if mod(i, costSampleInterval) == 0
            cost = (1 / (2 * m)) * sum((X * theta - Y).^2);
            costHistory(costIndex) = cost;
            costIndex = costIndex + 1;
        end
    end
    
    % Trim cost history to actual length
    costHistory = costHistory(1:costIndex-1);
end
