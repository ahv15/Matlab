function [theta, costHistory] = stochasticGradientDescent(X, Y, alpha, numIterations)
    % STOCHASTICGRADIENTDESCENT Stochastic gradient descent optimization
    %
    % Purpose:
    %   Implements stochastic gradient descent (SGD) for linear regression.
    %   Uses one training example at a time to update parameters, providing
    %   faster but noisier convergence compared to batch gradient descent.
    %
    % Input:
    %   X             - Feature matrix (m x n) where m = samples, n = features
    %   Y             - Target values (m x 1)
    %   alpha         - Learning rate (default: 0.04)
    %   numIterations - Number of iterations (default: 3000)
    %
    % Output:
    %   theta       - Optimized parameter vector
    %   costHistory - Cost function values during optimization (sampled)
    %
    % Example:
    %   [theta, cost] = stochasticGradientDescent(features, labels, 0.04, 1000);
    
    % Set default parameters if not provided
    if nargin < 3
        alpha = 0.04;
    end
    if nargin < 4
        numIterations = 3000;
    end
    
    % Initialize parameters
    m = size(X, 1);
    theta = zeros(size(X, 2), 1);
    
    % Cost history for monitoring convergence (sample every 10 iterations)
    costSampleInterval = 10;
    costHistory = zeros(1, floor(numIterations / costSampleInterval));
    costIndex = 1;
    
    % Stochastic gradient descent loop
    for i = 1:numIterations
        % Randomly shuffle the training examples
        idx = randperm(m);
        XShuffled = X(idx, :);
        YShuffled = Y(idx, :);
        
        % Update parameters using one example at a time
        for j = 1:m
            % Compute gradient for single example
            gradJ = XShuffled(j, :)' * (XShuffled(j, :) * theta - YShuffled(j, :));
            
            % Update parameters
            theta = theta - (alpha / m) * gradJ;
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
