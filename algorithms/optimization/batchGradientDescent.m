function [theta, costHistory] = batchGradientDescent(X, Y, alpha, numIterations)
    % BATCHGRADIENTDESCENT Standard batch gradient descent optimization
    %
    % Purpose:
    %   Implements standard batch gradient descent for linear regression.
    %   Uses the entire training dataset to compute the gradient at each
    %   iteration, providing stable but potentially slow convergence.
    %
    % Input:
    %   X             - Feature matrix (m x n) where m = samples, n = features
    %   Y             - Target values (m x 1)
    %   alpha         - Learning rate (default: 0.04)
    %   numIterations - Number of iterations (default: 3000)
    %
    % Output:
    %   theta       - Optimized parameter vector
    %   costHistory - Cost function values during optimization
    %
    % Example:
    %   [theta, cost] = batchGradientDescent(features, labels, 0.04, 1000);
    
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
    
    % Cost history for monitoring convergence
    costHistory = zeros(1, numIterations);
    
    % Batch gradient descent loop
    for i = 1:numIterations
        % Compute gradient using entire batch
        gradJ = X' * (X * theta - Y);
        
        % Update parameters
        theta = theta - (alpha / m) * gradJ;
        
        % Store cost for monitoring
        cost = (1 / (2 * m)) * sum((X * theta - Y).^2);
        costHistory(i) = cost;
    end
end
