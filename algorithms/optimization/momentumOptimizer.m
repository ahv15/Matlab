function [theta, costHistory] = momentumOptimizer(X, Y, alpha, numIterations, gamma)
    % MOMENTUMOPTIMIZER Gradient Descent with Momentum optimization algorithm
    %
    % Purpose:
    %   Implements gradient descent with momentum for optimization.
    %   Momentum helps accelerate gradients in the relevant direction 
    %   and dampens oscillations, leading to faster convergence.
    %
    % Input:
    %   X             - Feature matrix (m x n) where m = samples, n = features
    %   Y             - Target values (m x 1)
    %   alpha         - Learning rate (default: 0.04)
    %   numIterations - Number of iterations (default: 3000)
    %   gamma         - Momentum parameter (default: 0.000009)
    %
    % Output:
    %   theta       - Optimized parameter vector
    %   costHistory - Cost function values during optimization
    %
    % Example:
    %   [theta, cost] = momentumOptimizer(features, labels, 0.04, 1000, 0.9);
    
    % Set default parameters if not provided
    if nargin < 3
        alpha = 0.04;
    end
    if nargin < 4
        numIterations = 3000;
    end
    if nargin < 5
        gamma = 0.000009;
    end
    
    % Initialize parameters
    m = size(X, 1);
    theta = zeros(size(X, 2), 1);
    
    % Initialize momentum term
    velocity = zeros(size(theta));
    
    % Cost history for monitoring convergence
    costHistory = zeros(1, numIterations);
    
    % Momentum optimization loop
    for i = 1:numIterations
        % Compute gradient
        gradJ = X' * (X * theta - Y);
        
        % Update velocity (momentum term)
        velocity = gamma * velocity + (alpha / m) * gradJ;
        
        % Update parameters
        theta = theta - velocity;
        
        % Store cost for monitoring
        cost = (1 / (2 * m)) * sum((X * theta - Y).^2);
        costHistory(i) = cost;
    end
end
