function [theta, costHistory] = rmsPropOptimizer(X, Y, alpha, numIterations, rho)
    % RMSPROPOPTIMIZER Root Mean Square Propagation optimization algorithm
    %
    % Purpose:
    %   Implements the RMSProp optimization algorithm for gradient descent.
    %   RMSProp adapts the learning rate by dividing the gradient by a 
    %   running average of the magnitudes of recent gradients.
    %
    % Input:
    %   X             - Feature matrix (m x n) where m = samples, n = features
    %   Y             - Target values (m x 1)
    %   alpha         - Learning rate (default: 0.00004)
    %   numIterations - Number of iterations (default: 3000)
    %   rho           - Decay rate for running average (default: 0.999)
    %
    % Output:
    %   theta       - Optimized parameter vector
    %   costHistory - Cost function values during optimization
    %
    % Example:
    %   [theta, cost] = rmsPropOptimizer(features, labels, 0.00004, 1000, 0.999);
    
    % Set default parameters if not provided
    if nargin < 3
        alpha = 0.00004;
    end
    if nargin < 4
        numIterations = 3000;
    end
    if nargin < 5
        rho = 0.999;
    end
    
    % Initialize parameters
    m = size(X, 1);
    theta = zeros(size(X, 2), 1);
    
    % RMSProp parameters
    epsilon = 1e-8;
    
    % Initialize running average of squared gradients
    ravg = zeros(size(theta));
    
    % Cost history for monitoring convergence
    costHistory = zeros(1, numIterations);
    
    % RMSProp optimization loop
    for i = 1:numIterations
        % Compute gradient
        gradJ = X' * (X * theta - Y);
        
        % Update running average of squared gradients
        ravg = rho * ravg + (1 - rho) * (gradJ.^2);
        
        % Compute update step
        update = gradJ * alpha ./ sqrt(ravg + epsilon);
        
        % Update parameters
        theta = theta - update;
        
        % Store cost for monitoring
        cost = (1 / (2 * m)) * sum((X * theta - Y).^2);
        costHistory(i) = cost;
    end
end
