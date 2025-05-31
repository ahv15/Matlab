function [theta, costHistory] = adamOptimizer(X, Y, alpha, numIterations)
    % ADAMOPTIMIZER Adaptive Moment Estimation (Adam) optimization algorithm
    %
    % Purpose:
    %   Implements the Adam optimization algorithm for gradient descent.
    %   Adam combines the advantages of AdaGrad and RMSProp by computing
    %   adaptive learning rates for each parameter using estimates of first
    %   and second moments of the gradients.
    %
    % Input:
    %   X             - Feature matrix (m x n) where m = samples, n = features
    %   Y             - Target values (m x 1)
    %   alpha         - Learning rate (default: 0.002)
    %   numIterations - Number of iterations (default: 30)
    %
    % Output:
    %   theta       - Optimized parameter vector
    %   costHistory - Cost function values during optimization
    %
    % Example:
    %   [theta, cost] = adamOptimizer(features, labels, 0.002, 100);
    
    % Set default parameters if not provided
    if nargin < 3
        alpha = 0.002;
    end
    if nargin < 4
        numIterations = 30;
    end
    
    % Initialize parameters
    m = size(X, 1);
    theta = zeros(size(X, 2), 1);
    
    % Adam hyperparameters
    beta1 = 0.9;
    beta2 = 0.9;
    epsilon = 1e-8;
    
    % Initialize moment estimates
    mTheta = zeros(size(theta));
    vTheta = zeros(size(theta));
    
    % Cost history for monitoring convergence
    costHistory = zeros(1, numIterations);
    
    % Adam optimization loop
    for i = 1:numIterations
        % Compute gradient
        gradJ = X' * (X * theta - Y);
        
        % Update biased first moment estimate
        mTheta = beta1 * mTheta + (1 - beta1) * gradJ;
        
        % Update biased second raw moment estimate  
        vTheta = beta2 * vTheta + (1 - beta2) * (gradJ.^2);
        
        % Compute bias-corrected first moment estimate
        m1 = mTheta / (1 - (beta1^i));
        
        % Compute bias-corrected second raw moment estimate
        v1 = vTheta / (1 - (beta2^i));
        
        % Update parameters
        update = alpha * m1 ./ (sqrt(v1) + epsilon);
        theta = theta - update;
        
        % Store cost for monitoring
        cost = (1 / (2 * m)) * sum((X * theta - Y).^2);
        costHistory(i) = cost;
    end
end
