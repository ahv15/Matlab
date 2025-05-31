function [xNorm, mu, sigma] = featureNormalize(X)
    % FEATURENORMALIZE Normalizes the features in X
    %
    % Purpose:
    %   Normalizes features by subtracting the mean and dividing by the 
    %   standard deviation. This is essential for gradient descent algorithms
    %   to converge properly when features have different scales.
    %
    % Input:
    %   X - Matrix where each column represents a feature and each row 
    %       represents a training example
    %
    % Output:
    %   xNorm - Normalized feature matrix
    %   mu    - Vector of means for each feature
    %   sigma - Vector of standard deviations for each feature
    %
    % Example:
    %   [xNorm, mu, sigma] = featureNormalize(trainingData);
    %   testDataNorm = (testData - mu) ./ sigma;
    
    xNorm = X;
    mu = zeros(1, size(X, 2));
    sigma = zeros(1, size(X, 2));
    n = size(X, 2);
    
    for i = 1:n
        mu(i) = mean(X(:, i));
        sigma(i) = std(X(:, i));
        xNorm(:, i) = (X(:, i) - mu(i)) / sigma(i);
    end
end
