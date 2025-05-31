function [centroids, membershipMatrix, finalCost] = fuzzyCMeansClustering(X, c, fuzziness, maxIterations, tolerance, plotResults)
    % FUZZYCMEANSCLUSTERING Fuzzy C-Means clustering algorithm
    %
    % Purpose:
    %   Implements the Fuzzy C-Means (FCM) clustering algorithm where each
    %   data point can belong to multiple clusters with different degrees
    %   of membership. Unlike hard clustering, FCM provides soft clustering.
    %
    % Input:
    %   X             - Data matrix (n x m) where n = features, m = samples
    %   c             - Number of clusters (default: 2)
    %   fuzziness     - Fuzziness parameter (default: 2)
    %   maxIterations - Maximum iterations (default: 100)
    %   tolerance     - Convergence tolerance (default: 1e-5)
    %   plotResults   - Whether to plot results (default: true)
    %
    % Output:
    %   centroids        - Final cluster centroids (n x c)
    %   membershipMatrix - Membership values for each point (m x c)
    %   finalCost        - Final objective function value
    %
    % Example:
    %   [centroids, membership] = fuzzyCMeansClustering(data, 3, 2, 50);
    
    % Set default parameters
    if nargin < 2
        c = 2;
    end
    if nargin < 3
        fuzziness = 2;
    end
    if nargin < 4
        maxIterations = 100;
    end
    if nargin < 5
        tolerance = 1e-5;
    end
    if nargin < 6
        plotResults = true;
    end
    
    % Get dimensions
    [n, m] = size(X);
    
    % Initialize centroids randomly
    centroids = rand(n, c);
    
    % Scale centroids to data range
    for i = 1:n
        centroids(i, :) = centroids(i, :) * (max(X(i, :)) - min(X(i, :))) + min(X(i, :));
    end
    
    prevCentroids = centroids;
    membershipMatrix = zeros(m, c);
    
    % Main iteration loop
    for iteration = 1:maxIterations
        % Calculate membership matrix
        for i = 1:m
            distances = zeros(1, c);
            
            % Calculate distances to all centroids
            for j = 1:c
                distances(j) = norm(centroids(:, j) - X(:, i));
                if distances(j) == 0
                    distances(j) = eps; % Avoid division by zero
                end
            end
            
            % Calculate membership values
            for j = 1:c
                membershipSum = 0;
                for k = 1:c
                    membershipSum = membershipSum + (distances(j) / distances(k))^(2 / (fuzziness - 1));
                end
                membershipMatrix(i, j) = 1 / membershipSum;
            end
        end
        
        % Update centroids
        for j = 1:c
            numerator = zeros(n, 1);
            denominator = 0;
            
            for i = 1:m
                weight = membershipMatrix(i, j)^fuzziness;
                numerator = numerator + weight * X(:, i);
                denominator = denominator + weight;
            end
            
            if denominator > 0
                centroids(:, j) = numerator / denominator;
            end
        end
        
        % Check for convergence
        centroidChange = norm(centroids - prevCentroids);
        if centroidChange < tolerance
            fprintf('Converged after %d iterations\n', iteration);
            break;
        end
        
        prevCentroids = centroids;
        
        % Plot results if requested and data is 2D
        if plotResults && n == 2
            figure(1);
            clf;
            hold on;
            
            % Assign each point to cluster with highest membership
            [~, clusterAssignments] = max(membershipMatrix, [], 2);
            
            colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k'];
            for i = 1:c
                clusterPoints = X(:, clusterAssignments == i);
                if ~isempty(clusterPoints)
                    colorIndex = mod(i - 1, length(colors)) + 1;
                    scatter(clusterPoints(1, :), clusterPoints(2, :), 10, colors(colorIndex), 'o');
                end
                
                % Plot centroids
                scatter(centroids(1, i), centroids(2, i), 100, 'k', 'x', 'LineWidth', 3);
            end
            
            title(sprintf('Fuzzy C-Means Clustering - Iteration %d', iteration));
            xlabel('Feature 1');
            ylabel('Feature 2');
            pause(0.1);
        end
    end
    
    % Calculate final cost (objective function)
    finalCost = 0;
    for i = 1:m
        for j = 1:c
            distance = norm(X(:, i) - centroids(:, j));
            finalCost = finalCost + membershipMatrix(i, j)^fuzziness * distance^2;
        end
    end
    
    fprintf('Final objective function value: %.6f\n', finalCost);
end
