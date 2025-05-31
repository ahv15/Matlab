function [centroids, clusterAssignments, finalCost] = kMeansClustering(X, k, maxIterations, plotResults)
    % KMEANSCLUSTERING K-means clustering algorithm
    %
    % Purpose:
    %   Implements the K-means clustering algorithm to partition data
    %   into k clusters. The algorithm iteratively assigns points to 
    %   nearest centroids and updates centroids based on cluster means.
    %
    % Input:
    %   X             - Data matrix (m x n) where m = samples, n = features
    %   k             - Number of clusters
    %   maxIterations - Maximum number of iterations (default: 10)
    %   plotResults   - Whether to plot results (default: true)
    %
    % Output:
    %   centroids          - Final cluster centroids (k x n)
    %   clusterAssignments - Cluster assignment for each point (m x 1)
    %   finalCost          - Final within-cluster sum of squares
    %
    % Example:
    %   [centroids, assignments] = kMeansClustering(data, 3, 20, true);
    
    % Set default parameters
    if nargin < 3
        maxIterations = 10;
    end
    if nargin < 4
        plotResults = true;
    end
    
    % Initialize parameters
    m = size(X, 1);
    n = size(X, 2);
    
    % Initialize centroids randomly
    centroids = rand(k, n);
    
    % Scale centroids to data range
    for i = 1:n
        centroids(:, i) = centroids(:, i) * (max(X(:, i)) - min(X(:, i))) + min(X(:, i));
    end
    
    clusterAssignments = zeros(m, 1);
    
    % K-means iterations
    for iteration = 1:maxIterations
        % Assign each point to nearest centroid
        for i = 1:m
            minDistance = inf;
            closestCentroid = 1;
            
            for j = 1:k
                distance = sum((X(i, :) - centroids(j, :)).^2);
                if distance < minDistance
                    minDistance = distance;
                    closestCentroid = j;
                end
            end
            
            clusterAssignments(i) = closestCentroid;
        end
        
        % Update centroids
        for i = 1:k
            clusterPoints = X(clusterAssignments == i, :);
            
            if ~isempty(clusterPoints)
                centroids(i, :) = mean(clusterPoints, 1);
            end
        end
        
        % Plot results if requested and data is 2D
        if plotResults && n == 2
            figure(1);
            hold on;
            
            % Clear previous plot
            if iteration > 1
                clf;
            end
            
            % Plot clusters with different colors
            colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k'];
            for i = 1:k
                clusterPoints = X(clusterAssignments == i, :);
                if ~isempty(clusterPoints)
                    colorIndex = mod(i - 1, length(colors)) + 1;
                    scatter(clusterPoints(:, 1), clusterPoints(:, 2), 10, colors(colorIndex), 'filled');
                end
                
                % Plot centroids
                scatter(centroids(i, 1), centroids(i, 2), 100, 'k', 'filled', 'MarkerEdgeColor', 'w');
            end
            
            title(sprintf('K-Means Clustering - Iteration %d', iteration));
            xlabel('Feature 1');
            ylabel('Feature 2');
            legend('Cluster 1', 'Cluster 2', 'Centroids', 'Location', 'best');
            pause(0.5);
        end
    end
    
    % Calculate final cost (within-cluster sum of squares)
    finalCost = 0;
    for i = 1:m
        finalCost = finalCost + sum((X(i, :) - centroids(clusterAssignments(i), :)).^2);
    end
end
