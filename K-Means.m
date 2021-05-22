x=[0 0;1 0;1 1;0 1;5 5;5 6;6 5;6 6]
centroids= rand(2,2);
m=size(x,1)
k=2
for h=1:10
for i=1:m 
	K = 1; 
	min = sum((x(i,:) - centroids(1,:)) .^ 2); 
	for j=2:k 
		dist = sum((x(i,:) - centroids(j,:)) .^ 2); 
		if(dist < min) 
			min = dist; 
			K = j; 
		end 
	end 
	indices(i) = K; 
end
for i=1:k
    hold on
    xi = x(indices==i,:);
    if(i==1)
        scatter(xi(:,1),xi(:,2),10,'r')
    else
        scatter(xi(:,1),xi(:,2),10,'b')
    end
    ck = size(xi,1);
    centroids(i, :) = (1/ck) * [sum(xi(:,1)) sum(xi(:,2))];
    scatter(centroids(i, 1),centroids(i, 2),10,'g')
end
end
