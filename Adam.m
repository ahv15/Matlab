tic
data=readtable('Life Expectancy Data.csv');
data = removevars(data, 'Country');
data = removevars(data, 'Year');
data = removevars(data, 'Status');
data=rmmissing(data);
data1=table2array(data);

idx=randperm(size(data1,1),1154);
Train=data1(idx,:);
Test=data1;
Test(idx,:)=[];
Y=Train(:,1);
X=Train(:,2:end);
X=[X X.^2];
Ytest=Test(:,1);
Xtest=Test(:,2:end);
Xtest=[Xtest Xtest.^2];
[X mu sigma] =featurenormalize(X);
X=[X ones(size(X,1),1)];
theta=zeros(size(X,2),1);
alpha=0.002;
m2=size(X,1);
J_history=zeros(1,500);
ravg=theta;
gradJ=theta;
epsilon=0.00000001;
beta1=0.9;
beta2=0.9;
m=theta;
v=theta;

for i = 1:30
    gradJ=X'*(X*theta-Y);
    m = beta1 * m + (1 - beta1) * (gradJ);
    v = beta2 * v + (1 - beta2) * (gradJ.^2);
    m1 = m/(1 - (beta1^i));
    v1 = v/(1 - (beta2^i));
    update = alpha*m1./(sqrt(v1) + epsilon)
    theta=theta-update;
end



n=size(Xtest,2);

for i = 1:n
Xtest(:,i)= (Xtest(:,i)-mu(i))/sigma(i);
end

Xtest=[Xtest ones(size(Xtest,1),1)];
pred=Xtest*theta;


MSETest=(1/m2)*(pred-Ytest)'*(pred-Ytest)
MSETest2=(1/m2)*(X*theta-Y)'*(X*theta-Y)
toc

function [X_norm mu sigma] =featurenormalize(X)

X_norm=X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
n=size(X,2);
for i = 1:n
mu(i)=mean(X(:,i));
sigma(i)=std(X(:,i));
X_norm(:,i)= (X(:,i)-mu(i))/sigma(i);
end

end