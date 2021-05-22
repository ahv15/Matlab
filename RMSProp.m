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
alpha=0.00004;
m=size(X,1);
J_history=zeros(1,500);
ravg=theta;
gradJ=theta;
rho=0.99999999999999;
epsilon=0.00000001;

for i = 1:3000
    gradJ=X'*(X*theta-Y);
    ravg=rho*ravg+ (1-rho)*(gradJ.^2);
    update=gradJ*alpha./sqrt(ravg + epsilon);
    theta=theta-update;
end



n=size(Xtest,2);

for i = 1:n
Xtest(:,i)= (Xtest(:,i)-mu(i))/sigma(i);
end

Xtest=[Xtest ones(size(Xtest,1),1)];
pred=Xtest*theta;


MSETest=(1/m)*(pred-Ytest)'*(pred-Ytest)
MSETest2=(1/m)*(X*theta-Y)'*(X*theta-Y)
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