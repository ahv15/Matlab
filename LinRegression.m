filename = 'housing.txt';
urlwrite('http://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',filename);
inputNames = {'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'};
outputNames = {'MEDV'};
housingAttributes = [inputNames,outputNames];
formatSpec = '%8f%7f%8f%3f%8f%8f%7f%8f%4f%7f%7f%7f%7f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '',  'ReturnOnError', false);
fclose(fileID);
housing = table(dataArray{1:end-1}, 'VariableNames', {'VarName1','VarName2','VarName3','VarName4','VarName5','VarName6','VarName7','VarName8','VarName9','VarName10','VarName11','VarName12','VarName13','VarName14'});
% Delete the file and clear temporary variables
clearvars filename formatSpec fileID dataArray ans;
delete housing.txt
housing.Properties.VariableNames = housingAttributes;
X = housing{:,inputNames};
Y = housing{:,outputNames};


idx=randperm(506,355);
xTrain=X(idx,:);
xTest=X;
xTest(idx,:)=[];
yTrain=Y(idx,:);
yTest=Y;
yTest(idx,:)=[];
[xTrain mu sigma] =featurenormalize(xTrain);
xTrain=[xTrain ones(size(xTrain,1),1)];
theta=zeros(size(xTrain,2),1);
alpha=0.3;
m=size(xTrain,1);

for i = 1:3000
    gradJ=xTrain'*(xTrain*theta-yTrain);
    theta=theta-(alpha/m)*gradJ;
end

for i = 1:size(xTest,2)
    xTest(:,i)= (xTest(:,i)-mu(i))/sigma(i);
end

xTest=[xTest ones(size(xTest,1),1)];
pred=xTest*theta;
MSETest=(1/(2*size(xTest,1)))*(pred-yTest)'*(pred-yTest)
MSETrain=(1/(2*size(xTrain,1)))*(xTrain*theta-yTrain)'*(xTrain*theta-yTrain)


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
