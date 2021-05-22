load fisheriris
meas=meas(51:150,:);
species=species(51:150,:);

% class 0 is versicolor and class 1 is virginica
species_class(1:50,:)=0;
species_class(51:100,:)=1;

idx=randperm(100,70);
xTrain=meas(idx,:);
xTest=meas;
xTest(idx,:)=[];
yTrain=species_class(idx,:);
yTest=species_class;
yTest(idx,:)=[];

xTrain=[xTrain ones(size(xTrain,1),1)];
theta=zeros(size(xTrain,2),1);
alpha=0.3;
m=size(xTrain,1);

for i = 1:3000
    g=1./(1+exp(-(xTrain*theta))); %sigmoid function
    theta=theta-(alpha/m)*xTrain'*(g-yTrain);
end

xTest=[xTest ones(size(xTest,1),1)];
pred=xTest*theta;

% J=-(1/m)*sum(yTrain.*log(g)+(1-yTrain).*log(1-g)) cost function
costTest=-(1/m)*sum(yTest.*log(1./(1+exp(-(xTest*theta))))+(1-yTest).*log(1-(1./(1+exp(-(xTest*theta))))))
costTrain=-(1/m)*sum(yTrain.*log(1./(1+exp(-(xTrain*theta))))+(1-yTrain).*log(1-(1./(1+exp(-(xTrain*theta))))))

vir=0;
tvir=0;
ver=0;
tver=0;
g=1./(1+exp(-(xTrain*theta)));

%training accuracy for virginica
for i=1:70
    if(g(i,:)>0.5 & yTrain(i,:)==1)
        vir=vir+1;
    if(yTrain(i,:)==1)
        tvir=tvir+1;
    end
    end
end

%training accuracy for versicolor
for i=1:70
    if(g(i,:)<=0.5 & yTrain(i,:)==0)
        ver=ver+1;
    if(yTrain(i,:)==0)
        tver=tver+1;
    end
    end
end   
trainacc1=(vir/tvir)*100
trainacc2=(ver/tver)*100


vir=0;
tvir=0;
ver=0;
tver=0;
g=1./(1+exp(-(xTest*theta)));

%test accuracy for virginica
for i=1:30
    if(g(i,:)>0.5 & yTest(i,:)==1)
        vir=vir+1;
    if(yTest(i,:)==1)
        tvir=tvir+1;
    end
    end
end

%test accuracy for versicolor
for i=1:30
    if(g(i,:)<=0.5 & yTest(i,:)==0)
        ver=ver+1;
    if(yTest(i,:)==0)
        tver=tver+1;
    end
    end
end   
testacc1=(vir/tvir)*100
testacc2=(ver/tver)*100
    
    