load fisheriris
p=randperm(150);
k=zeros(105,4);
w=zeros(45,4);
set=[];
vir=[];
versi=[];
ps=0;
pvi=0;
pve=0;
k1=cell(105,1);
w1=cell(45,1);
for i= 1:105
    k(i,1:4)=meas(p(i),1:4);
    k1(i,1)=species(p(i),1);
end
for i= 106:150
    w(i-105,1:4)=meas(p(i),1:4);
    w1(i-105,1)=species(p(i),1);
end
for i= 1:105
    if(k1(i,1)=="setosa")                   %/ segregating the training data
       set(i,1:4)=k(i,1:4); %#ok<SAGROW>         into the classes  %/
       ps=ps+1;
    elseif(k1(i,1)=="virginica")
       vir(i,1:4)=k(i,1:4); %#ok<SAGROW>
       pvi=pvi+1;
    elseif(k1(i,1)=="versicolor")
       versi(i,1:4)=k(i,1:4); %#ok<SAGROW>
       pve=pve+1;
    end
end

ps=ps/105;                    %/ calculating the prior probabilities
pvi=pvi/105;                  %/   of the 3 classes
pve=pve/105;

set(~any(set,2),:)=[];
versi(~any(versi,2),:)=[];    %/  removing rows that are only zero
vir(~any(vir,2),:)=[];

figure(20)
histogram(k(:,1));          
figure(21)                    %/ creating the histogram of the 4 features
histogram(k(:,2));
figure(22)
histogram(k(:,3));
figure(23)
histogram(k(:,4));

meanset1=mean(set(:,3));
meanset2=mean(set(:,4));                %/ calculating the mean matrix for
meanm=[meanset1 meanset2];               %/  the samples of the class setosa

covar=cov(set(:,3),set(:,4));           %/ calculating the covariance matrix

x1 = 0.1:0.1:9;                         %/ we are assuming the range of the 
x2 = 0.1:0.1:6;                         %/ x and y values for which the graph
[X1,X2] = meshgrid(x1,x2);               %/  should be plotted
X = [X1(:) X2(:)];
y1 = mvnpdf(X,meanm,covar);
y1 = reshape(y1,length(x2),length(x1));   %/ this part is just plotting the
figure(1)                                  %/ multivariate gaussian distribution
surf(x1,x2,y1)
xlabel('petal length')
ylabel('petal width')
zlabel('Probability Density')

y1 = mvnpdf(X,meanm,covar);             %/ this is our pdf which is the matrix  
                                         %/ containing the corresponding
                                         %/ pdf values for each point in
                                         % / our grid(the range of values for x and y)

meanversi1=mean(versi(:,3));
meanversi2=mean(versi(:,4));
meanm1=[meanversi1 meanversi2];
covar1=cov(versi(:,3),versi(:,4));
y2 = mvnpdf(X,meanm1,covar1);
y2 = reshape(y2,length(x2),length(x1));
figure(2)
surf(x1,x2,y2)
y2 = mvnpdf(X,meanm1,covar1);
xlabel('petal length')
ylabel('petal width')
zlabel('Probability Density')

meanvir1=mean(vir(:,3));
meanvir2=mean(vir(:,4));
meanm2=[meanvir1 meanvir2];
covar2=cov(vir(:,3),vir(:,4));
y3 = mvnpdf(X,meanm2,covar2);
y3 = reshape(y3,length(x2),length(x1));
figure(3)
surf(x1,x2,y3)
y3 = mvnpdf(X,meanm2,covar2);
xlabel('petal length')
ylabel('petal width')
zlabel('Probability Density')

predictedy=cell(45,1);        %/ now testing our classifier on test data
correct=0;
for i= 106:150
    d1=(w(i-105,3));          %/ values of the features of test data
    d2=(w(i-105,4));
    i1=(d1-0.1)*600 + (d2*10);   %/ calculating the index of the y value(pdf) corresponding
    i1=int64(i1);                %/ to the values of the features of test data
    
    prs=(y1(i1,1))*ps;        %/ calculating a value proportional to posterior probability
    prve=(y2(i1,1))*pve;      %/ for each of the classes (ps,pve and pvi are prior probabilities)
    prvi=(y3(i1,1))*pvi;      %/ we are multiplying the prior with the likelihood which we get from
                              %/ the normal pdf's we constructed. We are
                              %/ excluding evidence from calculation as it is
                              %/ constant among the classes and it is not
                              %/ needed to find max posterior probability
                              
   
    if(prs>prve && prs>prvi)             %/ finding the class that has the maximum posterior probability
        predictedy(i-105,1)={'setosa'};  %/and then checking if predicted class matches with actual class
        if(w1(i-105,1)=="setosa")
            correct=correct+1;
        end
    elseif(prve>prs && prve>prvi)
        predictedy(i-105,1)={'versicolor'};
        if(w1(i-105,1)=="versicolor")
            correct=correct+1;
        end
    elseif(prvi>prs && prvi>prve)
        predictedy(i-105,1)={'virginica'};
        if(w1(i-105,1)=="virginica")
            correct=correct+1;
        end
    end
end
accuracy=(correct/45)*100
figure(30)
cm = confusionchart(w1,predictedy);
cm.NormalizedValues
cm.Title = 'Iris Flower Classification Using Bayes Classifier';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';


means1=mean(set(:,3));     %/mean of first feature
means2=mean(set(:,4));     %/mean of other feature
vars1=var(set(:,3));       %/variance of first feature
vars2=var(set(:,4));       %/variance of the other feature
x1 = 0.1:0.1:9;
x2 = 0.1:0.1:6;
Y1 = normpdf(x1,means1,vars1);  %/creating assumed normal distribution for first feature
figure(5)
plot(x1,Y1)
xlabel('petal length')
ylabel('Probability Density')
Y2 = normpdf(x2,means2,vars2);  %/creating assumed normal distribution for the other feature
figure(6)
plot(x2,Y2)
xlabel('petal width')            %/ plotting the normal distributions
ylabel('Probability Density')

meanve1=mean(versi(:,3));
meanve2=mean(versi(:,4));
varve1=var(versi(:,3));
varve2=var(versi(:,4));
Y3 = normpdf(x1,meanve1,varve1);
figure(7)
plot(x1,Y3)
Y4 = normpdf(x2,meanve2,varve2);
figure(8)
plot(x2,Y4)
xlabel('petal length')
ylabel('petal width')
zlabel('Probability Density')

meanvi1=mean(vir(:,3));
meanvi2=mean(vir(:,4));
varvi1=var(vir(:,3));
varvi2=var(vir(:,4));
Y5 = normpdf(x1,meanvi1,varvi1);
figure(9)
plot(x1,Y5)
Y6 = normpdf(x2,meanvi2,varvi2);
figure(10)
plot(x2,Y6)
xlabel('petal length')
ylabel('petal width')
zlabel('Proabbiltiy Density')


correct1=0;
predictedy1=cell(45,1);
for i= 106:150               %/ testing the naive bayes classifier on our test data
    d1=(w(i-105,3));         %/ getting the values of the features of the test data
    d2=(w(i-105,4));
    i1=d1*10;
    i1=int64(i1);            %/ getting the index of the y value(pdf value) corresponding to
    i2=d2*10;                %/ the given features
    i2=int64(i2);
    prs=(Y1(i1)*Y2(i2))*ps;
    prve=(Y3(i1)*Y4(i2))*pve;   %/calculating a value proportional to posterior proabability
    prvi=(Y5(i1)*Y6(i2))*pvi;    %/ for each of the classes
    if(prs>prve && prs>prvi)
        predictedy1(i-105,1)={'setosa'};
        if(w1(i-105,1)=="setosa")
            correct1=correct1+1; 
        end
    elseif(prve>prs && prve>prvi)            %/ finding the class with the highest posterior probability
        predictedy1(i-105,1)={'versicolor'};  %/ and making that our predicted class and then comparing 
        if(w1(i-105,1)=="versicolor")         %/ that with the actual class to find accuracy
            correct1=correct1+1;
        end
    elseif(prvi>prs && prvi>prve)
        predictedy1(i-105,1)={'virginica'};
        if(w1(i-105,1)=="virginica")
            correct1=correct1+1;
        end
    end
end
accuracy1=(correct1/45)*100
figure(31)
cm = confusionchart(w1,predictedy1);
cm.NormalizedValues
cm.Title = 'Iris Flower Classification Using Naive Bayes Classifier';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

