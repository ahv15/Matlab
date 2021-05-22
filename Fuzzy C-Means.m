x = [0 0;1 0;1 1;0 1;5 5;5 6;6 5;6 6]';
b = 2;
cen1 = rand(2,1);
cen2 = rand(2,1);
while true
    Pc1 = [];
    Pc2 = [];
    D1 = [];
    D2 = [];
    for i = 1:8
        d1 = norm(cen1 - x(:,i));
        d2 = norm(cen2 - x(:,i));
        if d1 < d2
            D1 = [D1; x(:,i)'];
        else
            D2 = [D2; x(:,i)'];
        end
        pc1 = (1/d1)^(2/(b -1));
        pc2 = (1/d2)^(2/(b -1));
        Pc1 = [Pc1 (pc1^b/(pc1^b + pc2^b))];
        Pc2 = [Pc2 (pc2^b/(pc1^b + pc2^b))];
    end
    mu1 = sum(Pc1.*x,2)/sum(Pc1);
    mu2 = sum(Pc2.*x,2)/sum(Pc2);
    hold on;
    scatter(D1(:,1),D1(:,2),'ro');
    scatter(cen1(1), cen1(2), 'rx');
    scatter(D2(:,1),D2(:,2),'bo');
    scatter(cen2(1), cen2(2), 'bx');
    if(cen1==mu1 & cen2==mu2)
        break;
    end
    cen1 = mu1;
    cen2 = mu2;
    clf
end