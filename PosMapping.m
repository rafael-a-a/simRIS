clearvars

frequency = 28;
pilots = 128;
iterations = 10;
parray = zeros(iterations,3);

for p = 1:iterations
    
    parray(p,:) = Simulation_v35(pilots,frequency,iterations,p);
   
end

parray;
p = [0.281 4.874 1];
ptx = [0 5 1];

%RIS%
K = 100;
d = 0.2;
p_k = zeros(K,3);

for i = 1:K/2

                                     %RIS distribution alongside x wall

    p_k(i,1) = -4.8 + (i - 1) * d;
    p_k(i,2) = 0;
    p_k(i,3) = 1;

end
clear i;
for i = K/2:K                       %RIS distribution alongside y wall


    p_k(i,1) = 5;
    p_k(i,2) = (i - K/2) * d;
    p_k(i,3) = 1;

end
clear i;

figure();
plot3(ptx(1,1),ptx(1,2),ptx(1,3), '+K', 'MarkerSize', 10);
hold on
plot3(p(1,1),p(1,2),p(1,3), 'o', 'MarkerSize', 10);
hold on
hold on
%plot3(lb(1,1),lb(1,2),lb(1,3),'*','MarkerSize',10);
hold on
%plot3(ub(1,1),ub(1,2),ub(1,3),'*','MarkerSize',10);
hold on

for i = 1:K
    plot3(p_k(i,1),p_k(i,2),p_k(1,3),'.', 'MarkerSize', 10);
    hold on
end

clear i

cont = 0;
for i = 1:iterations
    plot3(parray(i,1),parray(i,2),parray(i,3),'.','MarkerSize',3);
    if (norm(abs(parray(i,:) - p)) * 100) < 10
        cont = cont + 1 ;
    end
    hold on
end

legend('tx','UE','RIS','Estimates');
title('3D View');

fprintf("Points with an error less than 10cm: " + cont + "\n");
