
%Get table from .xls file
table = readtable('table1.xls');
fig = uifigure;
uit = uitable(fig,'Data',table,'Position',[10 10 550 400]);
drawnow

frequency = 28;
pilots = 8;
iterations = 500;
tests = 5;
error = zeros(iterations,tests);


for p = 1:tests
    for n = 1:iterations
        error(n,p) = Simulation_v35(pilots,frequency,iterations,n);
        n
    end
    p
    pilots = pilots*2;
end
% 
% for n = 1:iterations
%      error(n,1) = Simulation_v35(pilots,frequency,iterations,n);
%      n
% end


figure()

errormean = zeros(1,tests);
errorgeomean = zeros(1,tests);

for k = 1:tests
    errormean(1,k) = mean(rmoutliers(error(:,k)));
    errorgeomean(1,k) = geomean(rmoutliers(error(:,k)));
end

y = [8 16 32 64 128];
figure()
semilogy(y,errormean)
title('Average (arithmetic) localization error');
legend("3.5GHz, N=1","28GHz, N=1");
ylabel("Error (cm)");
xlabel("Pilots (T)");
xtickangle(0)

figure()
semilogy(y,errorgeomean)
title('Average (geometric) localization error');
legend("3.5GHz, N=1");
ylabel("Error (cm)");
xlabel("Pilots (T)");
xtickangle(0)
%xticks([1 2 3 4 5 6 7])
%xticklabels({'1','20','40','60','80','100','120','140'})
%xlim([1 5])

