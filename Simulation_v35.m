
%Get table from .xls file
clear all;

table = readtable('table1.xls');
fig = uifigure;
uit = uitable(fig,'Data',table,'Position',[10 10 550 400]);
drawnow

fprintf(" If you want to change any of the parameters (except carrier frequency) listed on Table 1 \n");
fprintf(" please change them in the .xls file and re-run the simulation \n");

text1 = 'Do you want to change carrier frequency? (y/n) \n';
change = input(text1,"s");

                            %Initializing parameters

f_c = table{1:1,["Value"]} * 1e9;     %GHz
G_T = 10^(table{2:2,["Value"]}/10);
G_R = 10^(table{3:3,["Value"]}/10);
N_t = table{4:4,["Value"]};
%N = table{5:5,["Value"]};
N = 16; %FOR NOW !!!!!!!!!!!!!!!!!!!!!!
df = table{6:6,["Value"]} * 1e3;  %KHz
TX_pwr_sub = 10^(table{7:7,["Value"]}/10)/1000;
T_a = table{8:8,["Value"]} * 1e-9; %nanoseconds
n_f = table{9:9,["Value"]};
sig2 = 10^(table{10:10,["Value"]}/10)/1000;
k = table{11:11,["Value"]};
F_o = table{12:12,["Value"]};
J = table{13:13,["Value"]};
n_e = table{14:14,["Value"]};
S_T = table{15:15,["Value"]} * 1e-9;
L = table{16:16,["Value"]};
T_T = table{17:17,["Value"]} * 1e-9; %nanoseconds
Ne = table{18:18,["Value"]};
%L_x = table{19:19,["Value"]};
%L_y = table{19:19,["Value"]};
d = table{20:20,["Value"]} / 100; %cm
K = table{21:21,["Value"]};
N_x = table{22:22,["Value"]};
N_y = table{23:23,["Value"]};
%T = 128;
Gc = pi;
P_T = 1;


disp('This simulation assumes that UE is always in NLOS');
zeta_0 = 0;


switch change
    case 'y'
        text2 = 'Enter carrier frequency in GHz \n';
        f_c = input(text2) * 1e9 ;
    otherwise

end

T = input('Please choose how many(1-120) pilots (T) \n');  %pilots
lambda = 3e8 / (f_c);
L_x = lambda / 2;                              % L_x = lambda/2
L_y = L_x;
d_x = L_x;      %reference 31 page 27
d_y = L_y;

f_n = zeros(N,2);

%subcarriers frequencies and wavelengths, wavelength is the column 2

for i = 1:N
    f_n(i,1) = f_c + ( i - ( N + 1 )/2) * df;
    f_n(i,2) = 3e8/f_n(i,1);
end
clear i;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                        % UE location

p_x = unifrnd(-4,4);
p_y = unifrnd(1,10);
p_z = 1;

%p = [p_x p_y p_z];
p = [0 1 1];
                                        %TX location

ptx = [0 5 1];

                                        %RIS distribution

p_k = zeros(K,3);

%%fazer eq 13 (ignorar mp)
%%refazer alpha bem

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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                      %THETA(solid angle,vector norm) calculations


THETAi_k = zeros(K,3);         %Creating Kx3 matrix to store THETAi_k and last parameter is ||ptx-p_k||

for i= 1:K
    if i<K/2
    [phi, theta, r] = cart2sph(p_k(i,1)-ptx(1,1),p_k(i,3)-ptx(1,3),p_k(i,2)-ptx(1,2));
    else
    [phi, theta, r] = cart2sph(p_k(i,2)-ptx(1,2),p_k(i,3)-ptx(1,3),p_k(i,1)-ptx(1,1));%change reference in the y wall    
    end

    
    THETAi_k(i,1) = phi+pi/2;%make phi start from z xis and rotate in xz
    THETAi_k(i,2) = pi/2-abs(theta);%avoid negative theta, theta=0 in front of the antenna
    THETAi_k(i,3) = r;

end
clear i;

THETAr_k = zeros(K,3);        %Creating Kx3 matrix to store THETAr_k and last parameter is ||pt-p_k||

%in this simulation, theta varies in xy plane,
%being pi/2 if the point is in the negative x axis, 0 if its in the y axis
%and pi/2 if its on the positive x axis, this function only returns from -pi/2 to
%pi/2, but its still correct; the reference is transferred to the y wall
%cells
%phi varies in the xz plane.

for i= 1:K

    if i<K/2
    [phi, theta, r] = cart2sph(p_k(i,1)-p(1,1),p_k(i,3)-p(1,3),p_k(i,2)-p(1,2));
    else
    [phi, theta, r] = cart2sph(p_k(i,2)-p(1,2),p_k(i,3)-p(1,3),p_k(i,1)-p(1,1));  %change reference in the y wall      
    end

    THETAr_k(i,1) = phi+pi/2;%make phi start from z xis and rotate in xz
    THETAr_k(i,2) = pi/2-abs(theta);%avoid negative theta, theta=0 in front of the antenna
    THETAr_k(i,3) = r;

end
clear i;

figure();
plot(THETAi_k(:,2)*180/pi)
hold on
plot(THETAr_k(:,2)*180/pi)
legend('Incident theta','Reflected theta')
title('Theta');



%THETAr_k     phi is 270 when the tile is to the RIGHT of ptx and PI when is to the LEFT of ptx.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                                %Calculating beta_t,k
                   %arrayfact(phi, theta, lambda,Ne, N_x, N_y, d_x, d_y)

AF = zeros(K,2);      % column 1 referes to i,column 2 referes to r, check text just before eq 1
F = zeros(K,2);

%eq 2


for i = 1:K
    AF(i,1) = functionslibv7.arrayfact(THETAi_k(i,1), THETAi_k(i,2),lambda, Ne, N_x, N_y, d_x, d_y);
    AF(i,2) = functionslibv7.arrayfact(THETAr_k(i,1), THETAr_k(i,2),lambda, Ne, N_x, N_y, d_x, d_y);
end
clear i;
%eq3
figure();
plot(AF(:,1))
hold on
plot(AF(:,2))
legend('Incident AF','Reflected AF')
title('Array Factor');

for i = 1:K
    F(i,1) = functionslibv7.normpwr(THETAi_k(i,1),THETAi_k(i,2));
    F(i,2) = functionslibv7.normpwr(THETAr_k(i,1),THETAr_k(i,2));
end
clear i;

PSIt_k = zeros(T,K);
%check page 4290:(ii) Random
%binary: Independent random reflection phases {Ψt,k} taking
%the values {0, π} with identical probability;

for t = 1:T
    for cont = 1:K
    %    PSIt_k(i,cont) = functionslibv7.refphase();
    PSIt_k(t,cont) = 2*pi*t*floor(cont*T/K)/T;
    end
end

BETAt_k = zeros(T,K);
%eq 1
for i = 1:T
    for cont = 1:K     %In this paper eq 1 is written as: F(THETA(k,i) F(THETA(k,r)  AF(THETA(k,i) AF(THETA(k,r)
        BETAt_k(i,cont) = functionslibv7.reflectioncoeff( F(cont,1), F(cont,2), AF(cont,1), AF(cont,2), PSIt_k(i,cont) );
    end
end

clear i;

figure();
plot(abs(BETAt_k(2,:)));
legend('Betat_k for subcarrier 2');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                        %Direct channel coefficent
                                        %(hd_n = hdp_n+hmp_n)


hdp = zeros(N,1);
hmp = zeros(N,1);
hd = zeros(N,1);
normtxp = norm(ptx-p);

earg = zeros(N,1); %argument of exponencial function in eq 5


for i = 1:N

    earg(i,1) = -j*(2*pi*f_n(i,1)/3e8)*normtxp;

end
clear i;
%RV´s page 4290

phi_0 = unifrnd(0,2*pi);
t_0 = unifrnd(0,T_a);

Tau = zeros(L,1,'double');
Tau = poissrnd(T_T*1e9,L,1) * 1e-9 ;  %poissrnd

PL_0 = fspl(1,lambda);     %path loss in dB for one meter
ddplg_0 = - PL_0 - 10 * n_e *log10(d);  %distant dependent path loss gain

S = 0;

for l = 1:L

    S = S + exp(-Tau(l,1)/S_T);

end


pwrdelay = zeros(L,1);
%%alpha = zeros(L,1);
alpha = zeros(1,L);

for l = 1:L

    pwrdelay(l,1) = ddplg_0 / S * exp( -Tau(l,1) / S_T);

end

for l = 1:L
    alpha(1,l) = sqrt(pwrdelay(l,1)/2)*(randn(1,1) + 1i*randn(1,1));
end

%%alpha = sqrt(pwrdelay(1,1)/2)*(randn(1,L)+1*j*randn(1,L)); %CN distribution https://www.researchgate.net/post/How_can_I_generate_circularly_symmetric_complex_gaussian_CSCG_noise
%%alpha should be obtained using all of pwr

%eq 5
for i = 1:N
    hdp(i,1) = zeta_0 * (sqrt(G_T*G_R*TX_pwr_sub) * f_n(i,2))/(4*pi*normtxp) * exp(earg(i,1) + j*2*pi*f_n(i,1)*t_0 + j*phi_0);
end
clear i;
%should always be 0 because its in NLOS

%eq 6
for i = 1:N
    for l = 1:L
        %alpha is 1xl matrix
        hmp(i,1) = hmp(i,1) + alpha(1,l) * exp(-j*2*pi*f_n(i,1) * ( Tau(l,1) + normtxp/3e8 - t_0 ));
    end
end

clear i;

%eq 4
hd = hdp + hmp;

% figure();
% plot(abs(hd(:,1)));
% legend('hd = hmp absolute value');
% 
% figure();
% plot(angle(hd(:,1))*180/pi);
% legend('hd = hmp phase value');

zeta_k = 1;  %for now, assuming each tile is always i LOS to p

normtxk = zeros(K,1); %||ptx-p_k||

for i = 1:K
    normtxk(i,1) = norm(ptx-p_k(i,:));
end

clear i;

gdp = zeros(N,K);
gmp = zeros(N,K);

%eq 8
for n = 1:N
    for i = 1:K
        gdp(n,i) = zeta_k * (sqrt(G_T*G_R*TX_pwr_sub) .* f_n(n,2))/(4*pi*normtxk(i,1)) * exp(-j*2*pi*f_n(n,1)/3e8*normtxk(i,1) + j*2*pi*f_n(n,1)*t_0 + j*phi_0);
    end
end
clear i;
%same model as eq 6

figure()
surf(abs(gdp))
legend('absolute value of gdp');

for n = 1:N
    for i = 1:K
        for l = 1:L
            gmp(n,i) = gmp(n,i) + alpha(1,l) * exp(-j*2*pi*f_n(n,1) * ( Tau(l,1) + normtxk(i,1)/3e8 - t_0 ));
        end
    end
end
clear i;

figure()
surf(abs(gmp))
title('gmp absolute value')

% figure();
% plot(angle(gmp(:,50)*180/pi))
% legend('gmp phase value for sub50');

%eq7
gr = zeros(N,K);

gr = gmp + gdp;

figure()
surf(abs(gr))
title('gr absolute value')

normpk = zeros(K,1); % ||p - p_k||

for i = 1:K
    normpk(i,1) = norm(p-p_k(i,:));
end

clear i;

bdp = zeros(N,K);
bmp = zeros(N,K);
br = zeros (N,K);

%for now assuming eta(k) = 1 -> all tiles are in LOS of p
eta = 1;

%eq 10
for n = 1:N
    for i = 1:K
        bdp(n,i) = eta * sqrt(G_R)* f_n(n,2) / (4*pi*normpk(i,1)) * exp(-j*2*pi*f_n(n,1)/3e8*normpk(i,1));
    end
end
clear i;
%same model as eq 6 (assumed)
for n = 1:N
    for i = 1:K
        for l = 1:L
            bmp(n,i) = bmp(n,i) + alpha(1,l) * exp(-j*2*pi*f_n(n,1) * ( Tau(l,1) + normpk(i,1)/3e8 - t_0 ));
        end
    end
end
clear i;
%%the plots bellow prove (tested for ptx = p) that if gdp and gmp are done correctly,
%%then  bdp and bmp are too.
% figure()
% surf(abs(bdp))
% legend('abs of bdp')
% 
% figure()
% surf(abs(bmp))
% legend('abs of bmp')
% %eq 9

br = bdp + bmp;


figure()
surf(abs(br))
legend('abs of br')

hr = zeros(N,K);

hdpnk = zeros(N,K); %eq 13

%eq 12
for n = 1:N
    for i = 1:K
        hr(n,i) = gr(n,i) * br(n,i);
    end
end
clear i;

hdpnk = bdp .* gdp;

figure()
surf(abs(hr));
title('hr abs');

figure()
surf(abs(hdpnk));
title('hdpnk abs');
%ynt

y = zeros(N,T);
w = zeros(N,T);


for i = 1:N
    aux = sqrt(sig2/2)*(randn(1,T)+1*j*randn(1,T));
    w(i,:) = aux(1,:);

end
clear i;
%eq 14
for n = 1:N
    for t = 1:T

        sumBhr = 0;

        for i = 1:K
            sumBhr = sumBhr + BETAt_k(t,i) * hdpnk(n,i);%%to account for the multipath, use hr(n,i) instead of hdpnk
        end

        y(n,t) =  0 + sumBhr + w(n,t);  %%to account for the multipath, substitute 0 for hd(n,1), direct position section
    end
end
clear i;

figure()
surf(abs(y));                           
title('y abs');

yd = zeros(N,1);
w2 = sqrt(sig2/2)*(randn(1,N)+1*1i*randn(1,N));

%eq 17, could also be done with first expression of eq 16 ?
for n = 1:N
    yd(n,1) = hdp(n,1) + hmp(n,1) + w2(1,n);
end

yr = zeros(N,T);

%eq 18
for n = 1:N
    for t = 1:T
        yr(n,t) = y(n,t) - yd(n,1);
    end
end

clear i;

%Beta_0 defined between eq 19 and 20
Beta_0 = Ne * Gc;

%zn = sqrt(sig2/2)*(randn(1,N)+1*j*randn(1,N));

um = zeros(K,N);
auxvar = zeros(K,N);
%eq 21

for m = 1:K
    auxvar(m,:) = hr(:,m); %because hr is NxK and um is KxN
    um(m,:) = Beta_0*auxvar(m,:) + sqrt(sig2/2)*(randn(1,N)+1*1i*randn(1,N));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%OPTIMIZATION%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%This code pertains to the first algorithm, please ignore.
pestimation = zeros(N,3);
%x0 = [p_x+0.6 p_y+0.3 p_z+0.7];
%x0 = [0.2 0.5 1.3];
%options = optimset('MaxFunEvals',1000000000000000, 'TolX',1e-18);%v0
%opts =
%optimoptions(@optimize.globalsearch,'MaxFunctionEvaluations',1e5,'MaxIterations',1e3,
%'Display','off', 'Solver','fmincon');v1
%options = optimoptions('fmincon','MaxFunctionEvaluations',1e5,'MaxIterations',1e3, 'Display','off');

options = optimoptions('particleswarm','MaxIterations',10e99);

lb = [-3 0 0.6];
ub = [3 7 1.4];

%lb = [p_x-0.4 p_y-0.38 p_z-0.2];
%ub = [p_x+0.7 p_y+0.4 p_z+0.4];

searchspacevolume = abs(ub(1,1) - lb(1,1)) * abs(ub(1,2) - lb(1,2)) * abs(ub(1,3) - lb(1,3));

figure();
plot3(ptx(1,1),ptx(1,2),ptx(1,3), 'o', 'MarkerSize', 10);
hold on
plot3(p(1,1),p(1,2),p(1,3), '+k', 'MarkerSize', 10);
hold on
hold on
plot3(lb(1,1),lb(1,2),lb(1,3),'*','MarkerSize',10);
hold on
plot3(ub(1,1),ub(1,2),ub(1,3),'*','MarkerSize',10);
hold on

for i = 1:K
    plot3(p_k(i,1),p_k(i,2),p_k(1,3),'.', 'MarkerSize', 10);
    hold on
end

clear i;

legend('tx','UE','Lower Bound','Upper Bound','RIS');
title('3D View');

for n = 1:N
  %p_ = fminsearch(@(p_)functionslibv7.objective(p_,n,K,T,y,gdp,G_R,f_n,p_k,BETAt_k,sig2,phi_0,t_0),x0,options);
  %v0
  %p_ =
  %optimize.globalsearch(@(p_)functionslibv7.objective(p_,n,K,T,y,gdp,G_R,f_n,p_k,BETAt_k,sig2,phi_0,t_0),x0,opts);
  %v1
  %p_ = fmincon(@(p_)functionslibv7.objective(p_,n,K,T,y,gdp,G_R,f_n,p_k,BETAt_k,sig2,phi_0,t_0),x0,[],[],[],[],[],[],[],options);
  %v2
  p_ = particleswarm(@(p_)functionslibv7.objective(p_,n,K,T,y,gdp,G_R,f_n,p_k,BETAt_k,sig2,phi_0,t_0),3,lb,ub,options);
  pestimation(n,:) = p_(1,:);
end


pfinal = functionslibv7.outrem(pestimation,N);
error = abs(norm(p) - norm(pfinal)) * 100;   %% error in cm
fprintf("Error: " + error + "cm" + "\n")
fprintf("Search space volume: "+ searchspacevolume + 'm^3'+ "\n");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%Second Algorithm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


idft  = dsp.IFFT('FFTLengthSource','Property', ...
    'FFTLength',N*F_o); %definition of idft function, oversampled by a factor of F_o

%nota: ordenar melhores estimativas por ordem decrescente.

%eq 32
zd = idft(yd);
%zm = zeros(K,N*F_o);

um_ = transpose(um); % transposed um for idft, this is because idft only works for arrays (any number x 1)
%since um is KxN, its needed to use um transpose matrix in order to get IDFT relative to each k.
zm = zeros(N*F_o,K);

for m = 1:K
    zm(:,m) = idft(um_(:,m)); % this is the consequence of what was written in line 421
    %idft is stored in row m of matrix zm
end


%power calculation, this is for testing
P = zeros(K,1); %power
SNR = zeros(K,1);

for m  = 1:K
    P(m,1) = pow2db(sum(abs(zm(:,m)).^2)/(N*F_o));
end

for m  = 1:K
    SNR(m,1) = P(m,1) + 30 - sig2;
end


xaxis = [1:1:100];
%plot(xaxis,SNR) %plot of snr.

%below is commented, not needed right now



%
%zm = idft(um(1,:));
% zdtime(1,:) = zd(:,1);
% zmtime(1,:) = (zm(:,1));

%[m, Tau_d] = max(zdtime);
%[m, Tau_m] = max(zmtime);

%Tau_d = Tau_d * 100/(N*F_o) * 1e-9;

%3e8*(Tau_d) - norm(ptx-p)


%t_instant = 1/(N*F_o):100/(N*F_o):100;

%plot(t_instant,abs(zdtime));

%
%%%%%%%IGNORE%%%%%%%%%

% x0 = -5:0.1:5;
% y0 = 0:0.1:10;
% [X,Y] = meshgrid(x0,y0);
%
% ar = zeros(1,3);
%
%
% Z = zeros(101,101);
% for i = 1:101
%     for b = 1:101
%         ar(1,1) = x0(1,i);
%         ar(1,2) = y0(1,b);
%         ar(1,3) = 1;
%         Z(i,b) = functionslibv4.objective(ar,10,K,T,y,gdp,G_R,f_n,p_k,BETAt_k,sig2,phi_0,t_0);
%     end
% end
%
% surf(X,Y,Z)
% for t = 1:T
%     phit = ...
%
% end


%guardar

%err = (guardar ~= hdp)


% plot snr to tile (m)
return



