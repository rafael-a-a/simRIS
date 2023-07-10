classdef functionslibv7
   methods (Static)
      function res = func1(a)
         res = a * 5;
      end
      
      function res = func2(x)
         res = x .^ 2;
      end

      function [azimuth, elevation, r] = solidangle(x,y,z)

        azimuth = atan2(z,x);
        elevation = atan2(y,sqrt(x.^2 + z.^2));
        r = sqrt(x.^2 + y.^2 + z.^2);

      end

      function af = arrayfact(phi, theta, lambda,Ne, N_x, N_y, d_x, d_y)
          d1 = sin(( pi*N_x*d_x )./ lambda .* sin(theta) .* sin(phi) );
          D1 = sin(( pi*d_x )./ lambda .* sin(theta) .* sin(phi) );
          d2 = sin(( pi*N_y*d_y )./ lambda .* sin(theta) .* cos(phi) );
          D2 = sin(( pi*d_y )./ lambda .* sin(theta) .* cos(phi) );

%           if  ( sin(theta)*sin(phi) < 1e-8 )
%               D1 = ( pi*d_x )./ lambda .* sin(theta) .* sin(phi);
%               
%               af = 1/sqrt(Ne) * d1./D1 .* d2./D2;
          %versão luís    

          if (theta < 1e-8)
              af = 1/sqrt(Ne) * N_x * N_y; 
              %Para theta = 0, limite da expressão é a acima (linha 32)
          %versão rafael
          
          elseif ( sin(theta)*cos(phi) < 1e-8 )
              
              D2 = ( pi*d_y )./ lambda .* sin(theta) .* cos(phi);
              af = 1/sqrt(Ne) * d1./D1 .* d2./D2;
          else
              af = 1/sqrt(Ne) * d1./D1 .* d2./D2;
              
          
          end

      end

      function npr = normpwr(phi, theta)
          if ( theta >= 0 ) && ( theta <= pi/2 )

              q = 0.57;
              npr = (cos(theta)) ^ q;
          else
              npr = 0;

          end


      end

      function psi = refphase()
          psi = randi([0,1],1) * pi;    % randi:  returns an array containing integers drawn from the discrete uniform distribution on the interval [imin,imax], using any of the above syntaxes.
   
      
      end

      function betatk = reflectioncoeff(npri, nprr, afi, afr,psitk)

          betatk = sqrt(npri * nprr) * afi * afr * pi * exp(1i*psitk); %Gc=pi
      end

      function phat = objective(p_,n,K,T,y,gdp,G_R,f_n,p_k,BETAt_k,sig2,phi0,t0)   %in main, call this function n times.

          eta = 1;
          bnkdp = zeros(1,K);
          gnkdp = zeros(1,K);
          %hnkdp = zeros(1,K);
          phi_t = zeros(T,1);



          for i = 1:K
              bnkdp(1,i) = eta * sqrt(G_R) * f_n(n,2) / (4*pi*norm(p_- p_k(i,:))) * exp(-j*2*pi*f_n(n,1)/3e8*norm(p_- p_k(i,:)));
          end



          for i = 1:K
              gnkdp(1,i) = gdp(n,i);
          end



          
          hnkdp = gnkdp .* bnkdp;
         


          for t = 1:T
              sumbh = 0;
              for i = 1:K
                  sumbh = sumbh + BETAt_k(t,i) * hnkdp(1,i);
              end
              phi_t(t,1) = angle(sumbh) - 2*pi*t0 - phi0;
          end
          
          sump = 0;
          for t = 1:T
              sump = sump + abs(y(n,t)^2)/sig2^2 * sin(angle(y(n,t)) - phi_t(t,1) - phi0)^2;
          end

          phat = sump;

      end
      
      
      function finalp = outrem(pestimation,N)

          if N == 1
              finalp = pestimation

          else

              av = mean(pestimation);
              sd = std(pestimation);
              P = 0;


              for n = 1:N
                  c = (pestimation(n,:) <= av + sd);
                  d = (pestimation(n,:) >= av - sd);
                  f = all(c == 1);
                  g = all(d == 1);
                  if f && g
                      P = P + 1;
                  end
                  
              end
                
             
              if P ~= 0
                  padd = zeros(P,3);

                  cont = 1;

                  for n = 1:N
                      c = (pestimation(n,:) < av +  sd);
                      d = (pestimation(n,:) > av -  sd);
                      f = all(c == 1);
                      g = all(d == 1);
                      if f && g && (cont <= P)
                          padd(cont,:) = pestimation(n,:);
                          cont = cont + 1;
                      end
                  end


                  padd
                  finalp = mean(padd);
              else
                  finalp = mean(pestimation);
              end
         

          end
      end
      
   end
end

