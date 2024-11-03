clear all;
close all;
clc;
tic

hbc = 197.4; % hbar*c (eV*nm)

LED_Min_lam = 400; LED_Max_lam = 500; %min and max of lambda

lambda_vec = linspace(LED_Min_lam,LED_Max_lam, LED_Max_lam-LED_Min_lam+1); %in nm
w_vec = 1240./(lambda_vec); % w in eV, lambda in nm

A1= []; Tavg=[]; T_opt=[]; Tavg2=[]; T_theta =[];


%% Worked with this
hval = 1:1:10;  R1 = 5:1:25; ST = 1:1:10; theta_list = 0:1:26;

L = size(hval,2);
R1_len = length(R1);


parfor ii = 1: L
    T_theta_temp = zeros(1000);
    avg_T = zeros(1000);
    Tavg_temp = zeros(1000);
    for mm = 1: R1_len
        
        %% JC Ag Permittivity
        
        wave= [397.4	413.3	430.5	450.9	471.4	495.9	520.9	548.6	582.1	616.8	659.5	704.5	756	821.1	892	984	1088	1216 ...
            1393	1610	1937];
        Re_eps_AgJC_values = [-4.2824	-5.173125	-6.059844	-7.058049	-8.228661	-9.564149	-11.046476	-12.855796	-14.881664 ...
            -17.235504 -20.094789	-23.404644	-27.477664	-32.796929	-39.839744	-48.886464	-60.760425	-77.925484	-101.9931 ...
            -140.4	-198.1888];
        Im_eps_AgJC_values = [0.207	0.2275	0.19696	0.21256	0.2869	0.3093	0.3324	0.43032	0.3858	0.49824	0.4483	0.38704	0.31452 ...
            0.45816	0.50496 0.55936	0.6236	1.58904	2.626	3.555	6.7584];
        
        Re_AgJC = spline(wave,Re_eps_AgJC_values,lambda_vec);
        Im_AgJC = spline(wave,Im_eps_AgJC_values,lambda_vec);
        
        eps_AgJC= Re_AgJC+ 1i*Im_AgJC;
        
        
        wave1 =[391.7	405.5	419.7	434.5	449.8	465.7	482.1	499.1	516.7	534.9	553.7	573.2	593.4	614.3	636	658.4	681.6...
            705.6	730.5	756.2	782.9	810.4	839	868.6	899.2	930.8	963.6	997.6	1033	1069	1107	1146	1186	1228 ...
            1271 	1316	1362	1410	1460	1512	1565	1620	1677	1736	1797	1861	1926];
        Re_eps_SiO2_values = [2.163995494	2.159514453	2.155381684	2.151516381	2.147927169	2.144572889	2.141457419	2.138544848	2.135821462 ...
            2.133273721	2.130888413	2.128641786	2.126524526	2.124527233	2.122632208	2.120840005	2.119134398	2.117508175	2.115948075	2.114453822 ...
            2.113007772	2.111614727	2.110253831	2.108924981	2.107622729	2.106341582	2.105068436	2.103798634	2.102520188	2.101256345 ...
            2.099953011	2.098639875	2.097311216	2.095928656	2.094519762	2.093046073	2.091534505	2.089946073	2.088273972	2.086510881	2.08468377 ...
            2.082750988	2.080704053	2.078533991	2.076231331	2.073746706	2.071147156];
        
        eps_SiO2 = spline(wave1,Re_eps_SiO2_values,lambda_vec);
        
        %%
        eps_AlGaInP = 3.49^2;
        
        %eps_InGaN = 2.59^2;
        
        
        
        eps_med = 1.58^2;
        
        eps0 = eps_AlGaInP;%eps_AlGaInP;
        eps1 = eps_AlGaInP; % can be a film of thickness = ft
        eps2 = eps_med; % layer between that film and NP layer thickness = hs
        eps_met = eps_AgJC;  % NP material, layer with thickness d =1
        eps4 = eps_med; % NP surrounding medium
        
        isHex=1; % 1 if HEX array, 0 if SQR array
        
        ft = 0;
        %hs=  215; % from bottom face of the film (with thickness ft) to the top of NP surface
        d = 1;
        
        % Assuming the NP layer as a 1nm thick equivalent film, this is h
        
        
        
        %d = 4*pi*R0.^3./(3*a.^2); % For HEX or SQR array, d is same
        %h = d/2 + hs; % This does not work, I checked with SIMULATION
        %h = R0 + hs;  % This works very well with Simulation %R0 - d/2 + hs;
        
        
        [Rs,Rp,Ts,Tp,As,Ap,theta_deg, hs,r1,st] = Bappa_modified_parforGET_code(isHex, d, w_vec, eps_met, eps0, eps1, eps2,...
            eps4, ft,sqrt(eps0),eps_SiO2, ii, mm,  hval, R1, ST, theta_list);
        
        
        %T=Tp';
        Ts;
        Tp;
        T_unpol = (Ts+Tp)/2;
        %Avg_T = (min(T)+max(T))/2;
        Avg_T = mean(T_unpol);
        
        theta_deg;
        
        %fetching the size of each to estimate the number of zeros
        %to be appended to concat array for assigning value to
        %T_theta_temp
        ts_size = length(Ts);
        Tp_size = length(Tp);
        Avg_T_size = length(Avg_T);
        theta_deg_size = length(theta_deg);
        T_theta_size = size(T_theta_temp);
        concat_size = T_theta_size(2) -(ts_size + Tp_size + Avg_T_size + theta_deg_size); %number of zeros to append
        concat_array = [theta_deg Ts Tp Avg_T zeros(1,concat_size)];
        
        T_theta_temp(mm,:,ii) = concat_array; %multi-dimensional array. Different page for each ii; diff row for each mm; each row is a row matrix
        
    end
    
    avg_T(ii,:) = mean(T_theta_temp(:,:,ii)); %each row in avg_T = mean of each page of T_theta_temp
    %each column in each row of avg_T = mean of each column(of
    %each page) of T_theta_temp
    
    if avg_T(:,end) > 83.9194
        addendum_array = [hs r1 st avg_T zeros(1,(300-207))];
        Tavg_temp(ii,:) = addendum_array;
        
    end
    
    T_theta(:,:,ii) = T_theta_temp(:,:,ii); %copying values from temporary variable to global variable T_theta
    Tavg(ii,:) = Tavg_temp(ii,:); %---do--- Tavg
    Tavg2(ii,:) = avg_T(ii,:); %---do--- Tavg2
    % [Tval,I] = max(Tavg(:,2));
    %  g_val = Tavg(:,1);
    %  g_opt = g_val(I);
    
    
end

%T_opt = [T_opt; hs R0 g_opt Tval];
%T_opt = [T_opt; L0 g_opt Tval]


% ind1 = T_opt(:,4) > 0;
% A1 = T_opt(ind1,:)
%
% [maxT_opt,yy] = max(A1(:,4));
% h_temp = A1(:,1);
% hs_opt = h_temp(yy);
% R0_temp = A1(:,2);
% R0_opt = R0_temp(yy);
% g0_temp = A1(:,3);
% g0_opt = g0_temp(yy);

toc

% figure
%
% plotyy(T_opt(:,1),T_opt(:,3), T_opt(:,1),T_opt(:,2))
