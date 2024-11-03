clear all;
close all;
clc;
tic

hbc = 197.4; % hbar*c (eV*nm)

LED_Min_lam = 400; LED_Max_lam = 1000; %min and max of lambda

lambda_vec = linspace(LED_Min_lam,LED_Max_lam, LED_Max_lam-LED_Min_lam+1); %in nm
w_vec = 1240./(lambda_vec); % w in eV, lambda in nm

% Worked with this
hval = 0:1:10;  R1 = 5:1:150; G = 2:1:200;

theta_list = 0:1:26; %upper limit is subject to change
 

L = size(hval,2);
R1_len = length(R1);


parfor ii = 1: L
          
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
     
        %% JC Au Permittivity
         
        %{
        wave = [397.4	413.3	430.6	450.9	471.5	496.0	521.0	548.7	582.2	616.9	659.6	704.5	756.1	821.2	892.1	984.1	1087.7 ...
            1215.7	1393.3	1610.4	1937.5]; 
        Re_eps_AuJC_values = [-1.64940	-1.70216	-1.69220	-1.75900	-1.70270	-2.27829	-3.94616	-5.84213	-8.11267	-10.66190 ...
            -13.64820	-16.81770	-20.60110	-25.81130	-32.04070	-40.27410	-51.04960	-66.21850	-90.42650	-125.35100	-189.04200];
        Im_eps_AuJC_values = [5.73888	5.71736	5.6492	5.28264	4.84438	3.81264	2.58044	2.1113	1.66054	1.37424	1.03516	1.06678	1.27148	1.62656 ...
            1.92542	2.794	3.861	5.7015	8.18634	12.5552	25.3552];
        
        Re_AuJC = spline(wave,Re_eps_AuJC_values,lambda_vec);
        Im_AuJC = spline(wave,Im_eps_AuJC_values,lambda_vec);
        
        eps_AuJC= Re_AuJC+ 1i*Im_AuJC;
        %}
        
        
        %% JC TiN Permittivity        
        %{
        wave = [310 354.2 413.3 495.9 619.9 826.6 1239.9 2479.7];
        Re_eps_TiNJC_values = [3.64 3.456 1.946 -1.2825 -5.4937 -11.2037 -18.1655 -47.6321];
        Im_eps_TiNJC_values = [5.5842 4.5368 3.6192 3.96 7.1016 13.8684 27.1152 74.58];
        
        Re_TiNJC = spline(wave,Re_eps_TiNJC_values,lambda_vec);
        Im_TiNJC = spline(wave,Im_eps_TiNJC_values,lambda_vec);
        
        eps_TiNJC= Re_TiNJC+ 1i*Im_TiNJC;
         %}
        
        %% JC Cu Permittivity
        %{
        wave = [397.4 413.3 430.5 450.9 471.4 495.9 520.9 548.6 582.1 616.8 659.5 704.5 756 821.1 892 984 1088 1216 1393 1610 1937];        
        Re_eps_CuJC_values = [-2.735056 -3.232449 -3.750525 -4.208009 -4.602789 -5.085696 -5.409264 -5.600529 -6.821616 -10.182025 -13.991609 ...
            -17.637925 -21.704625 -26.7648 -33.179824 -41.126841 -51.955489 -67.749625 -88.734721 -123.0768 -179.1768 ];        
        Im_eps_CuJC_values = [5.58624 5.64992 5.7625 5.94456 6.2075 6.25616 6.15488 5.25708 3.7856 1.923 1.64868 1.7661 2.2392 2.6936 3.4608 ...
            4.10944 5.19624 7.9152 11.3268 16.9024 29.2774];
        
        Re_CuJC = spline(wave,Re_eps_CuJC_values,lambda_vec);
        Im_CuJC = spline(wave,Im_eps_CuJC_values,lambda_vec);
        
        eps_CuJC= Re_CuJC+ 1i*Im_CuJC;
        %}
        
        %% ZrN Permittivity
        %{
        wave = [300 400 500 600 700 800 900 1000 1200 1500];
        Re_eps_ZrN_values = [3.0613 -0.442 -3.8779 -8.2745 -13.356 -18.5504 -23.9412 -30.912 -67.0672 -111.945600000000];
        Im_eps_ZrN_values = [2.4684 1.4112 1.818 2.5608 3.7638 5.307 6.7184 9.2168 28.0896 57.116];
        
        Re_ZrN = spline(wave,Re_eps_ZrN_values,lambda_vec);
        Im_ZrN = spline(wave,Im_eps_ZrN_values,lambda_vec);
        
        eps_ZrN= Re_ZrN+ 1i*Im_ZrN;
        NP_mater = "ZrN";
        %}
        
        
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
%         eps_AlGaInP = 3.49^2;
        
        %eps_InGaN = 2.59^2;
        n1 = 1; % 1 for air, 1.58 for epoxy
        n2 = 2.5; % 2.5, 2.8, 3.1, 3.4, 3.7, 4.0, 4.3
        isHex=0; % 1 if HEX array, 0 if SQR array
        
        eps_semicon = n2^2;
        eps_med = n1^2;
        
        eps0 = eps_semicon;%eps_AlGaInP;
        eps1 = eps_semicon; % can be a film of thickness = ft
        eps2 = eps_med; % layer between that film and NP layer thickness = hs
        eps_met = eps_AuJC;  % NP material, layer with thickness d =1
        eps4 = eps_med; % NP surrounding medium
        
        
        
        ft = 0;
        %hs=  215; % from bottom face of the film (with thickness ft) to the top of NP surface
        d = 1;
        
        % Assuming the NP layer as a 1nm thick equivalent film, this is h
            
        
        [Rs,Rp,Ts,Tp,As,Ap] = getFn_for_data_gen_code_1(isHex, d, w_vec, eps_met, eps0, eps1, eps2, eps4,ft,sqrt(eps0),eps_SiO2,ii,...
    mm, hval, R1, G, n1, n2, lambda_vec, theta_list);
        
    end
end
fclose('all');
toc
