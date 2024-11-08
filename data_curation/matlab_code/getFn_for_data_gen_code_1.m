function [Rs,Rp,Ts,Tp,As,Ap] = getFn_for_data_gen_code_1(isHex, d, w_vec, eps_met, eps0, eps1, eps2, eps4,ft,n,eps_SiO2,ii,...
    mm, hval, R1, G, n1, n2, lambda_vec, theta_list)

g_len = length(G);
theta_list_len = length(theta_list);
Avg_T_lambda = zeros(1,theta_list_len);

for nn = 1: g_len
    
    for tt = 1: theta_list_len
              
        hs = hval(1,ii);
        r1 = R1(mm);
        g= G(nn);
        st = g/2;
        a= 2*(r1+st);
        theta_deg = theta_list(tt);
        
        theta = theta_deg * pi/180;
%         theta = 0;
        h = r1+st+hs;
        k0 = (w_vec/197.4).*sqrt(eps0)*cos(theta);
        k1 = (w_vec/197.4).*sqrt(eps1-eps0*sin(theta)^2);
        k2 = (w_vec/197.4).*sqrt(eps2-eps0*sin(theta)^2);
        k4 = (w_vec/197.4).*sqrt(eps4-eps0*sin(theta)^2);
        
        
        e1= eps_met;  e2 = eps_SiO2;
        r2= r1+st;
        
        P = 1-(r1/r2)^3;
        ea= e1.*(3-2*P)+ 2*e2.*P;
        eb = e1.*P+ e2.*(3-P);
        
        chi_l = eps4.*(r2.^3).*(e2.*ea-eps4.*eb)./(e2.*ea+2.*eps4.*eb);
                   
        
        if isHex == 1
            Ua = 11.03354;  % for hexagonal lattice
            beta_p = chi_l./(1 + chi_l.*(1/eps4).*((-0.5*Ua/a^3) )); % parallel
            beta_o = chi_l./(1 + chi_l.*(1/eps4).*(Ua/a^3)); % perpendicular
            f=1.732/2;
        else
            Ua = 9.031;  % for sqaure lattice
            beta_p = chi_l./(1 + chi_l.*(1./eps4).*((-0.5*Ua./a.^3) )); % parallel
            beta_o = chi_l./(1 + chi_l.*(1./eps4).*(Ua./a.^3 )); % perpendicular
            f=1;
        end
        
        eps3_p = eps4 + 4*pi*beta_p/(f*d*a^2);
        eps3_o = eps4.^2./(eps4-4*pi*beta_o/(f*d*a^2));
              
        
        k3_p = (w_vec/197.4).*sqrt(eps3_p-eps0*sin(theta)^2);
        k3_o = (w_vec/197.4).*sqrt(eps3_p./eps3_o).*sqrt(eps3_o-eps0*sin(theta)^2);
        
            
        s1n = exp(1i*k1*ft);
        s2n = exp(1i*k2*(h-d/2));
        s3n_p = exp(1i*k3_p*d);
        s3n_o = exp(1i*k3_o*d);
        
        
        r01p = (eps0.*k1-eps1.*k0)./(eps0.*k1+eps1.*k0);
        r12p = (eps1.*k2-eps2.*k1)./(eps1.*k2+eps2.*k1);
        r23p = (eps2.*k3_o-eps3_p.*k2)./(eps2.*k3_o+eps3_p.*k2);
        r34p = (eps3_p.*k4-eps4.*k3_o)./(eps3_p.*k4+eps4.*k3_o);
        
        r01s = (k0-k1)./(k0+k1);
        r12s = (k1-k2)./(k1+k2);
        r23s = (k2-k3_p)./(k2+k3_p);
        r34s = (k3_p-k4)./(k3_p+k4);
        
        
        t01s = (2*k0)./(k0+ k1);
        t12s = (2*k1)./(k1+ k2);
        t23s = (2*k2)./(k2+ k3_p);
        t34s = (2*k3_p)./(k3_p+ k4);
        
        t01p = (2*sqrt(eps0).*sqrt(eps1).*k0)./(k0.*eps1+ k1.*eps0);
        t12p = (2*sqrt(eps1).*sqrt(eps2).*k1)./(k1.*eps2+ k2.*eps1);
        t23p = (2*sqrt(eps2).*sqrt(eps3_p).*k2)./(k2.*eps3_p+ k3_o.*eps2);
        t34p = (2*sqrt(eps3_p).*sqrt(eps4).*k3_o)./(k3_o.*eps4+ k4.*eps3_p);
        
        
        rs= (((r12s.*s1n)./s2n + r01s./(s1n.*s2n))./s3n_p + r34s.*(s3n_p.*(s1n.*s2n + (r01s.*r12s.*s2n)./s1n) +...
            r23s.*s3n_p.*((r12s.*s1n)./s2n + r01s./(s1n.*s2n))) + (r23s.*(s1n.*s2n + (r01s.*r12s.*s2n)./s1n))./s3n_p)./ ...
            ((1./(s1n.*s2n) + (r01s.*r12s.*s1n)./s2n)./s3n_p + r34s.*(s3n_p.*((r12s.*s2n)./s1n + r01s.*s1n.*s2n) + ...
            r23s.*s3n_p.*(1./(s1n.*s2n) + (r01s.*r12s.*s1n)./s2n)) + (r23s.*((r12s.*s2n)./s1n + r01s.*s1n.*s2n))./s3n_p);
        
        rp= (((r12p.*s1n)./s2n + r01p./(s1n.*s2n))./s3n_o + r34p.*(s3n_o.*(s1n.*s2n + (r01p.*r12p.*s2n)./s1n) + r23p.*s3n_o.*((r12p.*s1n)./s2n + ...
            r01p./(s1n.*s2n))) + (r23p.*(s1n.*s2n + (r01p.*r12p.*s2n)./s1n))./s3n_o)./ ...
            ((1./(s1n.*s2n) + (r01p.*r12p.*s1n)./s2n)./s3n_o + r34p.*(s3n_o.*((r12p.*s2n)./s1n + r01p.*s1n.*s2n) + r23p.*s3n_o.*(1./(s1n.*s2n) + ...
            (r01p.*r12p.*s1n)./s2n)) + (r23p.*((r12p.*s2n)./s1n + r01p.*s1n.*s2n))./s3n_o);
        
        
        ts= (t01s.*t12s.*t23s.*t34s)./((1./(s1n.*s2n) + (r01s.*r12s.*s1n)./s2n)./s3n_p + r34s.*(s3n_p.*((r12s.*s2n)./s1n + r01s.*s1n.*s2n) + ...
            r23s.*s3n_p.*(1./(s1n.*s2n) + (r01s.*r12s.*s1n)./s2n)) + (r23s.*((r12s.*s2n)./s1n + r01s.*s1n.*s2n))./s3n_p);
        
        tp= (t01p.*t12p.*t23p.*t34p)./((1./(s1n.*s2n) + (r01p.*r12p.*s1n)./s2n)./s3n_o + r34p.*(s3n_o.*((r12p.*s2n)./s1n + r01p.*s1n.*s2n) + ...
            r23p.*s3n_o.*(1./(s1n.*s2n) + (r01p.*r12p.*s1n)./s2n)) + (r23p.*((r12p.*s2n)./s1n + r01p.*s1n.*s2n))./s3n_o);
                
        
        Rs=100.*abs(rs).^2;
        Rp=100.*abs(rp).^2;
        
        
        theta_t = asin(n*sin(theta)/sqrt(eps4));
        Ts=100.*abs(ts.^2) .* (sqrt(eps4)./n) .*cos(theta_t)/cos(theta);
        Tp=100.*abs(tp.^2) .* (sqrt(eps4)./n) .*cos(theta_t)/cos(theta);
        
        T_unpol = (Ts+Tp)/2;
        Avg_T_lambda(tt) = mean(T_unpol);
        
        
        As = 100-Rs-Ts;
        Ap = 100-Rp-Tp;
        
        Ts_mod = round(Ts,3);
        Tp_mod = round(Tp,3);
        Rs_mod = round(Rs,3);
        Rp_mod = round(Rp,3);
%         As_mod = round(As,3);
%         Ap_mod = round(Ap,3);
        
        %config details
        met = "Au";
        shape = isHex;
        N1 = n1;
        N2 = n2;
        rad = r1;
        gap = g;
        height = hs;
        lambd = lambda_vec;
        %config details end
        
         
        mgs_varName = {'met', 'isHex', 'n2', 'n1', 'height', 'rad', 'gap'}; %creating output array 
        trs_varName = {'lambda_val', 'Ts', 'Tp', 'Rs', 'Rp'};
%         writematrix(output,file_name,'WriteMode','append')  
        
        mgs_table = table(met, shape, N2, N1, height, rad, gap, 'VariableNames',mgs_varName);
        trs_table = table(lambd', Ts_mod', Tp_mod', Rs_mod', Rp_mod', 'VariableNames', trs_varName);
       
        mgs_table = repmat(mgs_table,2101,1);
        
        OUTPUT = [mgs_table, trs_table];

