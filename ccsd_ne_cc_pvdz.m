% This Matlab code computes the ground state energy for neon atom from the Coupled Cluster Singles and Doubles (CCSD) method 
% using the cc-pvdz basis set. 
%
% The self-consistent field (SCF) calculation was carried out using the UNDMOL
% computational chemistry package at University of North Dakota; the
% two-electron integral (Ne_tei_data_cc_pvd.txt) data was obtained from the SCF calculation.  
%
% Refs: John F. Stanton, Jorgen Gauss, John D. Watts, and Rodney J.
% Bartlett, J. Chem. Phys. 94, 4334–4345 (1991); 
%
% Written by Tsogbayar Tsednee (PhD)
% Email: tsog215@gmail.com
% July 13, 2023 & University of North Dakota 
%
function [] = ccsd_ne_cc_pvdz
%
clear; clc; format long
%
dim2B = 28; % size of spin-orbital basis
tol = 1e-10;
%
tei_n = 5565;  % total number of the unique TEI. 
%
read_tei_data = fopen('Ne_tei_data_cc_pvdz.txt', 'r');
tei_data_n5 = textscan(read_tei_data, '%d %d %d %d %f');
%
p = zeros(tei_n,1); q = zeros(tei_n,1); r = zeros(tei_n,1); s = zeros(tei_n,1); vals = zeros(tei_n,1);
p(1:tei_n) = tei_data_n5{1};
q(1:tei_n) = tei_data_n5{2};
r(1:tei_n) = tei_data_n5{3};
s(1:tei_n) = tei_data_n5{4};
vals(1:tei_n) = tei_data_n5{5};
for i = 1:tei_n
    tei(p(i),q(i),r(i),s(i)) = vals(i);
    tei(q(i),p(i),r(i),s(i)) = vals(i);    
    tei(p(i),q(i),s(i),r(i)) = vals(i);    
    tei(q(i),p(i),s(i),r(i)) = vals(i);   
    %
    tei(r(i),s(i),p(i),q(i)) = vals(i);    
    tei(s(i),r(i),p(i),q(i)) = vals(i);        
    tei(r(i),s(i),q(i),p(i)) = vals(i);        
    tei(s(i),r(i),q(i),p(i)) = vals(i);            
end
%
E0 =  -128.488775551741;                                          % the SCf ground state energy
En_orb = [1.694558, -0.832097, ...
          1.694558, -0.832097, ...
          1.694558, -0.832097, ...
          5.196711, 5.196711, 5.196711, ...
          5.196711, 5.196711, 2.159425, -1.918798, ...
        -32.765635];  % An orbital energies
%
fs = kron(diag(En_orb), eye(2)); % Fock matrix elements in molecular spin-orbital basis sets
%
holes = [28, 27, 26, 25, 12, 11, 8, 7, 4, 3 ]; % hole states
particles  = [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 10, 9, 6, 5, 2, 1]; % virtial (particle) states 
%
Gamma = zeros(dim2B,dim2B,dim2B,dim2B);  % antisymmetrized two-electron integral (Gamma(p,q,r,s))
for p = 1:dim2B
    for q = 1:dim2B
        for r = 1:dim2B
            for s = 1:dim2B
                p1 = floor((p+1)/2);
                q1 = floor((q+1)/2);
                r1 = floor((r+1)/2);                
                s1 = floor((s+1)/2);
                val1 = tei(p1,r1,q1,s1) * (rem(p,2)==rem(r,2)) * (mod(q,2)==rem(s,2));
                val2 = tei(p1,s1,q1,r1) * (rem(p,2)==rem(s,2)) * (rem(q,2)==rem(r,2));
                Gamma(p,q,r,s) = val1 - val2;
            end
        end
    end
end
%
%%%
% Init empty T1 (ts) and T2 (td) arrays
ts = zeros(dim2B,dim2B);
td = zeros(dim2B,dim2B,dim2B,dim2B);

% Initial guess T2 --- from MP2 calculation!
for a = particles 
    for b = particles
        for i = holes 
            for j = holes
                td(a,b,i,j) = td(a,b,i,j) + Gamma(i,j,a,b)/(fs(i,i) + fs(j,j) - fs(a,a) - fs(b,b));
            end
        end
    end
end
%
% Make denominator arrays Dai, Dabij
% Equation (12) of Stanton
Dai = zeros(dim2B,dim2B);
for a = particles 
    for i = holes 
        Dai(a,i) = fs(i,i) - fs(a,a);
    end
end
%
% Stanton eq (13)
Dabij = zeros(dim2B,dim2B,dim2B,dim2B);
for a = particles 
    for b = particles 
        for i = holes 
            for j = holes
                Dabij(a,b,i,j) = fs(i,i) + fs(j,j) - fs(a,a) - fs(b,b);
            end
        end
    end
end
%
%%%
Niter = 100;
En_ccsd_old = 0.;
for iter = 1:Niter
    %
    [taus] = tau_eq9(td, ts, dim2B, particles, holes);
    [tau] = tau_eq10(td, ts, dim2B, particles, holes);
    %
    [Fae] = interm_eq3(fs, ts, Gamma, taus, dim2B, particles, holes);
    [Fmi] = interm_eq4(fs, ts, Gamma, taus, dim2B, particles, holes);
    [Fme] = interm_eq5(fs, ts, Gamma, dim2B, particles, holes);
    [Wmnij] = interm_eq6(ts, Gamma, tau, dim2B, particles, holes);
    [Wabef] = interm_eq7(ts, Gamma, tau, dim2B, particles, holes);
    [Wmbej] = interm_eq8(ts, Gamma, td, dim2B, particles, holes);
%
    [ts_new] = t1_eq1(fs, ts, Gamma, td, Fae, Fmi, Fme, Dai, dim2B, particles, holes);
    [td_new] = t1_eq2(ts, Gamma, td, Fae, Fmi, Fme, tau, Wabef, Wmbej, Wmnij, Dabij, dim2B, particles, holes);
    %
    ts = ts_new;
    td = td_new;
    %
    [En_ccsd_new] = ccsd_energy(fs, ts, td, Gamma, particles, holes)
    %
    if (abs(En_ccsd_new - En_ccsd_old) < tol )
        break
    end
    %    
    En_ccsd_old = En_ccsd_new;
end
%
En_ccsd = E0 + En_ccsd_new;
[E0, En_ccsd] % [-128.4887755517410  -128.6796369356081] % vs  [−128.48878, −128.67964] from Br. Verstichel et al., PHYSICAL REVIEW A 80, 032508 (2009). 



%%%
return
end

%%%
function [taus] = tau_eq9(td, ts, dim2B, particles, holes)
%  Stanton eq (9)
taus = zeros(dim2B,dim2B,dim2B,dim2B);
for a = particles 
    for b = particles 
        for i = holes 
            for j = holes
                taus(a,b,i,j) = td(a,b,i,j) + 0.5*(ts(a,i)*ts(b,j) - ts(b,i)*ts(a,j));
            end
        end
    end
end
%
return
end
%
%%%
function [tau] = tau_eq10(td, ts, dim2B, particles, holes)
%  Stanton eq (10)
tau = zeros(dim2B,dim2B,dim2B,dim2B);
for a = particles 
    for b = particles 
        for i = holes 
            for j = holes
                tau(a,b,i,j) = td(a,b,i,j) + ts(a,i)*ts(b,j) - ts(b,i)*ts(a,j);
            end
        end
    end
end
%
return
end
%
%%%
function [Fae] = interm_eq3(fs, ts, Gamma, taus, dim2B, particles, holes)
%
% Stanton eq (3)
Fae = zeros(dim2B,dim2B);
for a = particles
    for e = particles
        Fae(a,e) = (1. - (a == e))*fs(a,e);
    end
end
for a = particles
    for e = particles
        for m = holes
            Fae(a,e) = Fae(a,e) - 0.5 * fs(m,e) * ts(a,m);
        end
    end
end
for a = particles
    for e = particles
        for m = holes
            for f = particles
                Fae(a,e) =  Fae(a,e) + ts(f,m) * Gamma(m,a,f,e);
            end
        end
    end
end
for a = particles
    for e = particles
        for m = holes
            for f = particles
                for n = holes
                    Fae(a,e) =  Fae(a,e) - 0.5 * taus(a,f,m,n)*Gamma(m,n,e,f);
                end
            end
        end
    end
end
%%%
return
end

%
function [Fmi] = interm_eq4(fs, ts, Gamma, taus, dim2B, particles, holes)
%
% Stanton eq (4)
Fmi = zeros(dim2B,dim2B);
for a = particles
    for e = particles
        Fmi(a,e) = (1. - (a == e))*fs(a,e);
    end
end
for m = holes
    for i = holes
        for e = particles
            Fmi(m,i) = Fmi(m,i) + 0.5*ts(e,i)*fs(m,e);
        end
    end
end
for m = holes
    for i = holes
        for e = particles
            for n = holes
                Fmi(m,i) = Fmi(m,i) + ts(e,n)*Gamma(m,n,i,e);
            end
        end
    end
end
for m = holes
    for i = holes
        for e = particles
            for n = holes
                for f = particles
                    Fmi(m,i) = Fmi(m,i) + 0.5*taus(e,f,i,n)*Gamma(m,n,e,f);
                end
            end
        end
    end
end
%%%
return
end
%%%
%
function [Fme] = interm_eq5(fs, ts, Gamma, dim2B, particles, holes)
%
% Stanton eq (5)
Fme = zeros(dim2B,dim2B);
for m = holes
    for e = particles
        Fme(m,e) = fs(m,e);
    end
end
for m = holes
    for e = particles
        for n = holes
            for f = particles
                Fme(m,e) = Fme(m,e) + ts(f,n)*Gamma(m,n,e,f);
            end
        end
    end
end
%%%
return
end
%%%
%
function [Wmnij] = interm_eq6(ts, Gamma, tau, dim2B, particles, holes)
%
% Stanton eq (6)
Wmnij = zeros(dim2B, dim2B, dim2B, dim2B);
for m = holes
    for n = holes
        for i = holes
            for j = holes
                Wmnij(m,n,i,j) = Gamma(m,n,i,j);
            end
        end
    end
end
for m = holes
    for n = holes
        for i = holes
            for j = holes
                for e = particles
                    Wmnij(m,n,i,j) = Wmnij(m,n,i,j) + (ts(e,j)*Gamma(m,n,i,e) - ts(e,i)*Gamma(m,n,j,e));
                end
            end
        end
    end
end
for m = holes
    for n = holes
        for i = holes
            for j = holes
                for e = particles
                    for f = particles
                        Wmnij(m,n,i,j) = Wmnij(m,n,i,j) + 0.25*tau(e,f,i,j)*Gamma(m,n,e,f);
                    end
                end
            end
        end
    end
end
%%%
return
end
%%%
function [Wabef] = interm_eq7(ts, Gamma, tau, dim2B, particles, holes)
%
% Stanton eq (7)
Wabef = zeros(dim2B, dim2B, dim2B, dim2B);
for a = particles
    for b = particles  
        for e = particles
            for f = particles
                Wabef(a,b,e,f) = Gamma(a,b,e,f);
            end
        end
    end
end
for a = particles
    for b = particles  
        for e = particles
            for f = particles
                for m = holes
                    Wabef(a,b,e,f) = Wabef(a,b,e,f) - ts(b,m)*Gamma(a,m,e,f) + ts(a,m)*Gamma(b,m,e,f);
                end
            end
        end
    end
end
for a = particles
    for b = particles  
        for e = particles
            for f = particles
                for m = holes
                    for n = holes
                        Wabef(a,b,e,f) = Wabef(a,b,e,f) + 0.25*tau(a,b,m,n)*Gamma(m,n,e,f);
                    end
                end
            end
        end
    end
end
%%%
return
end
%%%
function [Wmbej] = interm_eq8(ts, Gamma, td, dim2B, particles, holes)
%
% Stanton eq (8)
Wmbej = zeros(dim2B,dim2B,dim2B,dim2B);
for m = holes
    for b = particles
        for e = particles
            for j = holes
                Wmbej(m,b,e,j) = Gamma(m,b,e,j);
            end
        end
    end
end
for m = holes
    for b = particles
        for e = particles
            for j = holes
                for f = particles
                    Wmbej(m,b,e,j) = Wmbej(m,b,e,j) + ts(f,j)*Gamma(m,b,e,f);
                end
            end
        end
    end
end
for m = holes
    for b = particles
        for e = particles
            for j = holes
                for n = holes
                    Wmbej(m,b,e,j) = Wmbej(m,b,e,j) - ts(b,n)*Gamma(m,n,e,j); 
                end
            end
        end
    end
end
for m = holes
    for b = particles
        for e = particles
            for j = holes
               for n = holes
                   for f = particles
                       Wmbej(m,b,e,j) = Wmbej(m,b,e,j) - (0.5*td(f,b,j,n) + ts(f,j)*ts(b,n))*Gamma(m,n,e,f); 
                   end
                end
            end
        end
    end
end
%%%
return
end
%%%
function [ts_new] = t1_eq1(fs, ts, Gamma, td, Fae, Fmi, Fme, Dai, dim2B, particles, holes)
%
% Stanton eq (1)
ts_new = zeros(dim2B,dim2B);
for a = particles
    for i = holes
        ts_new(a,i) = fs(i,a);
    end
end
for a = particles
    for i = holes
        for e = particles
            ts_new(a,i) =  ts_new(a,i) + ts(e,i)*Fae(a,e);
        end
    end
end
for a = particles
    for i = holes
        for m = holes
            ts_new(a,i) =  ts_new(a,i) -ts(a,m)*Fmi(m,i);
        end
    end
end
for a = particles
    for i = holes
       for m = holes
          for e = particles
              ts_new(a,i) =  ts_new(a,i) + td(a,e,i,m)*Fme(m,e);
           end
        end
    end
end
for a = particles
    for i = holes
       for m = holes
          for e = particles
              for f = particles
                  ts_new(a,i) =  ts_new(a,i) - 0.5*td(e,f,i,m) * Gamma(m,a,e,f);
              end
           end
        end
    end
end
for a = particles
    for i = holes
       for m = holes
          for e = particles
              for n = holes
                  ts_new(a,i) =  ts_new(a,i) -0.5*td(a,e,m,n) * Gamma(n,m,e,i);
              end
           end
        end
    end
end
for a = particles
    for i = holes
       for n = holes
          for f = particles
              ts_new(a,i) =  ts_new(a,i) - ts(f,n) * Gamma(n,a,i,f);
           end
       end
       ts_new(a,i) = ts_new(a,i)/Dai(a,i);
    end
end
%%%
return
end
%%%
function [td_new] = t1_eq2(ts, Gamma, td, Fae, Fmi, Fme, tau, Wabef, Wmbej, Wmnij, Dabij, dim2B, particles, holes)
%
% Stanton eq (2)
td_new = zeros(dim2B,dim2B,dim2B,dim2B);
for a = particles
    for b = particles
        for i = holes
            for j = holes
                td_new(a,b,i,j) = td_new(a,b,i,j) + Gamma(i,j,a,b);
            end
        end
    end
end
for a = particles
    for b = particles
        for i = holes
            for j = holes
                for e = particles
                    td_new(a,b,i,j) = td_new(a,b,i,j) + td(a,e,i,j)*Fae(b,e) - td(b,e,i,j)*Fae(a,e);
                end
            end
        end
    end
end
for a = particles
    for b = particles
        for i = holes
            for j = holes
                for e = particles
                    for m = holes
                        td_new(a,b,i,j) = td_new(a,b,i,j) - ...
                                          0.5*td(a,e,i,j)*ts(b,m)*Fme(m,e) + 0.5*td(b,e,i,j)*ts(a,m)*Fme(m,e);
                    end
                end
            end
        end
    end
end
for a = particles
    for b = particles
        for i = holes
            for j = holes
                for m = holes
                    td_new(a,b,i,j) = td_new(a,b,i,j) - td(a,b,i,m)*Fmi(m,j) + td(a,b,j,m)*Fmi(m,i);
                end
            end
        end
    end
end
for a = particles
    for b = particles
        for i = holes
            for j = holes
                for m = holes
                    for e = particles
                        td_new(a,b,i,j) = td_new(a,b,i,j) - ...
                                           0.5*td(a,b,i,m) * ts(e,j)*Fme(m,e) + 0.5*td(a,b,j,m)*ts(e,i)*Fme(m,e);
                    end
                end
            end
        end
    end
end
for a = particles
    for b = particles
        for i = holes
            for j = holes
                for e = particles
                    td_new(a,b,i,j) = td_new(a,b,i,j) + ts(e,i)*Gamma(a,b,e,j) - ts(e,j)*Gamma(a,b,e,i);
                 end
            end
        end
    end
end
for a = particles
    for b = particles
        for i = holes
            for j = holes
                for e = particles
                    for f = particles
                        td_new(a,b,i,j) = td_new(a,b,i,j) + 0.5*tau(e,f,i,j)*Wabef(a,b,e,f);
                    end
                 end
            end
        end
    end
end
for a = particles
    for b = particles
        for i = holes
            for j = holes
                for m = holes
                    td_new(a,b,i,j) = td_new(a,b,i,j) - ts(a,m)*Gamma(m,b,i,j) + ts(b,m)*Gamma(m,a,i,j);
                 end
            end
        end
    end
end
for a = particles
    for b = particles
        for i = holes
            for j = holes
                for m = holes
                    for e = particles
                        td_new(a,b,i,j) = td_new(a,b,i,j) + td(a,e,i,m)*Wmbej(m,b,e,j) - ts(e,i)*ts(a,m)*Gamma(m,b,e,j);
                        td_new(a,b,i,j) = td_new(a,b,i,j) - td(a,e,j,m)*Wmbej(m,b,e,i) + ts(e,j)*ts(a,m)*Gamma(m,b,e,i); 
                        td_new(a,b,i,j) = td_new(a,b,i,j) - td(b,e,i,m)*Wmbej(m,a,e,j) + ts(e,i)*ts(b,m)*Gamma(m,a,e,j);  
                        td_new(a,b,i,j) = td_new(a,b,i,j) + td(b,e,j,m)*Wmbej(m,a,e,i) - ts(e,j)*ts(b,m)*Gamma(m,a,e,i);                       
                    end
                 end
            end
        end
    end
end
for a = particles
    for b = particles
        for i = holes
            for j = holes
                for m = holes
                    for n = holes
                        td_new(a,b,i,j) = td_new(a,b,i,j) + 0.5*tau(a,b,m,n)*Wmnij(m,n,i,j);
                    end
                end
                td_new(a,b,i,j) = td_new(a,b,i,j)/Dabij(a,b,i,j);                
            end
        end
    end
end
%%%
return
end
%%%

function [En_ccsd] = ccsd_energy(fs, ts, td, Gamma, particles, holes)
%
En_ccsd = 0.;
for i = holes
    for a = particles
        En_ccsd = En_ccsd + fs(i,a)*ts(a,i);
    end
end
for i = holes
    for a = particles
        for j = holes
            for b = particles
                En_ccsd = En_ccsd + 0.25*Gamma(i,j,a,b)*td(a,b,i,j) + 0.5*Gamma(i,j,a,b)*(ts(a,i))*(ts(b,j));
            end
        end
    end
end
%%%
return
end




