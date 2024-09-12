clc; clear all; close all;

files = ...
["Aer_EcAW_60K_TmEchoDEcay_32scansph"
 "Aer_EcAW_80K_TmEchoDEcay_16scans"
 "Aer_EcAW_100K_TmEchoDEcay"
 "Aer_EcAW_120K_TmEchoDEcayph"
 "Aer_EcAW_140K_TmEchoDEcayph"
 "Aer_EcAW_150K_TmEchoDEcayph"
 "Aer_EcAW_160K_TmEchoDEcayph"
 "Aer_EcAW_180K_TmEchoDEcayph"
 "Aer_EcAW_200K_TmEchoDEcay_16scans"
 "Aer_EcAW_220K_TmEchoDEcay_70scans"];

T = [60 80 100 120 140 150 160 180 200 220];

for n = 1:length(files)
    
[t,I] = eprload(files{n});

I = real(I);

f = fit(t,I,'exp1');

a(n) = f.a;
b(n) = f.b;

ci = confint(f,0.95);
b_upper(n) = ci(1,2); 
b_lower(n) = ci(2,2);
end

Tm = -2./b/1000;
Tm_lower = -2./b_upper/1000;
Tm_upper = -2./b_lower/1000;

err = 0.5*(Tm_upper-Tm_lower);

errorbar(T,Tm,(Tm-Tm_lower),Tm_upper-Tm)
ylim([0 10])
ylabel('T_m (\mus)');
xlabel('Temperature (K)')
ylabel('T_m (\mus)')

C = [T' Tm' err'];
C = round(C,4);
writematrix(C,'Aer_2H14N_repeat_T_Tm_TmErr.txt')