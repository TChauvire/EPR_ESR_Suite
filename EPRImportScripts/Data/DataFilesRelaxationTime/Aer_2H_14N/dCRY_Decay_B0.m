%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WT
clc; clear all; close all;

decay_file = 'DCRY_23p9mgml_WT_150K_TmEchoDecay_B0.DTA';
FSE_file   = 'DCRY_23p9mgml_WT_150K_FSE2';

[A,I] = eprload(decay_file);

t = cell2mat(A(1,1));
B = cell2mat(A(1,2));

I = real(I);
span = [3:18];

for n = 1:length(span)
n = span(n);    
f = fit(t',I(:,n),'exp1');

a(n-span(1)+1) = f.a;
b(n-span(1)+1) = f.b;

ci = confint(f,0.95);
b_upper(n-span(1)+1) = ci(1,2); 
b_lower(n-span(1)+1) = ci(2,2);
end

[Bsw,Isw]=eprload(FSE_file);
yyaxis left
figure(1)
plot(Bsw,Isw)
xlim([min(Bsw) max(Bsw)])
xlabel('Field (G)')
ylabel('Intensity (a.u.)');

yyaxis right
Tm = -2./b/1000;
Tm_lower = -2./b_upper/1000;
Tm_upper = -2./b_lower/1000;
errorbar(B(span),Tm,(Tm-Tm_lower),Tm_upper-Tm)
ylim([0 4])
ylabel('T_m (\mus)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EN
clc; clear all;

decay_file = 'DCRY_20p1mgml_EN_150K_TmEchoDecay.DTA';
FSE_file   = 'DCRY_20p1mgml_EN_150K_FSE';

[A,I] = eprload(decay_file);

t = cell2mat(A(1,1));
B = cell2mat(A(1,2));

I = real(I);
span = [3:17];

for n = 1:length(span)
n = span(n);    
f = fit(t',I(:,n),'exp1');

a(n-span(1)+1) = f.a;
b(n-span(1)+1) = f.b;

ci = confint(f,0.95);
b_upper(n-span(1)+1) = ci(1,2); 
b_lower(n-span(1)+1) = ci(2,2);
end

[Bsw,Isw]=eprload(FSE_file);
yyaxis left
figure(2)
plot(Bsw,Isw)
xlim([min(Bsw) max(Bsw)])
xlabel('Field (G)')
ylabel('Intensity (a.u.)');

yyaxis right
Tm = -2./b/1000;
Tm_lower = -2./b_upper/1000;
Tm_upper = -2./b_lower/1000;
errorbar(B(span),Tm,(Tm-Tm_lower),Tm_upper-Tm)
ylim([0 4])
ylabel('T_m (\mus)');
