clear; 
close all; 
clc;

% Gc(s) = (s+c)/s 
c = 2;
number_c = [1 c];
c_den = [1 0];
Gc = tf(number_c, c_den);

% Gp(s) = 10/((s+1)*(s+9)) = 10/(s^2+10*s+9)
number_p = [10];
p_den = [1 10 9];
Gp = tf(number_p, p_den);

SystemOpenLoop = series(Gc, Gp);

% Root locus plot
figure;
rlocus(SystemOpenLoop)
Z = 0.5911;
wn = 1.8/1.2;
sgrid(Z,wn)
[k, poles] = rlocfind(SystemOpenLoop);


Kp = 0.8; % from rlocus
Ki = c * Kp;
SystemOpenLoop = Kp * SystemOpenLoop;
SystemClosedLoop = feedback(SystemOpenLoop, 1, -1);

% Step response
figure;
step(SystemClosedLoop);
Information = stepinfo(SystemClosedLoop);
text(1,0.25,['Rise Time:',num2str(Information.RiseTime)]);
text(1,0.2,['Overshoot: ',num2str(Information.Overshoot)]);
disp(['Rise Time of step response is: ',num2str(Information.RiseTime)]);
disp(['Overshoot of step response is: ',num2str(Information.Overshoot)]);


% Display gains
disp('Integral and proportional gains of the conventional PI controolers are:');
disp(['Kp = ',num2str(Kp)]);
disp(['K_i = ',num2str(Ki)]);