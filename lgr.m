a = zpk([],[0 -2 -4], 1);
b = tf([1], [1 -4 9]);
c = a*b;
rlocus(c);