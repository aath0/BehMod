function [BIC] = BIC_f(RSS,k,n)

BIC = k*log(n) + n*log(RSS/n);
%BIC = k*log(n) -2*log(RSS);
