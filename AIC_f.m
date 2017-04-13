function [AIC] = AIC_f(RSS,k,n)

AIC = 2*k + n*log(RSS/n);
%AIC = 2*k - log(RSS);