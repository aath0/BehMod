function [EV] = exvar(y, yhat)


%EV = 1-(var((y-yhat))/var(y));
%EV = var((y-yhat))/var(y);


SSreg = sum((yhat-mean(y)).^2);
SStot = sum((y-mean(y)).^2);
SSerr = sum((y-yhat).^2);

% EV = SSerr/SStot
% EV = 1-SSreg/SStot;

EV = 1-SSerr/length(y)/var(y);


if EV < 0
    EV
end