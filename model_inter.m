function [ls,ydata,xdata,Q1,Q2] = model_inter(x,cs,us,xdata,model,out_f,incl_us)

% Model interface function. Gives predictions for Reinforcement Learning,
% Bayesian and Null models. 

% Inputs:

%   x: vector of parameters of each model, the first 2 entries always
%       correspond to the mapping function (slope+intersect). Additional entries
%       in the vector, if any, will be the models' parameters.

%   cs: vector with the index of each trial: 1/2, for our two stimuli (CS+/-)
%   us: vector with the index of each trial: 1/0, for the presence of a reinforcer (US+/-)
%   xdata: vector with physiological estimates per trial, that will be modeled.
%   model: the label of the desired model.
%   out_f: mapping function, from model's estimates to physiological data
%   incl_us: will US+ trials be used in estimating model evidence? recommended: 0 (1/0)


% Outputs:

%   ls: RSS estimate
%   ydata: Model estimates
%   xdata: physiological estimates
%   Q1, Q2: Model estimates before the mapping

% First created on 02.12.2014 by Athina Tzovara, University of Zurich,
% Switzerland.
% Last updated on 13.04.2017 by Athina Tzovara.


beta_prior = 1;

switch model
    case 'RW1'
        init = 0.5;
        %initialize values for CS+ and CS-:
        Q1 = init*ones(size(us));
        Q2 = init*ones(size(us));
        
        %update rule for every trial:
        for i = 1:length(us)-1
            if cs(i) == 1 %cs+, update cs+ representation:
                Q1(i+1:end) = Q1(i)+x(end)*(us(i)-Q1(i));
                Q2(i+1:end) = Q2(i);
            else %update cs- representation:
                Q2(i+1:end) = Q2(i)+x(end)*(us(i)-Q2(i));
                Q1(i+1:end) = Q1(i);
            end
        end
        
    case 'PH1'
        
        initQ = 0;
        inita = 0;
        
        %initialize values for CS+ and CS-:
        Q1 = initQ*ones(size(us));
        Q2 = initQ*ones(size(us));
        a1 = inita*ones(size(us));
        a2 = inita*ones(size(us));
        lamda1 = ones(size(us));
        lamda2 = zeros(size(us));
        %update rule for every trial:
        for i = 1:length(us)-1
            
            if cs(i) == 1 %cs+, update cs+ representation:
                
                a1(i+1:end) = abs(us(i)-Q1(i));
                Q1(i+1:end) = Q1(i)+x(end)*us(i)*a1(i);

            else
                
                a2(i+1:end) = abs(us(i)-Q2(i));
                Q2(i+1:end) = Q2(i)+x(end)*us(i)*a2(i);
                
            end
        end
        
        
    case 'BM0'
        trials = length(cs);
        init = beta_prior;
        A = init;
        B = init;
        A1 = init;
        B1 = init;
        Q1 = (A/(A+B))*ones(trials,1);
        Q2 = (A1/(A1+B1))*ones(trials,1);
        
        for tr = 1:trials
            
            if cs(tr) == 1
                
                Q1(tr+1:end) = (sum(us(1:tr)) + A)/(length(find(cs(1:tr)== 1)) + A + B);
                
            else
                
                Q2(tr+1:end) = (A1)/(length(find(cs(1:tr)== 2)) + A1 + B1);
                
            end
        end
        
    case 'BH2'
        trials = length(cs);
        init = beta_prior;
        A = init*ones(trials+1,1);
        B = init*ones(trials+1,1);
        A1 = init*ones(trials+1,1);
        B1 = init*ones(trials+1,1);
        Q1 = (A(1)/(A(1)+B(1)))*ones(trials,1);
        Q2 = (A1(1)/(A1(1)+B1(1)))*ones(trials,1);
        
        for tr = 1:trials
            st = x(end);
            ra2 = st*tr;
            
            if cs(tr) == 1
                
                %model update:
                A(tr+1:end) = A(tr)+us(tr);
                B(tr+1:end) = B(tr)-us(tr)+1;
                
                %bayesian mean update:
                bbl = A(tr+1)/(A(tr+1)+B(tr+1));
                
                a_prio = A(tr);
                b_prio = B(tr);
                %bayesian surprise:
                bel =  log(beta(a_prio,b_prio))-(a_prio-1)*(psi(a_prio)-psi(a_prio+b_prio))-(b_prio-1)*(psi(b_prio)-psi(a_prio+b_prio));
                
                %final update:
                Q1(tr+1:end) = (1-ra2)*bbl + ra2*bel;
                
            else
                
                %model update:
                A1(tr+1:end) = A1(tr)+us(tr);
                B1(tr+1:end) = B1(tr)-us(tr)+1;
                
                %bayesian mean update:
                bbl = A1(tr+1)/(A1(tr+1)+B1(tr+1));
                
                a_prio = A1(tr);
                b_prio = B1(tr);
                %bayesian surprise:
                bel =  log(beta(a_prio,b_prio))-(a_prio-1)*(psi(a_prio)-psi(a_prio+b_prio))-(b_prio-1)*(psi(b_prio)-psi(a_prio+b_prio));
                
                %final update:
                Q2(tr+1:end) = (1-ra2)*bbl + ra2*bel;
                
            end
        end
        
        
        
    case 'VO0'
        trials = length(cs);
        init = beta_prior;
        A = init*ones(trials+1,1);
        B = init*ones(trials+1,1);
        A1 = init*ones(trials+1,1);
        B1 = init*ones(trials+1,1);
        Q1 = (A(1)/(A(1)+B(1)))*ones(trials,1);
        Q2 = (A1(1)/(A1(1)+B1(1)))*ones(trials,1);
        
        for tr = 1:trials
            
            
            if cs(tr) == 1
                
                %model update:
                A(tr+1:end) = A(tr)+us(tr);
                B(tr+1:end) = B(tr)-us(tr)+1;
                
                a_prio = A(tr);
                b_prio = B(tr);
                %bayesian surprise:
                
                bvol = 1-log(a_prio+b_prio);
                %final update:
                Q1(tr+1:end) = bvol;
                
            else
                
                %model update:
                A1(tr+1:end) = A1(tr)+us(tr);
                B1(tr+1:end) = B1(tr)-us(tr)+1;
                
                
                a_prio = A1(tr);
                b_prio = B1(tr);
                
                bvol = 1-log(a_prio+b_prio);
                %final update:
                Q2(tr+1:end) = bvol;
                
            end
        end
        
    case 'BH1'
        trials = length(cs);
        init = beta_prior;
        A = init*ones(trials+1,1);
        B = init*ones(trials+1,1);
        A1 = init*ones(trials+1,1);
        B1 = init*ones(trials+1,1);
        Q1 = (A(1)/(A(1)+B(1)))*ones(trials,1);
        Q2 = (A1(1)/(A1(1)+B1(1)))*ones(trials,1);
        
        for tr = 1:trials
            
            
            if cs(tr) == 1
                
                %model update:
                A(tr+1:end) = A(tr)+us(tr);
                B(tr+1:end) = B(tr)-us(tr)+1;
                
                %bayesian mean update:
                bbl = A(tr+1)/(A(tr+1)+B(tr+1));
                
                a_prio = A(tr);
                b_prio = B(tr);
                %bayesian surprise:
                bel =  log(beta(a_prio,b_prio))-(a_prio-1)*(psi(a_prio)-psi(a_prio+b_prio))-(b_prio-1)*(psi(b_prio)-psi(a_prio+b_prio));
                %                 varr = (A(tr+1)*B(tr+1))/((A(tr+1)+B(tr+1))^2*(A(tr+1) + B(tr+1) +1));
                bvol = 1-log(a_prio+b_prio);
                %final update:
                Q1(tr+1:end) = bvol*bbl + bel;
                
            else
                
                %model update:
                A1(tr+1:end) = A1(tr)+us(tr);
                B1(tr+1:end) = B1(tr)-us(tr)+1;
                
                %bayesian mean update:
                bbl = A1(tr+1)/(A1(tr+1)+B1(tr+1));
                
                a_prio = A1(tr);
                b_prio = B1(tr);
                %bayesian surprise:
                bel =  log(beta(a_prio,b_prio))-(a_prio-1)*(psi(a_prio)-psi(a_prio+b_prio))-(b_prio-1)*(psi(b_prio)-psi(a_prio+b_prio));
                %varr = (A1(tr+1)*B1(tr+1))/((A1(tr+1)+B1(tr+1))^2*(A1(tr+1) + B1(tr+1) +1));
                bvol = 1-log(a_prio+b_prio);
                %final update:
                Q2(tr+1:end) = bvol*bbl + bel;
                
            end
        end
        
        
    case 'KL0'
        trials = length(cs);
        init = beta_prior;
        A = init*ones(length(cs)+1,1);
        B = init*ones(length(cs)+1,1);
        A1 = init*ones(length(cs)+1,1);
        B1 = init*ones(length(cs)+1,1);
        Q1 = (A(1)/(A(1)+B(1)))*ones(trials,1);
        Q2 = (A1(1)/(A1(1)+B1(1)))*ones(trials,1);
        
        for tr = 1:trials
            
            if cs(tr) == 1
                
                A(tr+1:end) = A(tr)+us(tr);
                B(tr+1:end) = B(tr)-us(tr)+1;
                
                
                a_prio = A(tr);
                b_prio = B(tr);
                a_post = A(tr+1);
                b_post = B(tr+1);
                Q1(tr+1:end) = log(beta(a_prio, b_prio)/beta(a_post, b_post))-(a_prio-a_post)*psi(a_post)-(b_prio-b_post)*psi(b_post)+(a_prio-a_post+b_prio-b_post)*psi(a_post+b_post);
                
            else
                A1(tr+1:end) = A1(tr)+us(tr);
                B1(tr+1:end) = B1(tr)-us(tr)+1;
                
                a_prio = A1(tr);
                b_prio = B1(tr);
                a_post = A1(tr+1);
                b_post = B1(tr+1);
                Q2(tr+1:end) = log(beta(a_prio, b_prio)/beta(a_post, b_post))-(a_prio-a_post)*psi(a_post)-(b_prio-b_post)*psi(b_post)+(a_prio-a_post+b_prio-b_post)*psi(a_post+b_post);
                
            end
        end
        
        
    case 'BS0'
        trials = length(cs);
        init = beta_prior;
        A = init*ones(length(cs)+1,1);
        B = init*ones(length(cs)+1,1);
        A1 = init*ones(length(cs)+1,1);
        B1 = init*ones(length(cs)+1,1);
        Q1 = (A(1)/(A(1)+B(1)))*ones(trials,1);
        Q2 = (A1(1)/(A1(1)+B1(1)))*ones(trials,1);
        
        for tr = 1:trials
            
            if cs(tr) == 1
                
                A(tr+1:end) = A(tr)+us(tr);
                B(tr+1:end) = B(tr)-us(tr)+1;
                
                
                a_prio = A(tr);
                b_prio = B(tr);
                a_post = A(tr+1);
                b_post = B(tr+1);
                Q1(tr+1:end) =  log(beta(a_prio,b_prio))-(a_prio-1)*(psi(a_prio)-psi(a_prio+b_prio))-(b_prio-1)*(psi(b_prio)-psi(a_prio+b_prio));
                
            else
                A1(tr+1:end) = A1(tr)+us(tr);
                B1(tr+1:end) = B1(tr)-us(tr)+1;
                
                a_prio = A1(tr);
                b_prio = B1(tr);
                a_post = A1(tr+1);
                b_post = B1(tr+1);
                Q2(tr+1:end) =  log(beta(a_prio,b_prio))-(a_prio-1)*(psi(a_prio)-psi(a_prio+b_prio))-(b_prio-1)*(psi(b_prio)-psi(a_prio+b_prio));
                
            end
        end
        
        
    case 'TD1'
        init1 = 0.5;
        init2 = 0.5;
        %initialize values for CS+ and CS-:
        Q1 = init1*ones(size(us));
        Q2 = init1*ones(size(us));
        
        Qs1 = [ones(length(us),1) 0.5*ones(length(us),1) 0*ones(length(us),1)];
        Qs2 = [ones(length(us),1) 0.5*ones(length(us),1) 0*ones(length(us),1)];
        
        
        %update rule for every trial:
        for i = 1:length(us)-1
            if cs(i) == 1 %cs+, update cs+ representation:
                %for each trial we consider 3 steps, the CS onset and
                %US onset (or omission) and CS offset:
                %Qs = [Q1(i) Q1(i) Q1(i)];
                uss = [0 us(i) 0];
                for st = 1:length(uss)-1
                    Qs1(i+1:end,st) = Qs1(i,st) + x(end)*(uss(st) + Qs1(i,st+1) - Qs1(i,st));
                end
                Q1(i+1:end) = Qs1(i,1);
                
                Q2(i+1:end) = Q2(i);
            else %update cs- representation:
                uss = [0 us(i) 0];
                for st = 1:length(uss)-1
                    Qs2(i+1:end,st) = Qs2(i,st) + x(end)*(uss(st) + Qs2(i,st+1) - Qs2(i,st));
                    
                end
                Q2(i+1:end) = Qs2(i,1);
                
                Q1(i+1:end) = Q1(i);
            end
        end
        
    case 'HM1' %hybrid model between PH and RW
        init1 = 1;
        init2 = 1;
        %initialize values for CS+ and CS-:
        Q1 = 0.5*ones(size(us));
        Q2 = 0.5*ones(size(us));
        a1 = init1*ones(size(us));
        a2 = init2*ones(size(us));
        
        %update rule for every trial:
        for i = 1:length(us)-1
            if cs(i) == 1 %cs+, update cs+ representation:
                
                a1(i+1:end) = x(end)*abs(us(i)-Q1(i))+(1-x(end))*a1(i);
                Q1(i+1:end) = Q1(i)+x(end)*a1(i)*(us(i)-Q1(i));
                
            else %update cs- representation:
                a2(i+1:end) = x(end)*abs(us(i)-Q2(i))+(1-x(end))*a2(i);
                Q2(i+1:end) = Q2(i)+x(end)*a2(i)*(us(i)-Q2(i));
            end
        end
        
        Q1 = a1;
        Q2 = a2;
        
        
    case 'HM3' %hybrid model between HM1 and HM2:
        init1 = 1;
        init2 = 1;
        %initialize values for CS+ and CS-:
        Q1 = 0.5*ones(size(us));
        Q2 = 0.5*ones(size(us));
        a1 = init1*ones(size(us));
        a2 = init2*ones(size(us));
        
        %update rule for every trial:
        for i = 1:length(us)-1
            if cs(i) == 1 %cs+, update cs+ representation:
                
                a1(i+1:end) = x(end)*abs(us(i)-Q1(i))+(1-x(end))*a1(i);
                Q1(i+1:end) = Q1(i)+x(end)*a1(i)*(us(i)-Q1(i));
                
            else %update cs- representation:
                a2(i+1:end) = x(end)*abs(us(i)-Q2(i))+(1-x(end))*a2(i);
                Q2(i+1:end) = Q2(i)+x(end)*a2(i)*(us(i)-Q2(i));
            end
        end
        
        Q1 = a1+x(end-1)*Q1;
        Q2 = a2+x(end-1)*Q2;
        
        
    case 'HM2' %hybrid model between PH and RW
        init = 0.5;
        %initialize values for CS+ and CS-:
        Q1 = init*ones(size(us));
        Q2 = init*ones(size(us));
        a1 = 2*init*ones(size(us));
        a2 = 2*init*ones(size(us));
        
        %update rule for every trial:
        for i = 1:length(us)-1
            if cs(i) == 1 %cs+, update cs+ representation:
                
                a1(i+1:end) = x(end)*abs(us(i)-Q1(i))+(1-x(end))*a1(i);
                Q1(i+1:end) = Q1(i)+x(end)*a1(i)*(us(i)-Q1(i));
                
            else %update cs- representation:
                a2(i+1:end) = x(end)*abs(us(i)-Q2(i))+(1-x(end))*a2(i);
                Q2(i+1:end) = Q2(i)+x(end)*a2(i)*(us(i)-Q2(i));
            end
        end
        
        
    case 'NL0'
        Q1 = ones(length(cs),1);
        Q2 = zeros(length(cs),1);
        
end

switch out_f
    
    
    case 'li2' %linear function, no temporal component but with intersect
        
        Q1 = x(1)*Q1+x(2);
        Q2 = x(1)*Q2+x(2);
end

ydata1 = [Q1 Q2];
for i = 1:size(ydata1,1)
    ydata(i) = ydata1(i,cs(i));
end

xdata = xdata./max(xdata);

%compute sum of squared residuals (with or without US+ trials):
if ~incl_us
    
    ls = 0;
    for i = 1:length(ydata)
        if us(i) == 0
            ls = ls + (ydata(i)-xdata(i))^2;
        end
    end
    
else
    ls = 0;
    for i = 1:length(ydata)
        
        ls = ls + (ydata(i)-xdata(i))^2;
        
    end
    
end

