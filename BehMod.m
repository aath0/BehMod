
% Computational modeling tool
% Models physiological data, with Reinforcement Learning (RL), Bayesian (BL)
% and Null models.
% It requires Matlab optimisation toolbox and spm for the model comparison
% functions.

% First created on 02.12.2014 by Athina Tzovara, University of Zurich,
% Switzerland.
% Last updated and checked on 13.04.2017 by Athina Tzovara.


% 1. Estimate models or run just the comparison?
estimate_param = 1;

% 2. Which experiment?
exp = 'PubFe_SCR'; %label
filep = 'D:\Data\PubFe\';
filep_out = 'D:\Data\FR\Results\';
numtr = 160; % total number of trials

% 3. if we want to estimate the parameters from the scratch specify for
% which participants and which models and which optimisation options:
if estimate_param
    
    % List of participants:
    subj =  [1,2,3,5,7,8,9,10,11,13,14,16,17,18,19,20,21,22,23];
    
    % Model space:
    mm = ['RW1i';'PH1i';'HM1i';'HM2i';'HM3i';'TD1i';'BM0i';'KL0i';'BS0i';'BH1i';'BH2i';'VO0i';'NL0i'];
    
    % Optimisation options:
    %Include US+ trials when estimating models? (Recommended: 0)
    incl_us = 0; 
    
    optparam = optimset('fmincon');
    optparam.Algorithm = 'active-set';
    optparam.Algorithm = 'interior-point';
    optparam.MaxFunEvals = 1000000;
    optparam.MaxIter = 1000;
    optparam.Display = 'off';
    optparam.LargeScale = 'on';
    optparam.TolFun = 1e-30;
    optparam.TolX = 1e-30;
    optparam.DiffMinChange = 1e-30;
    %for fmincon
    A = 1;
    b = 1;
    Aeq = [];
    beq = [];
    lb = 0;
    ub = 1;
    nonlcon = [];
    
    %initialisation of a parameter matrix:
    param = zeros(6,length(mm),length(subj));
    
    %store the model estimates after optimisation here:
    xdatal = zeros(numtr,length(subj),size(mm,1));
    ydatal = zeros(numtr,length(subj),size(mm,1));
        
    for kk = 1:size(mm,1)
        
        model = mm(kk,1:end-1)
        
        out_f = ['l',mm(kk,end),'2'];
        
        for ss = 1:length(subj)
            
            clear indata x Data Header ls_m post resp total_tr output us cs xdata ydata block fval f
            s_id = num2str(subj(ss));
            
            
            %%%% This part here is experiment-specific. We need to load
            %%%% the physiological data and conditions for each subject.
            %%%% The physiological data are stored in "filen" file, which contains
            %%%% a 'xdata' vector with 1 estimate per trial:
            
            filep2 = [filep,'S', s_id, '\'];
            load([filep2, 'PubFe_',s_id,'_Session1.mat'])
            filen = ['S',s_id, '.prep_data_dcm0_a.mat'];
            indata = [PubFe{1, 1}.indata; PubFe{1, 2}.indata]; %trial information for both blocks.
            clear PubFe
            load([filep2, filen])
            
            % extract trial types:
            us = indata(:,3); % 1/0 for US+/-
            cs = indata(:,2); %1/2 for CS+/-s
            total_tr = length(us);
            
            %set the initial parameters:
            switch out_f %mapping function:
                
                case {'li2'}
                    x0(1) = 1; %slope cs+/-
                    x0(2) = 0; %intersect cs+/-
                    
                    %range of values:
                    low_lim = [-100 -100];
                    upp_lim = [100 100];
                    
            end
            
            %additional parameters per model, if there are any:
            
            switch model
                
                case 'RW1'
                    x0 = [x0 1];
                    low_lim = [low_lim 0];
                    upp_lim = [upp_lim 1];
                    
                case 'PH1'
                    
                    x0 = [x0 0.001];
                    low_lim = [low_lim 0];
                    upp_lim = [upp_lim 1];
                    
                case 'HM1'
                    
                    x0 = [x0 0.5];
                    low_lim = [low_lim 0];
                    upp_lim = [upp_lim 1];
                    
                case 'HM2'
                    
                    x0 = [x0 0.5];
                    low_lim = [low_lim 0];
                    upp_lim = [upp_lim 1];
                    
                case 'HM3'
                    
                    x0 = [x0 0.5 0.5];
                    low_lim = [low_lim 0 0];
                    upp_lim = [upp_lim 1 1];
                    
                case 'BH2'
                    x0 = [x0 1];
                    low_lim = [low_lim 0];
                    upp_lim = [upp_lim 1];
                    
            end
            
            % common part to all models:
            f = @(x)model_inter(x,cs,us,xdata,model,out_f,incl_us);
            [x,fval,exitflag,output] = fmincon(f,x0,[],[],Aeq,beq,[low_lim],[upp_lim],nonlcon,optparam);
            %optimal model:
            [ls_RW,ydata,xdata,Q1,Q2] = model_inter(x,cs,us,xdata,model,out_f,incl_us);
            param(1:length(x),kk,ss) = x;
            likel(kk,ss) = ls_RW/length(find(us==0));
            
            AIC(kk,ss) = AIC_f(ls_RW,length(x0),length(find(us == 0)));
            BIC(kk,ss) = BIC_f(ls_RW,length(x0),length(find(us == 0)));
            
            %exclude US+ trials before computing Explained Variance:
            xd = xdata(find(us == 0));
            yd = ydata(find(us == 0));
            EV(kk,ss) = exvar(xd,yd);
            clear x x0 ls_RW
            
            clear yd xd Data Header indata ydata xdata low_lim upp_lim Q1 Q2 cs us tr2keep
            
        end
    end
    
    save([filep_out, exp, '_Estimates2.mat'], 'xdatal', 'ydatal','likel', 'param', 'mm','subj','AIC','BIC','EV')
    
else
    load([filep_out, exp, '_Estimates.mat'])
end


% RFX comparison:
[alpha,exp_r,xp,pxp_BIC,bor] = spm_BMS(-BIC')

ec = [0.5 0.5 0.5];
ec2 = [0.3 0.3 0.3];

ff = figure;
set(ff,'Position', [100 100 500 250])
hold on,
pp = 1;
xt = [];
for kk = 1:size(BIC,1)
    bar(pp, pxp_BIC(kk),'FaceColor', ec, 'EdgeColor', ec2, 'LineWidth',2)
    pp = pp+1.5;
    
end

ylim([0 1.05])
xlim([0 pp-1])
set(gca,'YTick',[0:0.5:1])
set(gca,'XTick',xt)
set(gca,'XTickLabel','')
set(gca,'YTickLabel','')


% FFX comparison:

%FFX analysis:
sum_BIC = sum(BIC,2)';

[~,ref_model_BIC] = min(sum_BIC);
sum_BIC = sum_BIC-sum_BIC(ref_model_BIC);

ff = figure,
set(ff,'Position', [100 100 500 250])
hold on,
pp = 1;
xt = [];
for kk = 1:size(BIC,1)
    bar(pp, sum_BIC(kk),'FaceColor', ec, 'EdgeColor', ec2, 'LineWidth',2)
    pp = pp+1.5;
    
    
end
fsize = 15;
yl = round(max([sum_AIC sum_BIC])+10);
ylim([0 yl])
xlim([0 pp-1])
set(gca,'YTick',[0:100:yl])
set(gca,'XTick',xt)
set(gca,'FontWeight', 'Bold', 'FontSize', fsize)


