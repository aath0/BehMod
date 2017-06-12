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



