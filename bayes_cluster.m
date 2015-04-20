function [logprob, hier] = bayes_cluster(data,opt)
% Hierarchical clustering based on Bayesian model comparison
% SYNTAX: [ LOGPROB , HIER]= BAYES_CLUSTER ( DATA , [OPT] )
%
% DATA array ( n x p ) n observations drawn from p variables.
% OPT.FLAG_CORR (boolean, default false) 
%   false: use covariance.
%   true: use correlation. 
% OPT.NU0 (scalar, default p+1) the NU0 parameter for the a priori distribution.
% OPT.LAMBDA0 (array p x p, default identity matrix) 
%   The LAMBDA0 parameter of the a priori distribution. 
%
% LOGPROB (vector) the log probability of the model for the different levels 
%   of the hierarchy
% HIER (2D array) defining a hierarchy :
%   Column 1: Entity no x
%   Column 2: joining entity no y
%   Column 3: Level of new link
%
% EXAMPLE:
% data = [repmat(randn(100,1),[1 10])+randn(100,10) ...
%         repmat(randn(100,1),[1 10])+randn(100,10) ...
%         repmat(randn(100,1),[1 10])+randn(100,10) ];
% [logprob,hier] = bayes_cluster(data);
% 
% NOTE: the clustering model is based on an assumption of a multivariate Gaussian
% distribution with independent and identically distributed samples, and a zero 
% covariance between variables belonging to different clusters.
% NOTE 2: this code implements the 'BayesCov' and 'BayesCorr' methods (depending
%   on OPT.FLAG_CORR) described in the following paper http://arxiv.org/abs/1501.05194
% NOTE 3: the output HIER is formatted like the output Z of the function linkage. 
%   Use DENDROGAM to visualize the hierarchy, and CLUSTER to extract a partition.
%   All these functions are part of the statistics toolbox.
%
% Copyright (c) Guillaume Marrelec, Laboratoire d'imagerie fonctionnelle,
% Inserm, France, 2015.
% Maintainer: marrelec@imed.jussieu.fr
% See licensing information in the code.
% Keywords: clustering, Bayesan model, Gaussian

% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.

%% Setting up defaults

if ~exist('opt','var')
    opt = struct();
end

% flag to switch between covariance and correlation
if ~isfield(opt,'flag_corr')
    opt.flag_corr = false;
end

% parameter nu0 of the a priori distribution
if ~isfield(opt,'nu0')
    opt.nu0 = [];
end

if isempty(opt.nu0)
    opt.nu0 = size(data,2)+1;
end

% parameter lambda0 of the a priori distribution
if ~isfield(opt,'lambda0')
    opt.lambda0 = [];
end

if isempty(opt.lambda0)
    opt.lambda0 = eye(size(data,2));
end

%% Generation of the covariance (or correlation) matrix
[nb_ech,nb_var] = size(data);
if opt.flag_corr
    SS = (nb_ech-1)*corrcoef(data);
else
    SS = (nb_ech-1)*cov(data);
end

%% Hierarchical agglomerative clustering
nb_iter_max = nb_var-1;
list_var = 1:nb_var;
hier = zeros(nb_iter_max,4);

%% Initial values
part = 1:nb_var; % the partition, into 

% The sum of diagonal log-likelihood
logp = zeros(1,nb_var);
for num_v = 1:nb_var
    logp(num_v) = sub_loglikelihood(squeeze(SS(num_v,num_v)),nb_ech,opt.nu0-(nb_var-1),opt.lambda0(num_v,num_v));
end
logprob = zeros(1,nb_iter_max+1);
logprob(1) = sum(logp);

% The similarity matrix
logq = -Inf*ones(nb_var,nb_var);
for num_v1 = 1:nb_var % Loop over variables
    connex_v1 = num_v1+1:nb_var;
    for num_c = 1:length(connex_v1)
        num_v2 = connex_v1(num_c);
        % Extract submatrix
        gr = [num_v1 num_v2];
        nb_elem = length(gr);
        SSgr = SS(gr,gr);
        % Compute differences between log-likelihoods
        logq(num_v1,num_v2) = sub_loglikelihood(SSgr,nb_ech,opt.nu0-(nb_var-nb_elem),opt.lambda0(gr,gr))-logp(num_v1)-logp(num_v2);
    end
end


%% Agglomeration

% Turn logq into a symmetric matrix
logq = triu(logq)+triu(logq,1)';

% Find maximal pairings
[max_logq,list_varmax] = max(logq,[],1);

% Now merge most similar clusters, and iterate
for num_i = 1:nb_iter_max
    
    % Find max similarity
    [logqmax,gr1max] = max(max_logq);
    
    % If multiple pairs achieve maximal similarity, 
    % select a random pair 
    if length(gr1max) > 1
        gr1max = gr1max(d_unidrnd(length(gr1max)));
    end
    
    % Find the indices of the variables for merging
    gr2max = list_varmax(gr1max);
    tmp = [gr1max gr2max];
    gr1max = min(tmp);
    gr2max = max(tmp);
    
    % Document the merge in the HIER array
    hier(num_i,:) = [logqmax list_var(gr1max) list_var(gr2max) max(list_var)+1];
    
    % Update the partition
    part((part == list_var(gr1max)) | (part == list_var(gr2max))) = max(list_var)+1;
    
    % Update the list of variables
    list_var(gr1max) = max(list_var)+1;
    list_var(gr2max) = NaN;
    
    % Update logp
    logp(gr1max) = logp(gr1max)+logp(gr2max)+logq(gr1max,gr2max);
    logp(gr2max) = NaN;
    
    % Update the similarity matrix
    logq(gr1max,:) = -Inf;
    gr1 = (part == max(part));
    part_uniq = unique(part);
    for num_g2 = part_uniq(1:end-1)
        gr2 = (part == num_g2);
        gr = (gr1 | gr2);
        nb_elem = sum(gr);
        SSgr = SS(gr,gr,:);
            
        % difference of log probability
        logq(gr1max,list_var == num_g2) = sub_loglikelihood(SSgr,nb_ech,opt.nu0-(nb_var-nb_elem),opt.lambda0(gr,gr))-logp(gr1max)-logp(list_var == num_g2);
    end
    logq(:,gr1max) = logq(gr1max,:)';
    logq(:,gr2max) = -Inf;
    logq(gr2max,:) = -Inf;
    
    % Update the vector of maximal log probability
    % This step is implemented to avoid running a full max on the similarity matrix at each iteration
    [max_logq(list_varmax == gr2max),list_varmax(list_varmax == gr2max)] = max(logq(:,list_varmax == gr2max),[],1);
    [max_logq(list_varmax == gr1max),list_varmax(list_varmax == gr1max)] = max(logq(:,list_varmax == gr1max),[],1);
    max_logq(gr2max) = -Inf;
    list_varmax(gr2max) = NaN;
    
    [max_logq,ind_x] = max([max_logq ; logq(gr1max,:)],[],1);
    list_varmax(ind_x == 2) = gr1max;
    [max_logq(gr1max),list_varmax(gr1max)] = max(logq(gr1max,:));
    
    logprob(num_i+1) = sum(logp(~isnan(logp)));
    
end

%% Arrange hier according to matlab's conventions
hier = hier(:,[2 3 1]);

%% Marginal log-likelihood function
function llh = sub_loglikelihood(SS,nb_ech,nu0,lambda0)

nb_var = size(SS,1);
vpL0 = eig(lambda0);
nu = nu0+nb_ech-1;
vpL = eig(lambda0+SS);

lnh = 0.5*(nb_ech-1)*nb_var*log(2)+sum(gammaln((nu+1-(1:nb_var))/2))-sum(gammaln((nu0+1-(1:nb_var))/2))+nu0/2*sum(log(vpL0))-nu/2*sum(log(vpL));
llh = lnh/log(10);
