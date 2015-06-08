# arXiv-1501.05194
Companion code for manuscript http://arxiv.org/abs/1501.05194 . To use, just download the function `bayes_cluster.m`, add it to your matlab or octave path, and it should work. This code implements the 'BayesCov' and 'BayesCorr' methods (depending on`opt.flag_corr`). 
```matlab
data = [repmat(randn(100,1),[1 10])+randn(100,10) ...
        repmat(randn(100,1),[1 10])+randn(100,10) ...
       repmat(randn(100,1),[1 10])+randn(100,10) ];
[Z,logprob] = bayes_cluster(data);
```
To use the BIC approximation of the log-likelihood:
```matlab
opt_c.flag_bic = true;
data = [repmat(randn(100,1),[1 10])+randn(100,10) ...
        repmat(randn(100,1),[1 10])+randn(100,10) ...
       repmat(randn(100,1),[1 10])+randn(100,10) ];
[Z,logprob] = bayes_cluster(data,opt_c);
```
