# arXiv-1501.05194
Companion code for manuscript http://arxiv.org/abs/1501.05194 . To use, just download the function `bayes_cluster.m`, add it to your matlab or octave path, and it should work. This code implements the 'BayesCov' and 'BayesCorr' methods (depending on`opt.flag_corr`). 
```matlab
data = [repmat(randn(100,1),[1 10])+randn(100,10) ...
   repmat(randn(100,1),[1 10])+randn(100,10) ...
   repmat(randn(100,1),[1 10])+randn(100,10) ];
mat = corr(data);
[Z,logprob] = bayes_cluster(mat,100);
```
To use the BIC approximation of the log-likelihood:
```matlab
data = [repmat(randn(100,1),[1 10])+randn(100,10) ...
   repmat(randn(100,1),[1 10])+randn(100,10) ...
   repmat(randn(100,1),[1 10])+randn(100,10) ];
mat = corr(data);
flag.bic = true;
[Z,logprob] = bayes_cluster(mat,100,flag);
```
To use a covariance matrix:
```matlab
data = [repmat(randn(100,1),[1 10])+randn(100,10) ...
   repmat(randn(100,1),[1 10])+randn(100,10) ...
   repmat(randn(100,1),[1 10])+randn(100,10) ];
mat = cov(data);
flag.corr = false;
[Z,logprob] = bayes_cluster(mat,100,flag);
```

