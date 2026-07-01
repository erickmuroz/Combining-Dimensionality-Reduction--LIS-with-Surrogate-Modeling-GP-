function [mu, var, lml, alpha, L] = gp_predict(Xtr, ytr, Xte, kfun, sn2)
n = size(Xtr, 1);
K = kfun(Xtr, Xtr);
jitter = 1e-10;
L = chol(K + (sn2 + jitter)*eye(n), 'lower');

alpha = L' \ (L \ ytr);

Ks  = kfun(Xtr, Xte);            % (n x t)
mu  = Ks' * alpha;              % (t x 1)

v    = L \ Ks;                   % (n x t)
kss0 = kfun(Xte(1,:), Xte(1,:));% scalar k(x,x) = sf2 (stationary)
var  = kss0 - sum(v.^2, 1)';    % (t x 1)
var  = max(var, 0);

lml = -0.5*(ytr'*alpha) - sum(log(diag(L))) - 0.5*n*log(2*pi);
end
