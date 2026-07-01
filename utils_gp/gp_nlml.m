function nlml = gp_nlml(logparams, Xtr, ytr)
%% GP_NLML  Negative log marginal likelihood as a function of log-hyperparams.
%  Objective for hyperparameter optimization. Parameters are optimized in log
%  space so they stay positive: logparams = log([ell, sf2, sn2]).
%  Returns -lml from gp_predict (R&W eq. 2.30 with sign flipped). A try/catch
%  guards against non-PD kernels during the search, returning a large penalty.

ell = exp(logparams(1));
sf2 = exp(logparams(2));
sn2 = exp(logparams(3));

try
    kfun = @(XA, XB) kernel_se(XA, XB, ell, sf2);
    [~, ~, lml] = gp_predict(Xtr, ytr, Xtr(1,:), kfun, sn2);
    nlml = -lml;
    if ~isfinite(nlml)
        nlml = 1e10;
    end
catch
    nlml = 1e10;
end
end
