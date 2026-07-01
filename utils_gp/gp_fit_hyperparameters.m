function [ell, sf2, sn2, nlml] = gp_fit_hyperparameters(Xtr, ytr, ~)
%% GP_FIT_HYPERPARAMETERS  Evidence-optimal hyperparameters by NLML minimization.
%  Maximizes the log marginal likelihood (R&W 2006, Section 5.4.1) by
%  minimizing gp_nlml over log[ell, sf2, sn2] with fminsearch (Nelder-Mead,
%  base MATLAB/Octave - no Optimization Toolbox dependency).
%
%  Inputs:
%    Xtr (n x d), ytr (n x 1), third arg ('se') kept for call-site symmetry.
%  Outputs:
%    ell, sf2, sn2  fitted hyperparameters (positive), nlml at optimum.

% --- data-driven initial guesses ---
n = size(Xtr, 1);
% median pairwise distance for the length scale
if n > 1
    D2 = sum(Xtr.^2,2) + sum(Xtr.^2,2)' - 2*(Xtr*Xtr');
    D2 = max(D2, 0);
    dvec = sqrt(D2(triu(true(n), 1)));
    ell0 = median(dvec(dvec > 0));
    if isempty(ell0) || ~isfinite(ell0) || ell0 <= 0
        ell0 = 1;
    end
else
    ell0 = 1;
end
vy   = var(ytr);
if vy <= 0 || ~isfinite(vy), vy = 1; end
sf20 = vy;
sn20 = 0.1 * vy;

logp0 = log([ell0, sf20, sn20]);

obj  = @(lp) gp_nlml(lp, Xtr, ytr);
opts = optimset('Display','off', 'MaxFunEvals', 2000, 'MaxIter', 2000, ...
                'TolFun', 1e-6, 'TolX', 1e-6);
[logp_opt, nlml] = fminsearch(obj, logp0, opts);

ell = exp(logp_opt(1));
sf2 = exp(logp_opt(2));
sn2 = exp(logp_opt(3));
end
