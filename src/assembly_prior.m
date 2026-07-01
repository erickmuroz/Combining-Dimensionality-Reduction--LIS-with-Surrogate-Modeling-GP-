function [Gamma_pr, S_pr, mu_f, gamma_q] = assembly_prior(nele, L_mat, l, mu_q, sigma_q, ell)
%% ASSEMBLY_PRIOR  Singular nodal-force prior from an OU load covariance.
%  The exponential (Ornstein-Uhlenbeck) kernel is placed on the DISTRIBUTED
%  load q at the element midpoints, giving gamma_q. The nodal-force prior is
%  obtained by pushing that covariance through the consistent load operator:
%
%        Gamma_pr = L_mat * gamma_q * L_mat'.
%
%  Because L_mat maps a higher-dimensional load field to nodal forces, in the
%  general case Gamma_pr is rank deficient. We therefore keep the square-root
%  factor
%        S_pr = L_mat * chol(gamma_q,'lower'),
%  and NEVER invert Gamma_pr directly; LIS basis and posterior use S_pr and
%  the Woodbury form. (Spantini et al. 2015, prior-preconditioned setup.)
%
%  Parameterized (mu_q, sigma_q, ell passed in) so Part A can tune ell/sigma_q
%  while searching for r* = 1.

% element midpoints (quadrature points for the OU kernel)
XQ = l/2 * (1:2:2*nele);
xq = XQ(:);

% exponential kernel on the distributed load q
gamma_q = zeros(nele, nele);
for i = 1:nele
    for j = 1:nele
        gamma_q(i,j) = sigma_q^2 * exp(-abs(xq(i) - xq(j)) / ell);
    end
end

% map to nodal-force covariance and its square-root factor
Gamma_pr = L_mat * gamma_q * L_mat';
S_pr     = L_mat * chol(gamma_q, 'lower');

% prior mean nodal force
mu_f = L_mat * (mu_q .* ones(nele, 1));
end
