function S_theta = gen_samples(N, mu_f, S_pr)
%% GEN_SAMPLES  Draw N nodal-force samples from the prior theta ~ N(mu_f, Gamma_pr).
%  Using the square-root factor S_pr (Gamma_pr = S_pr*S_pr'),
%
%        theta_i = mu_f + S_pr * xi_i,    xi_i ~ N(0, I).
%
%  Sampling through S_pr avoids ever forming or factorizing the (possibly
%  singular) Gamma_pr. Samples live in nodal-force space, the same space as
%  theta_true and every forward evaluation y = G*theta.
%
%  Output:
%    S_theta   (n x N)  prior samples of the nodal-force parameter

n  = size(S_pr, 1);
k  = size(S_pr, 2);
xi = randn(k, N);
S_theta = mu_f + S_pr * xi;
end
