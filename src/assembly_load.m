function [load_func, theta_true, z_grid, x_nodes] = assembly_load(L, nele, L_mat)
%% ASSEMBLY_LOAD  Ground-truth distributed load and the nodal force it induces.
%  Defines the truth the inverse problem must recover: a distributed axial
%  load q(z) on the clamped bar (constant baseline + Gaussian bump at mid-span).
%  The unknown parameter theta in this implementation is the NODAL FORCE
%  vector, so the ground truth is mapped through the consistent load operator:
%
%        theta_true = L_mat * q_true(midpoints).
%
%  This keeps theta_true in the same space as the prior (mu_f, Gamma_pr) and
%  as every forward evaluation y = G*theta.

nnode   = nele + 1;
l       = L / nele;
x_nodes = (0:l:L)';

% interior nodes carry the unknown DOFs (clamped node 1 excluded)
z_grid  = x_nodes(2:end);

% element midpoints, where the distributed load is sampled for L_mat
xq = (l/2 * (1:2:2*nele))';

% true distributed-load shape: baseline + Gaussian bump at mid-span
q0   = 4e6;        % baseline distributed load
bump = 2e6;        % bump amplitude
z_c  = L/2;        % bump centre
w    = L/8;        % bump width

q_mid  = q0 + bump * exp(-(xq - z_c).^2 / (2*w^2));   % load at midpoints
theta_true = L_mat * q_mid;                           % nodal force (n x 1)

% interpolant of the distributed load (for plotting only)
load_func = @(z) interp1(xq, q_mid, z, 'linear', q0);
end
