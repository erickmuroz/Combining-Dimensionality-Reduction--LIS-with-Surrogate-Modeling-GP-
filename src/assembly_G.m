function [G, contracts] = assembly_G(L, nele, EA, m, BC_dof, sensor_seed)
%% ASSEMBLY_G  Forward operator G = C*K^{-1} for the static structural bar.
%  Physics: clamped 1D bar in static equilibrium, K u = f, observation
%  model y = C u, so the composite map load -> observation is
%        G = C K^{-1},   built column-wise as G = (K \ C')'.
%  Also returns the clamped stiffness K and the consistent load operator
%  L_mat, because the prior pushes the load covariance through L_mat
%  (Gamma_pr = L_mat gamma_q L_mat').
%
%  This file only BUILDS the operator G (a one-off assembly step). All later
%  forward EVALUATIONS y = G*theta go through forward_model.m / apply_forward.
%
%  Adapted from the verified reference assembly_G.m (Jakob's LIP_Setup style).

nnode = nele + 1;
l     = L / nele;

%% stiffness matrix K (linear bar elements)
Ke = EA / l;
K  = sparse(nnode, nnode);
K  = K + sparse(1:nele,   1:nele,    Ke, nnode, nnode);
K  = K + sparse(1:nele,   2:nele+1, -Ke, nnode, nnode);
K  = K + sparse(2:nele+1, 1:nele,   -Ke, nnode, nnode);
K  = K + sparse(2:nele+1, 2:nele+1,  Ke, nnode, nnode);

%% load mapping operator L_mat (distributed load -> nodal forces)
L_mat = sparse(nnode, nele);
L_mat = L_mat + sparse(1:nele,   1:nele, l/2, nnode, nele);
L_mat = L_mat + sparse(2:nele+1, 1:nele, l/2, nnode, nele);

%% sensor placement (random interior nodes, reference strategy)
rng(sensor_seed);
m_pos = randi([1, nnode], 1, m);
m_pos = unique(m_pos);
m_pos = setdiff(m_pos, BC_dof);
while numel(m_pos) ~= m
    m_pos = [m_pos, randi([2, nele], 1, m - numel(m_pos))];
    m_pos = unique(m_pos);
    m_pos = setdiff(m_pos, BC_dof);
end
m_pos = sort(m_pos);

%% observation operator C (m x nnode)
C = zeros(m, nnode);
for i = 1:m
    C(i, m_pos(i)) = 1;
end

%% apply clamp: drop fixed DOF from K (row+col), C (col) and L_mat (row)
K(BC_dof, :) = []; K(:, BC_dof) = [];
C(:, BC_dof) = [];
L_mat(BC_dof, :) = [];

%% forward operator G = C K^{-1}, built as (K \ C')'
G = (K \ C')';
G = full(G);

%% sensor info and operators needed downstream
x_nodes        = (0:l:L)';
contracts.node  = m_pos(:);
contracts.z     = x_nodes(m_pos);
contracts.C     = C;
contracts.L_mat = full(L_mat);
contracts.K     = K;
contracts.l     = l;
contracts.x_nodes = x_nodes;
end
