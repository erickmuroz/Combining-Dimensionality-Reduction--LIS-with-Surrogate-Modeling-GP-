%% ========================================================================
%  main.m  -  LIS on the 1D bar + GP surrogate replacing G (K^-1)
%
%  src    : assembly_G, assembly_prior, assembly_load    
%  utils_gp: kernel_se, gp_predict, gp_fit_hyperparameters
% =========================================================================
 
clear; clc; close all;
addpath('./src');
addpath('./utils_gp');
 
%%  PARAMETERS 
L = 2;
nele = 100;
EA = 1e7;
BC_dof = 1; 
m = 8;
sensor_seed = 41;
mu_q = 4e6;
sigma_q = 0.3*mu_q;
ell_pr = 0.5;
gamma_obs_var = 3e-2;         
l = L / nele;
 
 
%% ========================================================================
%  PART A - LIS
%  ========================================================================
 
%% 1. forward operator G and prior operators
[G, con] = assembly_G(L, nele, EA, m, BC_dof, sensor_seed);  % con = bundle of everything the FEM assembly computed besides G
L_mat = con.L_mat;
[Gamma_pr, S_pr, mu_f, gamma_q] = assembly_prior(nele, L_mat, l, mu_q, sigma_q, ell_pr);
 
%% 2. ground-truth load and noisy data  (y = G theta_true + noise)
[~, theta_true, z_grid, x_nodes] = assembly_load(L, nele, L_mat);
n = numel(theta_true);
noise_std = sqrt(gamma_obs_var);
Gamma_obs = gamma_obs_var * eye(m);
rng(41);
y = G*theta_true + noise_std*randn(m,1);
 
%% 3. exact posterior covariance + mean  (WOODBURY form, singular prior)
tmp       = G*Gamma_pr*G' + Gamma_obs;
Gamma_pos = Gamma_pr - Gamma_pr*G'*(tmp\G)*Gamma_pr;
mu_pos    = mu_f + Gamma_pr*G'*(tmp\(y - G*mu_f));
 
%% 4. LIS basis  (square-root SVD route, Jakob's section 5)
R = G' .* (1 ./ sqrt(diag(Gamma_obs)))';%remark 4: R is the square root factor of H, so that we can get the eigen decomp of (H, Prior precision) 
[U, Delta, Z] = svd(R' * S_pr); %svd via R^T S_pr
delta = diag(Delta);
 
%generalized eigenvalue of R' * S_pr are equal to the square roots of
%generalized eigenvalues of the pencil (H, Prior precision) 
 
tol   = max(size(R'*S_pr)) * eps(max(delta));   %numerical rank threshold
r_eff = sum(delta > tol); %keep only the numerically meaningful
U     = U(:, 1:r_eff);
delta = delta(1:r_eff);
Z     = Z(:, 1:r_eff);
 
V = S_pr * Z; %from LIS to full parameter vector via V_r * w 
W = R * U .* (1 ./ delta)'; %the other way around
r_plot = max(sum(delta.^2 > 1), 1);   
 
%% 5. OLR posterior at each rank  (reduced operator)
r_max          = r_eff;
Gamma_OLR      = cell(r_max,1);
forstner_rank  = zeros(r_max,1);
for r = 1:r_max
    Pr = V(:,1:r) * W(:,1:r)';
    Gr = G * Pr;
    tr = Gr*Gamma_pr*Gr' + Gamma_obs;
    Gamma_OLR{r}     = Gamma_pr - Gamma_pr*Gr'*(tr\Gr)*Gamma_pr;
    forstner_rank(r) = foerstner_distance(Gamma_OLR{r}, Gamma_pos, W);
end
 
 
%% ========================================================================
%  PART B - simplest GP in the LIS coordinate
%  GP swallows K^{-1}
%  ========================================================================
 
r  = 1; %fix rank to 1 
V1 = V(:,1); %extract only needed columns
W1 = W(:,1);
K  = con.K; 
 
%% training data: f_r sampled from its own prior, N(0,1)
% (W1' * Gamma_pr * W1 = 1, verified earlier -> this IS the exact prior on f_r)
n_train = 8; %fix the budget
rng(7); 
Fr_tr = randn(n_train, 1); %samples ntrain candidates of fr from its "exact" prior 
 
Ur_tr = zeros(n_train, 1);
for i = 1:n_train
    f_full   = V1 * Fr_tr(i); % lift: f_r -> (via V)
    u_full   = K \ f_full; % solve: f -> u (via K^{-1})  <- the real expensive FEM Solve
    Ur_tr(i) = W1' * u_full; % project: u -> u_r(via W') 
end
% get the U_r(fr) - > Ur_tr(i) = W1' * K^{-1} * V1 * Fr_tr(i)
 
%% GP fit -- identical call structure to basic one
%hyperparameters hand fixed, not the best ones yet, nothing adaptive
ell = 0.8;
sf2 = var(Ur_tr); %signal variance, variance of the n_train values of Ur_tr in the previous loop
sn2 = 1e-6 * sf2;   % noise variance relative to signal variance, not fixed. "how much noise there is in the observations"
%such that noise variance is not bigger than the signal itself and the GP
%does not ignore the data

kfun = @(A,B) kernel_se(A, B, ell, sf2); %kernel_Se with the hyperparameters inside 
 
Fr_te = linspace(-3, 3, 200)'; 
[mu_ur, var_ur] = gp_predict(Fr_tr, Ur_tr, Fr_te, kfun, sn2); %calls for predictive equations
sd_ur = sqrt(var_ur); %predictive variance of the GP from the gp_predict
 
%% truth on the test grid, for comparison only (cheap here since r=1, n small)
% exact curve on the 200 test points 
Ur_te_true = zeros(size(Fr_te));
for i = 1:numel(Fr_te)
    Ur_te_true(i) = W1' * (K \ (V1 * Fr_te(i)));
end
 
 
%% ========================================================================
%  PART C - GP-based posterior via plug-in Woodbury
%  ========================================================================
 
C = con.C;
 
n_train_list  = [2 3 5 8 15 30 60]; % training budgets to sweep
Fr_slope_grid = linspace(-0.8, 0.8, 21)'; % where the GP mean is read to fit the slope
%21 points for smaller range to calculate the c value, not the whole graph.
%sampling from ftr with randn (nt,1) makes that most of the samples (its a normal Gaussian) lie
%on that part of the grid. 
 
Gamma_GP    = cell(numel(n_train_list),1);
forstner_gp = zeros(numel(n_train_list),1);
 
for ti = 1:numel(n_train_list)
    nt = n_train_list(ti);
 
    %loop like in B but for the sweep over all the nt, amount of training points
    % training data: nt real FEM solves (the budget)
    rng(100+ti);
    Fr_tr_i = randn(nt,1);
    Ur_tr_i = zeros(nt,1);
    for i = 1:nt
        Ur_tr_i(i) = W1' * (K \ (V1*Fr_tr_i(i)));   % lift-solve-project, one solve
    end
 
    %  fit GP on (f_r, u_r) pairs 
    sf2_i  = max(var(Ur_tr_i), eps);
    sn2_i  = 1e-6 * sf2_i;
    kfun_i = @(A,B) kernel_se(A, B, ell, sf2_i);


 
    % GP mean over a grid -> slope c_hat  (u_r is exactly linear, so the
    % surrogate collapses to a single slope; c_hat -> W1'*K^{-1}*V1 as nt grows)
    mu_slope = gp_predict(Fr_tr_i, Ur_tr_i, Fr_slope_grid, kfun_i, sn2_i); %asks for the predictive mean on the 21 points
    p     = polyfit(Fr_slope_grid, mu_slope, 1); %matlab adjustment for coefficients of the line 
    c_hat = p(1); %just p1 since the problem goes to the origin
 
    % reconstruct the surrogate forward operator and its posterior
    Ghat = C * V1 * c_hat * W1';     % 8 x 100, rank-1 GP stand-in for G*P1
 

    %Calculate the approximation
    tr_i = Ghat*Gamma_pr*Ghat' + Gamma_obs;
    Gamma_GP{ti}    = Gamma_pr - Gamma_pr*Ghat'*(tr_i\Ghat)*Gamma_pr;
    forstner_gp(ti) = foerstner_distance(Gamma_GP{ti}, Gamma_pos, W);
end
 
 
%% ========================================================================
%  PLOTS 
%  ========================================================================
set(0,'defaulttextinterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
FS = 13;  FSt = 15;
col_exact = [0.15 0.28 0.62];      
col_olr   = [0.85 0.45 0.10];
col_thr   = [0.75 0.10 0.10];     
style_ax  = @(ax) set(ax,'FontSize',FS,'Box','off','LineWidth',0.9,'TickDir','out','XGrid','on','YGrid','on', 'GridAlpha',0.12,'Layer','top');
 
fig = figure('Color','w','Position',[60 60 1180 820], 'Name', 'LIS + GP results');
tg  = uitabgroup(fig);
 
%% plot 1: LIS eigenvalue spectrum
axes('Parent', uitab(tg, 'Title', 'P1 Spectrum'));
h = semilogy(1:r_max, delta.^2, 'o-', 'Color', col_exact, 'MarkerFaceColor', col_exact, 'MarkerSize', 7, 'LineWidth', 1.8);
hold on;
yline(1, '--', 'Color', col_thr, 'LineWidth', 1.6);
yl = ylim;
patch([0.5 r_plot+0.5 r_plot+0.5 0.5], [yl(1) yl(1) yl(2) yl(2)], col_exact, 'FaceAlpha', 0.06, 'EdgeColor', 'none');
uistack(h,'top');  ylim(yl);
text(r_plot+0.35, 1.6, '$\delta_i^2=1$', 'Color', col_thr, 'FontSize', 11);
text((r_plot+0.5+r_max)/2, yl(2)*0.4, sprintf('$r^*=%d$',r_plot), 'Color', col_exact, 'FontSize', 12, 'HorizontalAlignment','center');
style_ax(gca);
xlabel('LIS direction $i$'); ylabel('$\delta_i^2$');
title('LIS eigenvalue spectrum', 'FontSize', FSt);
xlim([0.5 r_max+0.5]);
 
%% plot 2: Forstner error of the OLR posterior vs rank
axes('Parent', uitab(tg, 'Title', 'P2 OLR error'));
semilogy(1:r_max, forstner_rank, 'o-', 'Color', col_olr, 'MarkerFaceColor', col_olr, 'MarkerSize', 7, 'LineWidth', 1.8);
hold on; xline(r_plot, '--', 'Color', col_thr, 'LineWidth', 1.5);
text(r_plot+0.15, max(forstner_rank)*0.4, sprintf('$r^*=%d$',r_plot), 'Color', col_thr, 'FontSize', 12);
style_ax(gca);
xlabel('Rank $r$'); ylabel('F\"orstner distance to exact');
title('OLR posterior accuracy vs rank', 'FontSize', FSt);
xlim([0.5 r_max+0.5]);
 
%% plot 3: GP on u_r(f_r)
axes('Parent', uitab(tg, 'Title', 'P3 GP fit'));
hold on; box on;
fill([Fr_te; flipud(Fr_te)], [mu_ur+2*sd_ur; flipud(mu_ur-2*sd_ur)], [0.85 0.90 0.98], 'EdgeColor','none');
plot(Fr_te, Ur_te_true, 'k--', 'LineWidth', 1.3);
plot(Fr_te, mu_ur,      'b-',  'LineWidth', 1.6);
plot(Fr_tr, Ur_tr,      'ko',  'MarkerFaceColor','k', 'MarkerSize', 5);
style_ax(gca);
xlabel('$f_r$'); ylabel('$u_r(f_r)$');
legend({'95\% band','truth','GP mean','training pts'}, 'Location','northwest');
title(sprintf('GP on $u_r(f_r)$, $r=1$, $n_{tr}=%d$', n_train), 'FontSize', FSt);
 
%% plot 4: Forstner error of the GP posterior vs training budget
axes('Parent', uitab(tg, 'Title', 'P4 GP vs n_{tr}'));
semilogy(n_train_list, forstner_gp, 'o-', 'Color', col_exact, 'MarkerFaceColor', col_exact, 'MarkerSize', 7, 'LineWidth', 1.8);
hold on;
yline(forstner_rank(1), '--', 'Color', col_olr, 'LineWidth', 1.6);
text(n_train_list(end), forstner_rank(1)*1.2, 'OLR floor ($r=1$)', 'Color', col_olr, 'FontSize', 11, 'HorizontalAlignment','right');
style_ax(gca);
xlabel('$n_{tr}$'); ylabel('F\"orstner distance to exact');
title('GP surrogate posterior error vs training budget', 'FontSize', FSt);
xlim([n_train_list(1) n_train_list(end)]);
 
%% plot 5: covariance comparison  Prior | Exact | OLR(1) | GP (best budget)
tab5 = uitab(tg, 'Title', 'P5 Covariances');
tl5  = tiledlayout(tab5, 1, 4, 'Padding','compact', 'TileSpacing','compact');
c_max = max(Gamma_pr(:));
mats  = {Gamma_pr, Gamma_pos, Gamma_OLR{1}, Gamma_GP{end}};
ttls  = {'Prior $\Gamma_{pr}$', 'Exact $\Gamma_{pos}$', 'OLR $r=1$', sprintf('GP ($n_{tr}=%d$)', n_train_list(end))};
for kk = 1:4
    ax = nexttile(tl5);
    imagesc(ax, mats{kk}); axis(ax,'equal','tight'); clim(ax,[0 c_max]);
    colormap(ax, parula);
    set(ax,'FontSize',10,'TickDir','out');
    title(ax, ttls{kk}, 'FontSize', 12);
    if kk==4, colorbar(ax); end
end
title(tl5, 'Covariance matrices', 'FontSize', FSt, 'Interpreter','latex');
 
%%  LOCAL FUNCTIONS
 
function df = foerstner_distance(gamma1, gamma_pos, W)
    Lc  = chol(W' * gamma1    * W, 'lower');
    Rc  = chol(W' * gamma_pos * W, 'lower');
    tmp = Lc' / Rc';
    s   = svd(tmp);
    lam = s.^2;
    lam(abs(lam) < eps) = [];
    df  = dot(log(lam), log(lam));
end