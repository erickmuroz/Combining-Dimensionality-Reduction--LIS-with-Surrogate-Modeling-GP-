%% ========================================================================
%  main.m  -  LIS on the 1D bar + GP surrogate replacing G
%
%  src/    : assembly_G, assembly_prior, assembly_load    
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
 
N_rep_gp = 8; %controls how many random GP training points sets get averaged over. 
 
%% ========================================================================
%  PART A - LIS  (Jakob's base)
%  ========================================================================
 
%% 1. forward operator G and prior operators
[G, con] = assembly_G(L, nele, EA, m, BC_dof, sensor_seed);  % con = bundle of everything the FEM assembly computed besides G
L_mat = con.L_mat;
[Gamma_pr, S_pr, mu_f, gamma_q] = assembly_prior(nele, L_mat, l, mu_q, sigma_q, ell_pr);
k_pr = size(S_pr, 2);% size of the random vector I need for sampling in gen samples i.e. input size of the sampling machine
 
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
 
tol   = max(size(R'*S_pr)) * eps(max(delta));   %numerical rank threshold
r_eff = sum(delta > tol); %keep only the numerically meaningful
U     = U(:, 1:r_eff);
delta = delta(1:r_eff);
Z     = Z(:, 1:r_eff);
 
V = S_pr * Z; %from LIS to full parameter vector via V_r * w 
W = R * U .* (1 ./ delta)'; %the other way around
r_plot = max(sum(delta.^2 > 1), 1);   
 
%% 5. OLR posterior at each rank  (reduced operator, Jakob's sections 6-7)
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
%  PART B - GP surrogate replacing G, trained in the LIS coordinate w
%  ========================================================================
 
%% 6. Forstner error of the GP posterior vs training budget, for r = 1,2,3
% Goal: If I only allow myself $n_{tr}$ calls to the true $G$, train a 
% surrogate on those calls, and then build a posterior using only the surrogate, 
% how close does the posterior get to the true one, and how does the closeness depend on $n_{tr}$ 
% and on the reduced dimension $r$?
 
r_gp_list    = [1 2 3];  
r_gp_list = r_gp_list(r_gp_list <= r_eff); %threshold for actual r*
n_train_list = [5 10 20 40 80];
Lw           = 4; %for lis_grid build
ng           = containers.Map({1,2,3}, {400, 41, 19}); %diff grid span for different r-dimension 
 
%empty matrices prefilled with NaN 
ERR_MEAN  = nan(numel(r_gp_list), numel(n_train_list)); %one row per rank tested, onw column per training budget tested 
ERR_STD   = nan(numel(r_gp_list), numel(n_train_list));
OLR_floor = nan(numel(r_gp_list), 1); %keeping the OLR error as a floor for the Foerstner x n plot 
 
for ri = 1:numel(r_gp_list)
    r  = r_gp_list(ri);
    Vr = V(:,1:r); %slice out just the first r columns of the full LIS 
    Wr = W(:,1:r); %bases built in section 4 
    [Wg, dV, is1d] = lis_grid(r, Lw, ng(r));
    OLR_floor(ri)  = foerstner_distance(Gamma_OLR{r}, Gamma_pos, W);
 
    for ti = 1:numel(n_train_list) %fix a training set size 
        nt   = n_train_list(ti);
        errs = zeros(N_rep_gp, 1);
        for rep = 1:N_rep_gp %% training data generation
            rng(1000*rep + ti);
            Theta = mu_f + S_pr*randn(k_pr, nt);%drawing samples from the full prior -> nt= 5 hypothetical situations to try
            Wtr   = (Wr' * (Theta - mu_f))';% project every one of the nt samples down into w (LIS) coordinate
            %this are inputs the GP will eventually see 
 
            Ytr   = (G * Theta)';
            %evaluation of the true expensive forward model at those nt
            %samples, output data the surrogate the surrogate will train on
            %(generating training data) => 
            % we get a table of (w_i, y_i) pairs (with w as the input, and
            % y as the output, ii.e. sensor redings) 
            
 
            covw  = gp_lis_covw(Wtr, Ytr, Wg, dV, y, gamma_obs_var, is1d); %posterior variance of w according to the GP 
            Ggp   = Gamma_pr - Vr*(eye(r) - covw)*Vr'; %expand it back to n x  n covariance in parameter space = Gamma_OLR{1}
            errs(rep) = foerstner_distance(0.5*(Ggp+Ggp'), Gamma_pos, W); %averaged measure of how far the surrogate model is from the exact 
        end
        % once all N_rep_gp = m have run, m independent random training
        % sets, m values of covw, m values of Ggp -> everything computed
        % into one mean and one standard deviation
        ERR_MEAN(ri,ti) = mean(errs); %solid error line in plot 
        ERR_STD(ri,ti)  = std(errs); %shaded band 
    end
    fprintf('GP r = %d done   (OLR floor = %.3e)\n', r, OLR_floor(ri));
end
 
%% 7. GP posterior covariance at r* (best budget, single draw) for the matrix plot
r  = min(r_plot, max(r_gp_list));
%using the larges training budget ntr =80
Vr = V(:,1:r);  Wr = W(:,1:r);
[Wg, dV, is1d] = lis_grid(r, Lw, ng(r));
rng(2024);
Theta = mu_f + S_pr*randn(k_pr, n_train_list(end));
Wtr   = (Wr' * (Theta - mu_f))';
Ytr   = (G * Theta)';
covw_star     = gp_lis_covw(Wtr, Ytr, Wg, dV, y, gamma_obs_var, is1d);
Gamma_GP_star = Gamma_pr - Vr*(eye(r) - covw_star)*Vr';
Gamma_GP_star = 0.5*(Gamma_GP_star + Gamma_GP_star');
 
%% 7b. GP posterior covariance at r* with the CHEAPEST budget (n_train_list(1))
% same machinery as section 7, only swapping n_train_list(end) -> n_train_list(1).
% At small n_tr a single draw is noisy (this is exactly the std band in P4), so
% among a handful of candidate seeds we keep the one whose Forstner error is
% closest to the already-computed MEAN error at this budget (ERR_MEAN) -- the
% matrix panel then shows a REPRESENTATIVE realization, not a lucky/unlucky one.
nt_cheap   = n_train_list(1);
ri_cheap   = find(r_gp_list == r, 1);
target_err = ERR_MEAN(ri_cheap, 1);
 
best_gap = Inf;
for s = 2025:2034
    rng(s);
    Theta_try = mu_f + S_pr*randn(k_pr, nt_cheap);
    Wtr_try   = (Wr' * (Theta_try - mu_f))';
    Ytr_try   = (G * Theta_try)';
    covw_try  = gp_lis_covw(Wtr_try, Ytr_try, Wg, dV, y, gamma_obs_var, is1d);
    Gtry      = Gamma_pr - Vr*(eye(r) - covw_try)*Vr';
    Gtry      = 0.5*(Gtry + Gtry');
    err_try   = foerstner_distance(Gtry, Gamma_pos, W);
    gap       = abs(err_try - target_err);
    if gap < best_gap
        best_gap = gap;  Gamma_GP_cheap = Gtry;  best_seed = s;
    end
end
fprintf('GP cheap panel: seed %d chosen (Forstner=%.3e, target mean=%.3e)\n', ...
        best_seed, foerstner_distance(Gamma_GP_cheap, Gamma_pos, W), target_err);
 
 
set(0,'defaulttextinterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
FS = 13;  FSt = 15;
col_exact = [0.15 0.28 0.62];      % deep blue
col_olr   = [0.85 0.45 0.10];      % orange
col_gp    = [0.06 0.55 0.46];      % teal
col_thr   = [0.75 0.10 0.10];      % red
cmap_cov  = parula;
style_ax  = @(ax) set(ax,'FontSize',FS,'Box','off','LineWidth',0.9, ...
                    'TickDir','out','XGrid','on','YGrid','on', ...
                    'GridAlpha',0.12,'Layer','top');
 
% one window, one tab per figure 
fig = figure('Color','w','Position',[60 60 1180 820], 'Name', 'LIS + GP results');
tg  = uitabgroup(fig);
 
%% plot 1: LIS eigenvalue spectrum
axes('Parent', uitab(tg, 'Title', 'P1 Spectrum'));
h = semilogy(1:r_max, delta.^2, 'o-', 'Color', col_exact, ...
        'MarkerFaceColor', col_exact, 'MarkerSize', 7, 'LineWidth', 1.8);
hold on;
yline(1, '--', 'Color', col_thr, 'LineWidth', 1.6);
yl = ylim;
patch([0.5 r_plot+0.5 r_plot+0.5 0.5], [yl(1) yl(1) yl(2) yl(2)], ...
      col_exact, 'FaceAlpha', 0.06, 'EdgeColor', 'none');
uistack(h,'top');  ylim(yl);
text(r_plot+0.35, 1.6, '$\delta_i^2=1$', 'Color', col_thr, 'FontSize', 11);
text((r_plot+0.5+r_max)/2, yl(2)*0.4, sprintf('$r^*=%d$',r_plot), ...
     'Color', col_exact, 'FontSize', 12, 'HorizontalAlignment','center');
style_ax(gca);
xlabel('LIS direction $i$'); ylabel('$\delta_i^2$');
title('LIS eigenvalue spectrum', 'FontSize', FSt);
xlim([0.5 r_max+0.5]);
 
%% plot 2: Forstner error of the OLR posterior vs rank
axes('Parent', uitab(tg, 'Title', 'P2 OLR error'));
semilogy(1:r_max, forstner_rank, 'o-', 'Color', col_olr, ...
        'MarkerFaceColor', col_olr, 'MarkerSize', 7, 'LineWidth', 1.8);
hold on; xline(r_plot, '--', 'Color', col_thr, 'LineWidth', 1.5);
text(r_plot+0.15, max(forstner_rank)*0.4, sprintf('$r^*=%d$',r_plot), ...
     'Color', col_thr, 'FontSize', 12);
style_ax(gca);
xlabel('Rank $r$'); ylabel('F\"orstner distance to exact');
title('OLR posterior accuracy vs rank', 'FontSize', FSt);
xlim([0.5 r_max+0.5]);
 
%% plot 3: covariance comparison  Prior | Exact | OLR(r*) | GP cheap | GP expensive
c_max = max(Gamma_pr(:));
tab3 = uitab(tg, 'Title', 'P3 Covariances');
tl3  = tiledlayout(tab3, 2, 5, 'Padding','compact', 'TileSpacing','compact');
mats = {Gamma_pr, Gamma_pos, Gamma_OLR{r_plot}, Gamma_GP_cheap, Gamma_GP_star};
ttls = {'Prior $\Gamma_{pr}$', 'Exact $\Gamma_{pos}$', ...
        sprintf('OLR $r=%d$', r_plot), ...
        sprintf('GP cheap ($n_{tr}=%d$)', nt_cheap), ...
        sprintf('GP expensive ($n_{tr}=%d$)', n_train_list(end))};
for kk = 1:5
    nexttile(tl3);
    imagesc(mats{kk}); axis equal tight; clim([0 c_max]); colormap(gca, cmap_cov);
    set(gca,'FontSize',11,'TickDir','out');
    xlabel('$j$'); if kk==1, ylabel('$k$'); end
    title(ttls{kk}, 'FontSize', 13);
    if kk==5, cb = colorbar; cb.TickLabelInterpreter = 'latex'; end
end
 
% ---- row 2: correction-from-exact, OWN diverging color scale --------------
% The panels above all share the prior's scale (c_max), so the rank-r
% correction is only ~0.5-1% of that scale and is invisible by eye there,
% even though it differs clearly in Forstner distance (Section 6/7). These
% two panels isolate the correction itself, auto-scaled to its own range.
diff_cheap = Gamma_pos - Gamma_GP_cheap;
diff_star  = Gamma_pos - Gamma_GP_star;
d_max      = max(abs([diff_cheap(:); diff_star(:)]));
n_div      = 256;
cmap_div   = [linspace(col_exact(1),1,n_div/2)' linspace(col_exact(2),1,n_div/2)' linspace(col_exact(3),1,n_div/2)'; ...
              linspace(1,col_olr(1),n_div/2)'   linspace(1,col_olr(2),n_div/2)'   linspace(1,col_olr(3),n_div/2)'];
 
nexttile(tl3, 9);
imagesc(diff_cheap); axis equal tight; clim([-d_max d_max]); colormap(gca, cmap_div);
set(gca,'FontSize',11,'TickDir','out');
xlabel('$j$'); ylabel('$k$');
title(sprintf('Exact $-$ GP cheap ($n_{tr}=%d$)', nt_cheap), 'FontSize', 12);
 
nexttile(tl3, 10);
imagesc(diff_star); axis equal tight; clim([-d_max d_max]); colormap(gca, cmap_div);
set(gca,'FontSize',11,'TickDir','out');
xlabel('$j$');
title(sprintf('Exact $-$ GP expensive ($n_{tr}=%d$)', n_train_list(end)), 'FontSize', 12);
cb2 = colorbar; cb2.TickLabelInterpreter = 'latex';
 
%% plot 4: GP Forstner error vs training points, r = 1  (mean +/- 1 std band)
ri1 = find(r_gp_list==1, 1);
axes('Parent', uitab(tg, 'Title', 'P4 GP error r=1'));
hold on;
band_plot(n_train_list, ERR_MEAN(ri1,:), ERR_STD(ri1,:), col_gp);
yline(OLR_floor(ri1), '--', 'Color', col_olr, 'LineWidth', 1.6);
text(n_train_list(end), OLR_floor(ri1)*1.2, 'OLR floor', ...
     'Color', col_olr, 'FontSize', 11, 'HorizontalAlignment','right');
set(gca,'YScale','log'); style_ax(gca);
xlabel('GP training points $n_{tr}$'); ylabel('F\"orstner distance to exact');
title('GP surrogate in LIS coordinate ($r=1$)', 'FontSize', FSt);
legend({'$\pm 1$ std','mean','OLR floor'}, 'Location','northeast');
xlim([n_train_list(1) n_train_list(end)]);
 
%% plot 5: GP Forstner error vs training points, overlaid for r = 1,2,3
col_r = [0.06 0.55 0.46;      % r=1  teal
         0.15 0.28 0.62;      % r=2  blue
         0.55 0.20 0.55];     % r=3  purple
axes('Parent', uitab(tg, 'Title', 'P5 GP error r=1,2,3'));
hold on;
hleg = [];  labs = {};
for ri = 1:numel(r_gp_list)
    band_plot(n_train_list, ERR_MEAN(ri,:), ERR_STD(ri,:), col_r(ri,:));
    hleg(end+1) = plot(nan,nan,'o-','Color',col_r(ri,:), ...
                       'MarkerFaceColor',col_r(ri,:),'LineWidth',1.8);
    labs{end+1} = sprintf('$r=%d$', r_gp_list(ri));
end
yline(OLR_floor(find(r_gp_list==r_plot,1)), '--', 'Color', col_thr, 'LineWidth', 1.5);
hleg(end+1) = plot(nan,nan,'--','Color',col_thr,'LineWidth',1.5);
labs{end+1}  = sprintf('OLR floor ($r^*=%d$)', r_plot);
set(gca,'YScale','log'); style_ax(gca);
xlabel('GP training points $n_{tr}$'); ylabel('F\"orstner distance to exact');
title('GP surrogate posterior error vs reduced dimension $r$', 'FontSize', FSt);
legend(hleg, labs, 'Location','northeast');
xlim([n_train_list(1) n_train_list(end)]);
 
set(0,'defaulttextinterpreter','remove');
set(0,'defaultAxesTickLabelInterpreter','remove');
set(0,'defaultLegendInterpreter','remove');
 
%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================
 
function df = foerstner_distance(gamma1, gamma_pos, W)
    % Squared Forstner distance in projected (whitened) coordinates.
    % Force-space covariances are singular -> project into W'(.)W (SPD there),
    % then squared Forstner via generalized eigenvalues. 
    Lc  = chol(W' * gamma1    * W, 'lower');
    Rc  = chol(W' * gamma_pos * W, 'lower');
    tmp = Lc' / Rc';
    s   = svd(tmp);
    lam = s.^2;
    lam(abs(lam) < eps) = [];
    df  = dot(log(lam), log(lam));
end
 
function [Wg, dV, is1d] = lis_grid(r, Lw, ngrid)
    % Integration grid for the posterior over the LIS coordinate w in R^r.
    g = linspace(-Lw, Lw, ngrid); %straigth line 
    if r == 1
        Wg = g'; %makes the linspace as a column vector 
        dV = NaN; %no cell volume needed for numerical integrattion 
        is1d = true; %is it 1D?
    else % more dimension case (AI)
        dx   = g(2) - g(1);
        cols = cell(1, r);  [cols{:}] = ndgrid(g);
        Wg   = zeros(numel(cols{1}), r);
        for k = 1:r, Wg(:,k) = cols{k}(:); end
        dV   = dx^r;  is1d = false;
    end
end
 
function covw = gp_lis_covw(Wtr, Ytr, Wg, dV, y_obs, gobs, is1d)
    % GP surrogate posterior over the LIS coordinate w  (folders 01-09 style).
    %  - one independent GP per sensor output j (hyperparameters by marginal
    %    likelihood), predictive mean + variance on the grid;
    %  - noise-inflated marginal likelihood (Villani eq. 3): S_j = gobs+var_GP;
    %  - multiply by the prior N(0,I_r), normalize on the grid, return covw.
    m      = numel(y_obs);
    loglik = zeros(size(Wg,1), 1);
    for j = 1:m
        [ell, sf2, sn2] = gp_fit_hyperparameters(Wtr, Ytr(:,j), 'se'); %call for determining the hyperparameters
        kf = @(A,B) kernel_se(A, B, ell, sf2);
        [mu, vlat] = gp_predict(Wtr, Ytr(:,j), Wg, kf, sn2); %GP predictions over lis_grid
        Sj     = gobs + vlat + sn2; %real noise + uncertainty, how much total unceratinty is there about what sensor j would read at this w 
        loglik = loglik - 0.5*log(2*pi*Sj) - 0.5*(y_obs(j) - mu).^2 ./ Sj;
    end
    logpost = -0.5*sum(Wg.^2, 2) + loglik;                   % prior N(0,I_r) + lik
    p = exp(logpost - max(logpost));
 
    if is1d
        p    = p / trapz(Wg, p);
        mw   = trapz(Wg, Wg .* p);
        covw = trapz(Wg, (Wg - mw).^2 .* p);
    else
        p    = p / (sum(p) * dV);
        mw   = (sum(Wg .* p, 1) * dV)';
        dW   = Wg - mw';
        covw = (dW' * (dW .* p)) * dV;
    end
 
    covw  = 0.5*(covw + covw');
    [Q,D] = eig(covw);
    dd    = min(max(real(diag(D)), 1e-12), 1);               % clip to (0,1]
    covw  = Q * diag(dd) * Q';
    covw  = 0.5*(covw + covw');
end
 
function band_plot(x, m, s, col)
    % mean line + shaded +/-1 std band (clamped positive for the log axis)
    lo = max(m - s, m*0.15);
    hi = m + s;
    fill([x fliplr(x)], [lo fliplr(hi)], col, 'FaceAlpha', 0.15, 'EdgeColor','none');
    plot(x, m, 'o-', 'Color', col, 'MarkerFaceColor', col, ...
         'MarkerSize', 7, 'LineWidth', 1.8);
end