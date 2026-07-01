# Combining Dimensionality Reduction (LIS) with Surrogate Modeling

MATLAB implementation combining the **Likelihood-Informed Subspace (LIS)** framework
with **Gaussian-process (GP) surrogate modeling** for a linear Bayesian inverse
problem: recovering a distributed axial load on a clamped 1D structural bar from
sparse, noisy displacement measurements.

## Idea

A Bayesian inverse problem asks: given a noisy observation `y = Gθ + e` and a
prior on `θ`, what is the posterior `p(θ | y)`? When `θ` is high-dimensional but
the data is comparatively low-dimensional, most of `θ`'s directions are barely
updated by the data — only a handful of directions are genuinely
**likelihood-informed**.

This project does two things, in sequence:

1. **LIS reduction.** Identify the informed subspace exactly (Spantini et al.,
   2015): find the rank-`r` reduction of the forward operator that gives the
   *provably optimal* rank-`r` approximation to the posterior covariance, in the
   Förstner metric. The informed rank `r*` is not chosen by hand — it emerges
   from the physical/statistical parameters (noise level, sensor count, prior
   smoothness).
2. **GP surrogate.** Replace the (potentially expensive) forward operator `G`
   with a Gaussian-process surrogate trained *only* in the reduced
   `r`-dimensional LIS coordinate, and rebuild the posterior using the
   surrogate's noise-inflated predictions instead of ever calling `G` again
   during inference (Rasmussen & Williams, 2006; Villani et al., 2024). The
   experiment sweeps the surrogate's training budget and the reduced dimension
   `r`, measuring how close the surrogate-based posterior gets to the exact one.

The bar problem is small enough that the *exact* posterior can also be computed
in closed form (Woodbury identity), so the whole pipeline can be validated
against ground truth before being pointed at a problem where the forward model
is genuinely too expensive to call more than a handful of times.

## Repository structure

```
main.m                          single driver script, run this
src/
  assembly_G.m                  forward operator G = C K^-1 (clamped 1D bar FEM)
  assembly_prior.m               singular nodal-force prior (OU kernel + square root)
  assembly_load.m                ground-truth distributed load -> theta_true
  gen_samples.m                  prior sampling via the square-root factor
utils_gp/
  kernel_se.m                    squared-exponential kernel
  gp_predict.m                   GP predictive mean/variance (Cholesky, R&W Alg. 2.1)
  gp_nlml.m                      negative log marginal likelihood
  gp_fit_hyperparameters.m       hyperparameter fitting by evidence maximization
```

`main.m` is self-contained: `src/` builds the physical problem and prior,
`utils_gp/` is a generic GP regression toolkit with no knowledge of the bar
problem — the LIS coordinate is just another input to it.

## What `main.m` does

**Part A — LIS (Sections 1–5).** Assembles the bar's forward operator and a
singular prior (an Ornstein–Uhlenbeck kernel on the distributed load, pushed
through the FEM load-consistency operator into nodal-force space — singular by
construction, handled via its square-root factor throughout, never inverted).
Computes the exact posterior via the Woodbury identity, then builds the LIS
basis via the whitened SVD route (Spantini, Remark 4): the generalized
eigenvalues `δᵢ²` rank every direction in parameter space by how much the data
informs it relative to the prior. `r* = #{δᵢ² > 1}` emerges from the chosen
parameters. For every rank `r`, the optimal rank-`r` (OLR) posterior is
computed via the oblique projector `P_r = V_r W_r'` (Spantini, Cor. 3.2) and
compared to the exact posterior via the Förstner distance.

**Part B — GP replacing `G` (Sections 6–7b).** For `r ∈ {1,2,3}` and a sweep of
training budgets `n_tr ∈ {5,10,20,40,80}`: draw `n_tr` samples from the *full*
prior, project each into the LIS coordinate `w = W_r'(θ − μ_f)`, evaluate the
*true* forward model to get training targets, fit one independent GP per
sensor, and build the posterior over `w` via a noise-inflated marginal
likelihood (Villani eq. 3) evaluated on a grid. The resulting reduced posterior
is lifted back to full parameter space and compared to the exact posterior.
Each `(r, n_tr)` combination is averaged over multiple random training sets to
report a stable mean and spread. Sections 7/7b additionally keep one concrete
posterior-covariance matrix each at the cheapest and most expensive training
budgets, for direct visual comparison against the exact and OLR posteriors.

## Output

Running `main` opens a single tabbed figure window with five views:

| Tab | Content |
|---|---|
| P1 | LIS eigenvalue spectrum, `r*` threshold |
| P2 | OLR posterior accuracy (Förstner distance) vs. rank |
| P3 | Covariance matrices side by side (Prior / Exact / OLR / GP-cheap / GP-expensive), plus a second row isolating the correction-from-exact on its own color scale |
| P4 | GP surrogate error vs. training budget at `r = 1` (mean ± 1 std band vs. the OLR floor) |
| P5 | Same, overlaid for `r = 1, 2, 3` |

## Requirements

MATLAB (tested informally against Octave 8.4 for the computational core; the
tabbed-figure plotting relies on MATLAB-only UI components — `uitabgroup`,
`tiledlayout` inside a `uitab` — and is not expected to render in Octave).

## References

- Spantini, Solonen, Cui, Martin, Tenorio, Marzouk (2015). *Optimal low-rank
  approximations of Bayesian linear inverse problems.*
- Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning.*
- Villani et al. (2024). *Adaptive Gaussian Process Regression for Bayesian
  Inverse Problems.*
