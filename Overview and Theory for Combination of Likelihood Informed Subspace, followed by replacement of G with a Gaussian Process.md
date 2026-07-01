# 1. Physics and Prior Build (Source Folder)
#### 1.1 Assembly_G.m
**Physics**: The problem involves a clamped 1D bar under static axial load which obeys $Ku=f$. With $\text{nele}$ linear bar elements of length $l=\frac{L}{\text{nele}}$ and axial stiffness $EA$, each element contributes a local stiffness $k_{e}=\frac{EA}{l}$ assembled into a standard tridiagonal pattern in "sparse(nnode, nnode)". 

**Observational model**: The full displacement field is not observed, only $m$ sensor readings: $y=Cu$, where $C$ is a 0/1 selection matrix picking our sensor DOFs. With $u=K^{-1}f$, yields:

$$
y=CK^{-1}f=Gf, \qquad G=CK^{-1}
$$

which is the single linear map from nodal force to the sensor reading (i..e the entire forward model for the linear case)

**Boundary condition**: Clamping the bar, removes one DOF from all matrix operators involved (like $K,C,L_{mat}$)
$L_{mat}$ is not part of G, it is defined as the load consistency operator, mapping a distributed load (defined at element midpoints) to equivalent nodal forces via Standard FEM lumping

**Sensors**:
rng(sensor_seed) fixes which $m$ interior nodes are observed

#### 1.2 Assembly_load.m
This defines the whole $\theta_{true}$ true nodal force which is intended to recover, used to synthesize the noise observations $y_{obs}$ from it. 

**Physical Choice**: Distributed load is chosen to be a baseline plus a Gaussian bump at mid-span:

$$
q(z)=q_{0}+\text{bump}\cdot \exp(-\frac{(z-z_{c})^2}{2w^2})
$$

evaluated at element midpoints $x_{q}$, mapped to nodal force via $L_{mat}$: $\theta_{true}=L_{mat}q(x_{q})$. 

#### 1.3 Assembly_prior.m
**The kernel**: The prior belief about the distributed load $q$ at $\text{nele}$ midpoints is an Ornstein-Uhlenbeck (exponential) covariance: 

$$
\gamma_{q}(x_{i},x_{j})=\sigma^2_{q}\exp(-\frac{|x_{i},x_{j}|}{\ell})
$$

which encodes: load values close together (within correlation length $\ell$) are strongly correlated; far apart, nearly independent = hence smooth prior structure.

**Push-forward to nodal-force space**: The unknown parameter $\theta$ in the implementation is the nodal force $f$, not $q$ itself (note: Assembly_g's Forward map $G$ eats $\mathbf{f}$). So the prior on $q$ has to be pushed through the same linear map $L_{mat}$ used in the FEM load-lumping:

$$
\Gamma_{pr}=L_{mat}\gamma_{q}L_{mat}^T, \qquad \mu_{f}=L_{mat}\mu_{q}\mathbf{1}
$$

**Why this matters, what is the deal with the $L_{mat}$ mapping**:
= Rank deficiency: $L_{mat}\in \mathbb{R}^{n_{node}\times n_{ele}}$, in general not square, so pushing a covariance through a non-trivial linear map can produce a singular (rank-deficient or numerically near-singular) covariance prior matrix. The problem, a singular covariance has no inverse, but Bayesian updating requires the formulation of the prior precision $\Gamma_{pr}^{-1}$.

It will be fixed by not representing the prior precision directly, but only its square root. Since $\gamma_{q}$ itself is SPD, it admits a Cholesky factor $\gamma_{q}=L L^T$. Pushing through $L_{mat}$:

$$
\Gamma_{pr}=(L_{mat}L)(L_{mat}L)^T=S_{pr}S_{pr}^T, \qquad S_{pr}=L_{mat}\cdot \text{chol}(\gamma_{q}, \text{'lower'})
$$

such that $S_{pr}$ exists and it is well defined even when $\Gamma_{pr}$ is singular. Everything will be written in terms of $S_{pr}$. 

#### 1.4 gen_samples.m
Standard linear-Gaussian sampling trick: if $\theta\sim \mathcal{N}(\mu_{f},\Gamma_{pr})$ then: 

$$
\theta_{i}=\mu_{f}+S_{pr}\xi_{i}, \qquad \xi\sim \mathcal{N}(0,I)
$$

produces exact prior samples. This functions draws $N$ random vectors $\theta_1,\dots\theta_{N}$ from the prior distribution. In MATLAB "randn" only gives you samples from standard normal, but it is solved via $\theta_{i}$ from above. So that if one can find an $S$ that builds up $\Gamma_{pr}$, then $\theta_{i}=\mu_{f}+S_{pr}\xi_{i}$ has exactly the right mean and exactly the right covariance.

# 2.  GP utils in Isolation (Utilities Folder)
#### 2.1 kernel_se.m
A GP is defined by a covariance function $k(x,x')$, which encodes how correlated do I believe two outputs $f(x),f(x')$ are, just knowing how far apart $x$ and $x'$ are. 
SE stands for squared exponential kernel

$$
k(x,x')= \sigma_{f}^2\exp(-\frac{||x-x'||^2}{2 \ell ^2})
$$
- $\sigma^2_{f}$ - "sf2": signal variance for how much the function varies overall
- $\ell$ - "ell": length scale encoding over what distance the correlation decays

Computed via $||a-b||=||a||^2+||b||^2-2a\cdot b$ 

#### 2.2 gp_predict.m
This will be called via the util function gp_predict. Starting from the joint Gaussian prior over training and test points

$$
\begin{pmatrix}
	y_{tr} \\
	y(x_{*})
\end{pmatrix}\sim \mathcal{N}(0,\begin{pmatrix}
	K+\sigma^2_{n}I & k_{*} \\
	k_{*}^T&k_{**}
\end{pmatrix})
$$


, conditioning on the training data gives:

$$
\bar{y}(x_{*})=k_{*}^T(K+\sigma_{n}^2I)^{-1}y_{tr} \qquad (\text{RW 2.23})
$$
where $\bar{y}(x_{*})$ are the weighted sum of the training targets $y_{tr}$, where the weights $(\cdot)$ are large for training points $p_{i}$ that are close to $x_{*}$. 

$$
\sigma^2(x_{*})=k(x_{*},x_{*})-k_{*}^T(K+\sigma_{n}^2I)^{-1}k_{*}\qquad \text{(RW 2.24)}
$$here, $\sigma^2(x_{*})$ starts from the prior kernel variance and subtracts a non negative number that grows with how much information the training data carry about $x_{*}$. (At a training point $x_{*}=x_{i}$ the variance collapses to the prior variance, $k_{**}=k(x_{i},x_{i})=\sigma_{f}^2$. )
	Note: the subtracted term grows when $k_{*}$ has large entries, i.e. when the training points are close to $x_{*}$, relative to $l$. The uncertainty band is wide when the spacing between the training points $\delta$ is much larger than the ell 

#### 2.3 gp_nlml.m
$\text{nlml}=\text{- lml}$ from gp_predict and parameterized in the log-space such that kernel hyperparameters stay strictly positive. 

#### 2.4 gp_fit_hyperparameters.m
(R&W section 5.4.1): This accounts for empirical Bayes / evidence maximization. Instead of choosing hyperparameters by hand, the data itself picks them by maximizing the marginal likelihood (i.e. minimizing gp_nlml). A higher evidence means that the kernel explains the observed data well, without being needlessly flexible. 

With initial guesses:
- ell0= median pairwise distance between training points
- sf20=var(ytr): observation at training points is a good first guess for sf20 
- sn20=0.1 x var(ytr): Assuming data is mostly signal, less noise

# 3. LIS Framework (sec 1-5)
#### 3.0 Setup & parameters
**Informative directions $r^*$**:
$r^*$ is not forced, it emerges after picking the right parameters. It is chosen to be controlled by:
- gamma_obs_var (observation noise variance): $\delta^2_{i}\propto \frac{1}{\gamma_{obs_{var}}}$, i.e. larger gamma_obs_var, smaller $r^*$
- m (number of sensors): More sensors mean more rows in $G$, increasing the rank and the magnitude of the singular values of $GS_{pr}$. In general more sensors $\to$ more informed directions available. (Not a guarantee since the location of the sensors also play a role)
- ell_pr (prior correlation length): Larger correlation length makes the prior smoother, making the effective number of independent shapes the load can take is smaller. So in general, larger ell_pr, smaller $r^*$.
**Physical Setup**:
L=2 bar length, with 100 nele (finite elements), given axial stiffness with m=10 sensors with sensor_seed = 41 (fixes which 10 of the interior nodes are instrumented)

#### 3.1 Forward and prior Operators
Here assembly_G builds the forward operator $G$ and the FEM structure and assembly_prior buils the singular nodal-force prior $\Gamma_{pr}$ via its square-root factor $S_{pr}$. 

One new quantity is introduced k_pr = size(S_pr, 2), which is the latent dimension of the prior's square root factor. If we recall, $S_{pr}=L_{mat}\cdot \text{chol}(\gamma_{q})$ has shape $n_{node} \times n_{ele}$, its number of rows equals $n$ (the dimension of $\theta$) and the columns are equal to nele = input size of the sampling machine - the length of the raw standard-normal vector $S_{pr}$ needs to consume in order to produce one legitimate sample of $\theta$ from the prior $\mathcal{N}(\mu_{f},\Gamma_{pr})$. 

#### 3.2  Ground truth and synthetic observations
A true but unknown parameter $\theta_{true}$ exists, but one can only get to see a noisy observation from it: 

$$
y=G\theta_{true}+e, \quad e\sim \mathcal{N}(0,\Gamma_{obs}), \quad \Gamma_{obs}=\gamma_{obs_{var}}I_{m}
$$

This gives a standard linear-Gaussian observation model. Every sensor has identical Gaussian noise with variance 'gamma_obs_var'. $\theta_{true}$ comes from assembly_load function. The baseline + Gaussian bump distributed load is pushed through $L_{mat}$, into nodal-force space. 

#### 3.3 Exact posterior via the Woodbury Identity
For a linear Gaussian with a non singular prior, the standard Bayesian update would use the prior precision $\Gamma^{-1}_{pr}$. In this case, we established a potentially singular construction so it must be approached in the following way: 

$$\begin{align*}
	\Gamma_{pos}= \Gamma_{pr}- \Gamma_{pr}G^T(G\Gamma_{pr}G^T+\Gamma_{obs})^{-1}G\Gamma_{pr}, \\ 
	\mu_{pos}=\mu_{f}+ \Gamma_{pr}G^T(G\Gamma_{pr}G^T+\Gamma_{obs})^{-1}(y-G\mu_{f}).
\end{align*}
$$

The only inverse required is of $(G\Gamma_{pr}G^T+\Gamma_{obs})$, an $m\times m$ matrix, that is generically well conditioned because $\Gamma_{obs}$ adds a positive diagonal, which is more friendlier object to invert than the potentially singular $n\times n$ prior. 

#### 3.4 LIS Basis via the Whitened SVD
First, we get $R=G^T\Sigma_{obs}^{-1/2}$, dividing each sensor contribution to $G^T$ by the sensor's noise standard deviation (why: a sensor could look important merely because it has large raw sensitivity, even if its noise is also large). Whitening, puts every sensor on a common, noise-normalized footing, so that what remains, genuinely reflects information content. This mirrors the standard whitening step in generalized eigenvalue LIS formulations (Spantini et al. Remark 4): The pencil I want the eigendecomposition of is $(H,\Gamma_{pr}^{-1})$, where $H=G^T\Gamma_{obs}^{-1}G$ is the Fisher-information-like Hessian of the (log-)likelihood. $R$ is exactly the square-root factor of $H$, since $H=RR^T$. 

This "Whitening" avoids ever forming or inverting $\Gamma_{pr}$: instead of solving a generalized eigenvalue problem $Hv=\lambda\Gamma_{pr}^{-1}v$ directly, the code exploits that $\Gamma_{pr}=S_{pr}S_{pr}^T$ and takes a simple SVD of the product $R^TS_{pr}$. The generalized eigenvalues of this product are equal to the square roots of the generalized eigenvalues of the pencil $(H, \Gamma_{pr}^{-1})$.

- Numerical rank threshold: Floating point SVD will always return as many singular values as the smaller matrix dimension, but many of the smallest ones are just noise. Tol handles this.

Then the two bases $V$ and $W$ are built. 
- $V$ (reconstruction basis): given a reduced coordinate $w\in \mathbb{R}^r$, $V_{r}w$ maps it back up into full $\theta$-space. This is what will be used to go from the LIS coordinate to the full parameter vector.
- $W$ (projection/dual basis): given a full $\theta$, $W^T_{r}(\theta-\mu_{f})$ projects it down into the $r$-dimensional LIS coordinate $w$. 

Remark on coordinate $w$:
$w = W^T_{r}(\theta-\mu_{f})$ is a single real number (for r=1). Given any load vector $\theta$, $w$ tells how far, and in which direction, $\theta$ deviates from the prior mean along the one informed direction. 
Going back to a sensor reading: Given a candidate value of $w$, one can reconstruct a load vector $\theta(w)=\mu_{f}+V_{r}w$, so if I had the true forward $G$, I could ask, what would the sensor read for this $\theta$:

$$
y_{j}(w)=[G(\mu_{f}+V_{r}w]
_{j}\quad j=1, \dots, 8
$$
This here, is the function we are trying to learn

#### 3.5 OLR posterior per rank + Oblique Projector
Theorem 2.3 says: among all possible rank-$r$ approximations to the posterior covariance, the one built by projecting the forward operator through the top-$r$ LIS directions is probably the best one, in Förstner metric. So for chosen rank $r$:

$$
P_{r}=V_{r}W_{r}^T
$$

The reduced forward operator and its posterior can be computed using this: $G_{r}=GP_{r}$

The Förstner distance: foerstner_rank(r) measures for each $r$, how far Gamma_OLR{r} is from the true Gamma_pos. 
# 4. Replacing $G$ with a Gaussian Process (sec 6-7)
#### 4.1 (6): Sweeping GP Accuracy over Rank and Training Budget

**Replacing $G$ with a $GP$**: The conceptual reframe that is implemented in section 6 follows the idea of training a cheap statistical surrogate that predicts what $G$ would have said, as a function of the reduced LIS coordinate $w$ alone. Then, a posterior will be built, but using that surrogate's predictions instead of ever calling $G$ again during inference.

What is being predicted?: With the toy example, the input $x$ was literally a spatial coordinate, and the function being learned was  $f(x)=\sin(x)$, i.e. a fixed ordinary scalar function. The goal was to verify if the GP could correctly reconstruct a known function from sparse samples. Now in the LIS framework, the input is still a coordinate in a mathematical sense ($w\in \mathbb{R}^r$), but it is not a spatial position on the bar. It is a coordinate in the abstract, whitened LIS space constructed in section 4: $w=W_{r}^T(\theta-\mu_{f})$. It is a coordinate along a direction in load-shape space. 
The output can be understood as: For sensor $j$, the GP is predicting $y_{j}$ (the $j$-th sensor's displacement reading as a function of $w$.) $\to$ "If the true load happened to correspond to this particular LIS coordinate $w$, what would sensor $j$ read?" 

**Section 6 as Orchestrator:** In part A, every posterior was computed using the true $G$, expensive in principle, but the whole point is to validate a method meant for cases where $G$ is genuinely expensive. So, the question this section tries to answer is: "If I only allow myself $n_{tr}$ calls to the true $G$, train a surrogate on those calls, and then build a posterior using only the surrogate, how close does the posterior get to the true one, and how does the closeness depend on $n_{tr}$ and on the reduced dimension $r$?"

The job of this section is that for every combination or $r\in\{1,2,3\}$ and n_train $\in \{5, 10, 20, 40, 80\}$, a training set can be generated and this will be then handed off to the function gp_lis_covw. 

**Training data generation walkthrough:** Drawing samples from the full prior is drawing plausible loading scenarios. This is not training data yet, it is just $n_{tr}$ hypothetical situations to try. For this hypothetical loads, its single LIS coordinate $w$ will be computed "via Wtr". Now, "via Ytr", for each of the same hypothetical loads, the actual physics will be run (multiplied by $G$) to get what the m sensors would really read for that load. Note this is a genuine call to the expensive forward model ($n_{tr}$ times). This is the budget being spent. 

Remark: 
	There exists $n_{tr}$ training $w_{i}$'s, that came from sampling hypothetical loads and computing their LIS coordinate. Having $r=1$, means that $w$ is a scalar, not a vector. 

**The candidate grid "Wg, dV, is1d"**:
lis_grid builds the list of candidate hypotheses for $w$ discussed in the last section. In the case of $r=1$, g= linspace is a straight line of 400 evenly spaced numbers from -Lw, Lw. It is called before the training loop begins. 

**What is done with the training data via (gp_lis_covw)**:
Now given the training table I have $n_{tr}$ pairs of $(w_{i},y_{i})$ per sensor, $m$ sensors in total. gp_lis_covw takes the column of Ytr belonging to sensor 1 (its $n_{tr}$ training readings) together with Wtr, and calls gp_fit_hyperparameters. This function will find the $(\ell, \sigma^2_{f},\sigma^2_{n})$ that best explain sensor 1's $n_{tr}$ points. Actually, training a GP on sensors m's n points, means exactly finding the hyperparameters that make those n specific $w_{i}, y_{i}$ pairs plausible. Once the hyperparameters are found, the GP is trained. 

Note that gp_predict is also called: Recall that 

$$
\bar{y}(w_{*})=k_{*}^T(K+\sigma^2_{n}I)^{-1}y_{tr}
$$

so the prediction at new point $w_{*}$ is a weighted average of the actual training outputs $y_{tr}$. The hyperparameters decide the weights. Once the weights are obtained, gp_predict is called (i.e the prediction step) returning the mu ($\bar{y}(w_{*})$) and vlat ($\sigma^2(w_{*})$).

**P4:** 
- deviation band: drawing m different random training sets makes (fit those m GPs -> build the grid -> get the predictions -> lift again to parameter space dimension -> measure Förstner metric). It is run m times, and get m different error numbers, the solid line is the mean (line), the shaded band is the mean +- 1 standard deviation of those m numbers 
- Bump: Each point on the curve corresponds to a *fresh, independent* random draw of $n_{tr}$ training points — the $n_{tr}=10$ training set is **not** the $n_{tr}=5$ set with 5 more points appended; it is drawn from scratch with a different random seed and shares no points with the $n_{tr}=5$ set (verified directly: zero overlap across multiple repetitions). Because of this, there is no guarantee that a larger budget outperforms a smaller one *at any single realization* — only that it does so *on average*, over many realizations.
  The small rise at $n_{tr}=10$ reflects this: an unlucky combination of
   1. how the 10 random points happened to be distributed in $w$-space that trial, and 
  2. the fact that `gp_fit_hyperparameters` is a non-convex optimization (`fminsearch`), which can converge to a mildly suboptimal fit depending on where the search starts and what data it sees. Averaging over $N_{rep}=8$ independent training sets (the shaded band) smooths most of this out but does not eliminate it entirely, especially at small $n_{tr}$ where each individual fit is least constrained.

**P5:**
**What is and isn't shown.** The dashed red line is the OLR floor **for $r^*=1$ only** — the best any rank-1 approximation could achieve using the exact forward model. Ranks 2 and 3 have their own, strictly lower floors (not drawn here), since Spantini's Theorem 2.3 guarantees a higher-rank reduction can only be as good or better than a lower-rank one. This plot therefore should **not** be read as "distance from each curve's own optimum" — it shows how a GP surrogate operating at each rank compares against a fixed, common reference point. **$r=1$ (green):** well-behaved, converges cleanly toward its own floor, matching the P4 discussion above. **$r=2$ (blue):** starts near $r=1$ at small $n_{tr}$ (a 2D coordinate needs more data to resolve than a 1D one, so the extra dimension is initially a handicap), overtakes $r=1$ by roughly $n_{tr}\approx15$–20, and continues improving through $n_{tr}=80$, ending roughly an order of magnitude below $r=1$. This is a genuine, reliable improvement: the second LIS direction carries real information ($\delta_2^2$ is small but non-negligible), and with enough training data to resolve it, retaining it pays off. **$r=3$ (purple):** the informative case. It is not monotonic and its band is visibly wider than the other two curves throughout the sweep — it can hit very low error at some training budgets (e.g. $4.3\times10^{-6}$ at $n_{tr}=40$, the best value on the entire plot) but rises again at $n_{tr}=80$, and dips only modestly at $n_{tr}=20$. Two compounding factors explain this: 1. the third LIS direction is very weakly informed ($\delta_3^2 \ll \delta_1^2$, per the Section 4 eigenvalue spectrum), so a GP trying to resolve it is largely fitting noise rather than signal in that dimension; and 2. the numerical integration grid used to extract the posterior is necessarily coarser at $r=3$ (19 points per axis vs. 400 for $r=1$), since a tensor grid's cost scales as (resolution)$^r$. The combination makes $r=3$ estimation intrinsically higher-variance at these training budgets — occasionally excellent, but not dependably so. **Practical conclusion.** Increasing rank strictly increases the *best achievable* posterior, but does not straightforwardly improve what a *finite-budget GP surrogate* actually delivers: each added dimension both dilutes limited training data across more directions and coarsens the grid-based inference step. In this problem, $r=2$ is a clear, reliable win over $r=1$; $r=3$ is not — it trades consistency for an occasional best-case result. This suggests the practical stopping point for the GP-surrogate pipeline is $r=2$, one rank above the numerically-detected $r^*=1$, rather than either the minimal informed rank or the maximum available rank.