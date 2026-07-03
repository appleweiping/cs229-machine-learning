# CS229 — key derivations behind the code

Short notes tying each implementation to the math it comes from. These are the
non-obvious identities that make the vectorised code correct; the full problem
statements live in the official CS229 problem sets.

## PS1

**1(b) Logistic regression (Newton's method).**
With `h = σ(Xθ)`, the averaged negative log-likelihood has gradient
`∇J = (1/m) Xᵀ(h − y)` and Hessian `H = (1/m) Xᵀ diag(h(1−h)) X`. Newton's
update is `θ ← θ − H⁻¹ ∇J`. We solve `H δ = ∇J` with `np.linalg.solve` rather
than forming `H⁻¹`.

**1(e) GDA.** MLE gives `φ = (1/m)Σ 1{y=1}`, class means `μ₀, μ₁`, and a shared
covariance `Σ = (1/m) Σ (xⁱ − μ_{yⁱ})(xⁱ − μ_{yⁱ})ᵀ`. The posterior is logistic
in `x`, with `θ₁: = Σ⁻¹(μ₁ − μ₀)` and
`θ₀ = ½(μ₀+μ₁)ᵀΣ⁻¹(μ₀−μ₁) − log((1−φ)/φ)`. GDA is more efficient when its
Gaussian assumption holds but is less robust than logistic regression when it
does not — visible in the ds1 vs ds2 accuracy gap.

**2 Positive-only labels.** If a positive is labelled (`y=1`) independently with
probability `α`, then `p(y=1|x) = α · p(t=1|x)`. Estimating
`α ≈ mean(h(x))` over labelled validation positives lets us recover
`p(t=1|x) ≈ h(x)/α`. The naive model collapses to 0.50 accuracy against the true
`t`; the corrected model recovers ≈0.95.

**3(d) Poisson regression.** A GLM with canonical (log) link: `E[y|x]=exp(θᵀx)`.
The log-likelihood gradient is `(1/m) Xᵀ(y − exp(Xθ))`, giving a clean batch
gradient-ascent update.

**5 Locally weighted regression.** Weight each training point for a query `x` by
`wⁱ = exp(−‖x − xⁱ‖² / 2τ²)` and solve the weighted normal equations
`(XᵀWX)θ = XᵀWy` per query. Small `τ` = low bias/high variance; the validation
sweep selects `τ = 0.05`.

## PS2

**5 Kernel perceptron.** Maintaining the hypothesis as `Σ β_j K(x_j, ·)` lets the
perceptron use a nonlinear (RBF) kernel. The dot-product kernel is linear and
cannot separate the ds5 data (≈0.53 test acc); the RBF kernel reaches ≈0.92.

**6 Naive Bayes (multinomial) for spam.** With Laplace smoothing,
`φ_{k|y} = (Σ_{i:yⁱ=y} x_kⁱ + 1)/(Σ_{i:yⁱ=y} Σ_l x_lⁱ + |V|)`. Prediction
compares class log-joints `Σ_k x_k log φ_{k|y} + log φ_y`. The most spam-
indicative tokens are ranked by `log(φ_{k|1}/φ_{k|0})`.

## PS3

**4 EM for GMMs.** E-step responsibilities
`w_{ij} ∝ φ_j N(xⁱ; μ_j, Σ_j)`; M-step re-estimates `φ, μ, Σ` as
responsibility-weighted moments. The data log-likelihood is monotonically
non-decreasing (part a), which the logs confirm. The semi-supervised variant
adds labelled terms with weight `α`, which removes the label-permutation
ambiguity and converges in a fraction of the iterations.

**5 k-means.** Lloyd's algorithm: assign each pixel to its nearest centroid,
then move each centroid to the mean of its members; distortion decreases
monotonically. Representing 512×512 pixels with 16 colours needs 4 bits/pixel
instead of 24, a 6× compression of the colour information.

## PS4

**4 ICA.** Under a Laplace source prior, the log-likelihood gradient in the
unmixing matrix `W` is `(Wᵀ)⁻¹ − sign(Wx) xᵀ` for a sample `x`; annealed
stochastic gradient ascent recovers `W` up to scale/permutation.

**6 Cart-pole control.** Model-based RL: estimate the MDP transition
probabilities and rewards from observed counts, then run value iteration
`V(s) ← R(s) + γ max_a Σ_{s'} P(s'|s,a) V(s')`. Acting greedily w.r.t. `V`
improves the balance time from ~11 steps to ~100 steps over ~110 failures.
