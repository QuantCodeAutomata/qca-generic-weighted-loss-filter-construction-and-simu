# RESULTS

Simon Donker van Heel and Neil Shephard (2026). *Filtering without recursion
and some of its uses in financial economics*. arXiv:2609.07207.
<https://arxiv.org/abs/2609.07207>

The replication has four experiments:

* **Experiment 1. Generic Weighted-Loss Filter Construction and Simulation Approximation** — the generic weighted-loss filter and its
  simulation and stratified estimators (Sections 2–3, Theorem 1);
* **Experiment 2. Synthetic Replication of Weighted Median Filters Under Heavy-Tailed Noise** — weighted medians under Cauchy noise
  (Sections 2.4 and 3.4, Figures 1–5, Theorem 2, Example 9);
* **Experiment 3. Empirical NVDA Replication: High-Frequency Robust Price Filtering and Volatility Estimation** — NVDA over the twelve sessions of 1–16 October
  2024: the five filters, Tables 1–3, Figures 6–13, the half-life choices
  and the diurnal spot volatility (Section 4);
* **Experiment 4. Empirical Robustness and Appendix Replication: Tail Index, Alternative Volatility Lags, Simulation Diagnostics, and Cross-Asset Extension** — the appendices: Table A.1, s = 10 seconds, the
  rolling ψ*, the first-minute startup rule, the simulated signatures, and
  NVDA with the 54 other assets of Tables B.1–B.2 (Appendices A and B).

All files named below are in `results/`: `expN_summary.json` and the other
`expN_*` files hold every number quoted here.

The paper does not publish its Monte Carlo errors, so the Experiment 1 and 2
tables are checks against the theorems. Experiments 3 and 4 use the
consolidated tape for October 2024, cleaned as in the footnote to
Section 4. On 16 October the raw NVDA count is 1,616,806 and the cleaned
length is **T = 1,521,339**, both identical to the paper. The opening print
is 1,857,756 shares at $134.01; the closing print is 15,146,434 shares at
$135.72, stamped at 16:00:00.

All empirical price filters use the paper's stratified estimator: the
exponential head up to `h = W_t^{-1}(0.999) = 29` (lags 0 to 29) is exact
and **B = 100** tail draws cover the remaining 0.1% of the weight. Seeds are
fixed in each driver. The damping factor `c_{t,s}` of Table 2 is simulated
as in Section 4.4 (M = 100 Brownian paths on the day's trade clock, a
30-lag window, each filter's own `c`); the multi-day signatures and the spot
volatility use the closed form of the 30-lag linear filter, which
Appendix A.6 puts within 1% of the Huber factor (checked below: the median
gap between the two simulated factors at s = 1 is 0.3% at most on NVDA,
MRVL and PEP).

---

## Experiment 1. Generic Weighted-Loss Filter Construction and Simulation Approximation

Mean absolute gap between the exact filter and the simulated filter
(Definition 3), T = 1,000:

| scenario                    | B=100  | B=500  | B=5,000 | B=50,000 |
|-----------------------------|--------|--------|---------|----------|
| absolute, exponential λ=0.9 | 0.1447 | 0.0697 | 0.02734 | 0.015997 |
| squared, uniform ℓ=15       | 1.1365 | 0.4921 | 0.15532 | 0.048582 |
| huber, hyperbolic a=10      | 0.2500 | 0.1132 | 0.03590 | 0.011050 |
| bernoulli, exponential λ=0.9| 0.0172 | 0.0080 | 0.00249 | 0.000760 |

Squared, Huber and Bernoulli losses fall at the `B^{-1/2}` rate. The median
falls faster than that at large B because the simulated median lands
exactly on the true median with probability tending to one (Theorem 2).

The objective gap `Q_t(θ*_t) − Q_t(θ_t)` (mean over t, always ≥ 0) and the
change in one-step predictive loss `(1/(T−1)) Σ_t L(θ_{t−1}, Y_t)`:

| scenario | B=100 | B=500 | B=5,000 | B=50,000 |
|---|---|---|---|---|
| absolute, objective gap | 9.5e-3 | 2.2e-3 | 4.8e-4 | 2.5e-4 |
| squared, objective gap | 6.37 | 1.14 | 0.124 | 9.9e-3 |
| huber, objective gap | 1.7e-2 | 3.7e-3 | 3.7e-4 | 3.5e-5 |
| bernoulli, objective gap | 2.8e-2 | 3.1e-3 | 1.5e-4 | 1.1e-5 |
| absolute, predictive-loss change (exact −4.241) | +0.023 | −0.001 | −0.001 | +4e-5 |
| huber, predictive-loss change (exact 6.580) | +0.021 | +5e-4 | +1.2e-3 | +4e-4 |
| bernoulli, predictive-loss change (exact 0.321) | +0.232 | +0.166 | +0.020 | −2e-4 |

The objective gap falls roughly as `1/B`, faster than the parameter error,
because it is second order in `θ* − θ`. The Bernoulli predictive loss at
small B is large because with 100 or 500 draws the filtered probability is
often exactly 0 or 1, and the clipping at `1e-12` turns a miss into a loss
of about 27.6. The squared-loss predictive change is noisy (the Cauchy
data have no mean); its objective gap is the cleaner measure.

**Theorem 1.** For squared loss and exponential weights (λ = 0.9, B = 500,
2,000 replications) the ratio `B · Var(θ*_t) / σ_t²` is 0.988 at t = 50,
0.982 at t = 200 and 0.959 at t = 1,000, with `σ_t² = Σ_j w_{t,j}(Y_{t−j} − θ_t)²`.
The Monte Carlo standard error of each ratio is about 0.03.

Stratified estimator (head `W⁻¹(0.999)`, B = 100) against the exact filter:

| loss     | mean \|θ − θ_strat\| |
|----------|----------------------|
| absolute | 1.15e-02             |
| squared  | 1.26e-03             |
| huber    | 1.08e-04             |

Runtime with O(1) lag draws by inverse CDF, exponential median, B = 100,
against the exact weighted median:

| T       | stratified, µs/point | simulated, µs/point | exact, µs/point |
|---------|----------------------|---------------------|-----------------|
| 1,000   | 46.4                 | 7.6                 | 48.5            |
| 3,000   | 45.7                 | 7.6                 | 117.2           |
| 10,000  | 45.5                 | 7.6                 | 412.3           |
| 30,000  | 45.5                 | 7.7                 | –               |
| 100,000 | 46.0                 | 7.7                 | –               |

The cost per point of both estimators is flat in T, the paper's O(T) claim
for a full sweep; the exact filter's cost per point grows linearly in T,
so its sweep is O(T²) (`results/exp1_runtime_scaling.png`).

---

## Experiment 2. Synthetic Replication of Weighted Median Filters Under Heavy-Tailed Noise

Parts A and B use the Hazen / type-5 median of Remark 3. Part A, T = 1,000,
average of 10 paths: tracking error is minimized at ℓ = 15 (0.487) and
predictive loss at ℓ = 13, inside the paper's range ℓ ∈ {8, …, 20}.

Part B uses Ex. 7(b): the anchor `(1−α)/T` sits on every observation of
`Y_{1:T}` next to the rolling window. Minimum tracking error is 0.536 at
ℓ = 8 for α = 0.6, 0.557 at ℓ = 10 for α = 0.8, 0.564 at ℓ = 6 for α = 0.4,
and 0.602 at ℓ = 11 for the pure local median (α = 1). At ℓ = 15 the values
are 0.562, 0.573, 0.603 and 0.606. Anchoring helps, and the curves flatten in
ℓ as α falls (the range of the tracking error over ℓ = 4…50 is 0.157 at
α = 0.4 against 0.175 at α = 1), as the paper says. The paper's text favours
the smallest α; here α = 0.6 has the lowest tracking error and α = 0.4 comes
third.

Part C reproduces Figures 4 and 5: the nonstationary Ex. 7(a) process and the
exponential median with λ = 0.9. Theorem 2 is stated for the lower median, so
the exact filter here is the lower median as well. The error table below uses
five paths of length T = 1,000.

| B      | mean \|θ* − θ\| | share θ* = θ | Theorem 2 probability |
|--------|-----------------|--------------|-----------------------|
| 500    | 0.0705          | 0.590        | 0.604                 |
| 5,000  | 0.0199          | 0.829        | 0.829                 |
| 50,000 | 0.0069          | 0.933        | 0.934                 |

Means over five paths. The share of exact hits matches the Theorem 2
probability `F_Bin(B, F̂(θ−))(⌈B/2⌉−1) − F_Bin(B, F̂(θ))(⌈B/2⌉−1)` averaged
over t. Figure 4 is drawn, as in the paper, on one path of length 5,000 with
B = 500. Its predictive loss (right panel) over λ = 0.80…0.95 is lowest at
λ = 0.91 (−3.511) and within 0.005 of that from 0.88 to 0.92, so λ = 0.9 "is
not poor", as the paper says. The level of the curve depends on the path:
the loss is measured against `|Y_t|`, which grows with the random-walk
signal. Figure 5 uses the first 1,000 points of the same path.

The stratified estimator with B = 100 is within 4e-5 of the
lower median (exact on all but a few time points); against a type-5 target
the error is 0.010, because type 5 interpolates using support points deep in
the tail that the head never sees.

Figure 3 (`exp2_C_figure3.png`) draws Theorem 2's probability of exact recovery
in the paper's stylized case, `F̂(θ−) = 0.5 − ε` and `F̂(θ) = 0.5 + ε`. At
B = 101, 1,001 and 10,001 it is 0.69, 0.998 and 1.00 for ε = 0.05; 0.31,
0.79 and 1.00 for ε = 0.02; 0.16, 0.47 and 0.95 for ε = 0.01: it "drives
upwards towards 1 quickly as B increases, even when ε is quite small".

Figures: `results/exp2_C_figure3.png`, `exp2_C_figure4.png`, `exp2_C_figure5.png`,
`exp2_C_convergence.png`.

Part D is Example 9 and Remark 3: the rolling median simulated with
B = 20,000 uniform lag draws on a path of length 1,000. With an odd window
(ℓ = 7, so eight values and no unique median) the simulated median under
the smallest-value convention equals the lower middle value at 47.7% of
the time points and the upper one at 52.3%, the paper's limit of one half
each. With an even window (ℓ = 8) it equals the median at every point.
Computing the Hazen / type-5 median from the simulated lag frequencies, as
the paper proposes, recovers the interpolated median: the mean error is
0.011 for ℓ = 7 and 0.010 for ℓ = 8.

---

## Experiment 3. Empirical NVDA Replication: High-Frequency Robust Price Filtering and Volatility Estimation

Full cleaned sessions, 1–16 October 2024. Startup of the noise scale uses
the previous five opening minutes. Tables 1–3 and the Section 4.4.1 checks
are for 16 October, as in the paper; the twelve-session results follow.

On 16 October the published values are: Hill **1.43**,
**16,314** outliers (**1.07%**), mean absolute Huber residual **0.38** cents,
δ about **1.5** cents after the open and **2.97** cents at the open. This run:
Hill **1.427**, **16,341** outliers (1.074%), mean absolute residual
**0.376** cents, open δ **3.01** cents, median δ **1.48** cents. **62.3%** of
median-filter residuals are exactly zero (the paper: more than half). The
linear filter is pulled to $130.98 by the $118.85 print against a Huber price
of $134.32, a gap of about **$3.35** (the paper: about $3.4).

### Table 1, 16 October — cents, rounded as in the paper

Mean \|Δ\|/√k, then root-mean-square / √k. Paper in parentheses when the
rounded value differs.

| k | median mad | Huber mad | linear mad | trades mad | Huber rms | trades rms |
|---|------------|-----------|------------|------------|-----------|------------|
| 1 | 0.116 (0.115) | 0.090 | 0.106 | 0.442 | 0.435 (0.431) | 2.995 |
| 2 | 0.115 (0.114) | 0.103 | 0.125 | 0.362 | 0.391 (0.389) | 2.159 |
| 5 | 0.136 | 0.123 | 0.150 | 0.281 | 0.366 | 1.405 |
| 10 | 0.151 | 0.139 | 0.161 | 0.241 | 0.348 | 1.018 |
| 25 | 0.163 (0.162) | 0.153 | 0.166 | 0.207 | 0.315 | 0.682 |

The Huber mean absolute change matches every published cell; the Huber rms
matches from k = 5 on. Lag-1 autocorrelation of the one-trade difference:
median −0.25 (paper −0.24), Huber −0.19 (−0.18), linear −0.02, trades −0.48.
At lags 2, 5, 10 and 25 all sixteen published ACF cells match.

Hill index of `|X̂_t − X̂_{t−k}|`, k = 1, 2, 5, 10, 25 (paper in parentheses):

| path | Hill |
|---|---|
| median | 1.69 1.61 1.49 1.41 1.49 (1.73 1.60 1.51 1.41 1.55) |
| Huber | 1.63 1.47 1.34 1.30 1.43 (1.64 1.48 1.36 1.31 1.48) |
| linear | 1.74 1.91 1.94 1.79 1.65 (1.74 1.91 1.94 1.79 1.67) |
| trades | 1.77 1.80 1.83 1.88 2.11 (1.77 1.80 1.83 1.89 2.10) |

### Table 2, 16 October — annualized volatility in percent

Each filter uses its own simulated `c_{t,s}` (M = 100 paths, σ from the Huber
15-minute level). Raw trades use `c = 1`.

| s | median | paper | Huber | paper | linear | paper | trades | paper |
|---|--------|-------|-------|-------|--------|-------|--------|-------|
| 0.1s | 123.3 | – | 54.1 | 53.3 | 117.4 | 117.5 | 245.2 | 245.2 |
| 1s | 34.8 | 34.8 | 34.4 | 34.3 | 42.9 | 43.1 | 77.9 | 77.9 |
| 5s | 34.4 | 34.4 | 34.3 | 34.3 | 35.9 | 35.9 | 44.5 | 44.5 |
| 10s | 34.2 | 34.2 | 34.1 | 34.1 | 35.1 | 35.1 | 46.6 | 46.6 |
| 1 min | 33.4 | 33.4 | 33.3 | 33.3 | 33.4 | 33.4 | 35.7 | 35.7 |
| 15 min | 37.1 | 37.1 | 37.1 | 37.1 | 37.1 | 37.1 | 37.2 | 37.2 |

This is the paper's headline result: the Huber signature is flat from one
minute down to one second (34.4% against 33.3%), while raw trades give
77.9% at one second. The correction raises the Huber estimate by 4.97% at
1 second and 0.49% at 10 seconds (paper: 4.9% and 0.5%). With the analytic
linear-filter factor in place of the simulated one, the cells are Huber
53.3 / 34.4, linear 117.8 / 43.1 and median 65.6 / 35.3 at 0.1 s / 1 s. The 0.1-second cells move by about 1.5% between
the two methods, the size of the 1/M ratio bias the paper quotes (1.8% at
M = 100). The median cell at 0.1 s is unstable, as the paper says.

The level panel in cents (Huber / paper): 2.34 / 2.34, 1.91 / 1.91,
1.94 / 1.94, 1.97 / 1.97, 1.93 / 1.93, 2.06 / 2.06. Adjacent-return
correlations: Huber −0.24, 0.07, 0.09, 0.04, 0.06; trades −0.81, −0.37,
−0.03, −0.12, −0.02. `X̂_{t,s}` falls back to the opening print for the first
`s` seconds. Hill row, Huber: 1.77, 3.33, 7.65, 8.49 (paper 1.81, 3.38, 7.63,
8.44).

### Table 3, 16 October — residuals `p_t − X̂_t` in cents

| | median | paper | Huber | paper | linear | paper |
|---|---|---|---|---|---|---|
| mean \|ε̂\| | 0.298 | .30 | 0.376 | .38 | 0.406 | .41 |
| s.d. | 2.178 | 2.18 | 2.158 | 2.16 | 1.839 | 1.84 |
| Hill of score | ∞ | ∞ | ∞ | ∞ | 1.74 | 1.74 |
| score ACF 1 / 2 / 5 / 10 / 25 | .19 .07 .01 .00 .00 | .20 .08 .02 .00 .00 | .28 .17 .06 .02 .00 | .28 .17 .06 .02 .00 | −.02 −.03 −.03 −.01 .00 | −.02 −.04 −.03 −.01 .00 |
| ψ* (cents) | – | – | 0.222 | .22 | 0.558 | (.56) |

ψ* uses the exponential head (lags 0 to 29) and the share of residuals inside
`δ_{t−1}` as the estimate of `E ψ'`.

The confidence intervals for `θ_t − X_t` come from
`P(θ_t − X_t > η) ≈ (1/T) Σ_t 1(g_t(η) > 0)`, `g_t(η) = Σ_j w_{t,j} ψ(ε̂_{t−j} − η)`
over lags 0 to 29 (the head `h = 29`), solved for η by bisection. Every published cell is
reproduced:

| level | Huber | paper | linear | paper |
|---|---|---|---|---|
| 80% | −.26, .26 | −.26, .26 | −.28, .28 | −.28, .28 |
| 90% | −.36, .36 | −.36, .36 | −.40, .40 | −.40, .40 |
| 95% | −.46, .46 | −.46, .46 | −.54, .54 | −.54, .54 |
| 99% | −.72, .75 | −.72, .75 | −1.23, 1.15 | −1.23, 1.15 |

### Sections 4.2, 4.4 and 4.4.1, 16 October

* Off-exchange prints (Massive exchange id 4, the FINRA trade reporting
  facilities) are **51.4%** of the trades (paper: around 51% on 16 October)
  and **93.8%** of the outliers on 16 October; pooled over the twelve
  sessions they are **91.2%** of the outliers (paper: around 91%).
* The largest Huber residual holds **35.3%** of `Σ ε̂²` and 0.29% of `Σ |ε̂|`
  (paper: 35% and 0.3%); the largest 1,000 hold 85.8% and 7.3% (85% and 7%).
* Without the two largest residuals the Hill index is still **1.43** (paper
  1.43). On 12 random subsamples of 25,000 trades it ranges from 1.25 to
  1.77 (paper: 1.33 to 1.49); other subsample schemes are under "What still
  differs".
* The $118.85 trade is **30.5%** of the summed squared one-trade
  differences of the linear filter (paper: 30%). The Huber filter's largest
  one-trade move is a $1.04 repricing at 09:45:01, **3.75%** of its sum
  (paper: 3.8%, at 09:45).

### Direction of outliers, Figures 12 and 13

Outliers are trades with `|Y_t − θ_t| > δ_{t−1}`. In 2-minute bins the share
of outliers above the band falls as the Huber price rises: the correlation
between the two is **−0.47** on 16 October (−0.40 on 14 October, −0.53 on
15 October). Means over 10-cent groups trace the rotated S of Figure 12:
0.72 at −30 cents, 0.39 at 0 to +10 cents, 0.29 at +10 to +30 cents. The
clock half-life for `q_t` chosen by predictive Bernoulli log-likelihood over
a grid from 2 to 120 seconds is **13 seconds** on 16 October, the paper's
value. Across the twelve sessions it is 13 s on seven days and between 6
and 16 s on the others.

Figures: `results/exp3_2024-10-16_figure12.png`, `exp3_2024-10-16_figure13.png`.

### Twelve sessions

| date | T | Hill | outliers % | δ open (c) | Huber / linear / trades, 1 s | Huber, 15 min | q half-life (s) | direction corr | off-exchange: trades / outliers (%) |
|---|---|---|---|---|---|---|---|---|---|
| 10-01 | 1,929,784 | 1.44 | 1.06 | 2.96 | 43.9 / 56.2 / 99.6 | 44.0 | 10 | −0.51 | 44.0 / 91.0 |
| 10-02 | 1,196,097 | 1.80 | 1.29 | 2.83 | 34.3 / 37.5 / 57.6 | 34.9 | 13 | −0.44 | 51.3 / 92.4 |
| 10-03 | 1,618,606 | 1.48 | 1.15 | 2.95 | 37.4 / 43.6 / 69.3 | 37.0 | 13 | −0.46 | 46.9 / 90.0 |
| 10-04 | 1,396,786 | 1.41 | 1.02 | 3.12 | 30.5 / 33.2 / 49.7 | 28.3 | 16 | −0.41 | 48.5 / 89.6 |
| 10-07 | 2,157,651 | 1.22 | 0.98 | 3.12 | 36.9 / 49.5 / 96.2 | 47.1 | 10 | −0.52 | 46.4 / 91.7 |
| 10-08 | 1,729,866 | 1.61 | 1.04 | 3.17 | 31.9 / 40.5 / 87.7 | 26.4 | 13 | −0.49 | 50.2 / 91.8 |
| 10-09 | 1,456,982 | 1.62 | 1.28 | 3.19 | 33.1 / 60.0 / 131.6 | 34.7 | 13 | −0.53 | 51.8 / 93.9 |
| 10-10 | 1,430,948 | 1.67 | 1.16 | 3.13 | 34.6 / 41.5 / 69.1 | 28.6 | 13 | −0.54 | 51.3 / 91.8 |
| 10-11 | 1,132,845 | 1.63 | 1.13 | 3.17 | 25.3 / 27.7 / 41.9 | 20.6 | 13 | −0.36 | 51.6 / 93.5 |
| 10-14 | 1,496,150 | 1.58 | 1.32 | 3.19 | 26.3 / 45.5 / 124.0 | 33.2 | 10 | −0.40 | 52.3 / 90.3 |
| 10-15 | 2,484,638 | 1.46 | 1.46 | 3.14 | 47.8 / 51.0 / 74.9 | 57.4 | 6 | −0.53 | 43.5 / 88.6 |
| 10-16 | 1,521,339 | 1.43 | 1.07 | 3.01 | 34.4 / 42.9 / 77.9 | 37.1 | 13 | −0.47 | 51.4 / 93.8 |

The Hill index of the Huber residual is below 2 on every day (1.22 to
1.80; the paper's Table A.1: 1.23 to 1.81). The stale-trade correlation is
negative on every day, from −0.36 to −0.54 ("the pattern holds on each of
the 12 days"). At 15 minutes the Huber volatility ranges from 20.6% to
57.4% across the days (the paper: "between 20 and 57%").

### Thresholds and price paths, Figures 6–8

Figure 7 (`exp3_figure7_threshold_12days.png`): the twelve-day median of δ_t is
3.11 cents over the first minute and 1.47 cents after the first hour; each
day's median after the first ten minutes is 1.43 to 1.48 cents ("δ_t is
approximately 1.5 cents for most of the day but starts off higher, at
around 3 cents near the open"). Figure 6 (`exp3_2024-10-16_figure6.png`): with
half-lives of 30 seconds, 4 minutes and 15 minutes, all three start at
3.01 cents (the paper: 2.97); the 30-second scale moves 8.0 times as much
as the 4-minute one (the paper: about nine times) and the 15-minute scale
0.49 times. Figure 8 (`exp3_2024-10-16_figure8.png`) draws the full day and the
three 5-second windows of the paper; at the $118.85 trade the linear
filter is pulled about $3.35 down and recovers within half a second, the
Huber filter does not move.

### Half-life selection, Sections 4.3 and 4.4

`exp3_halflife_selection.png`. Pooled over the twelve sessions:

* **Noise scale.** The check loss of `b_{t,60}` for the nonzero `|ε̃_t|` is
  smallest at a half-life of **2 minutes** and within 0.65% of that from
  30 seconds to 15 minutes (relative losses 1.0010, 1.0003, 1, 1.0008,
  1.0031, 1.0065 at 30 s, 1, 2, 4, 8, 15 min). The paper: "smallest at
  about 2 minutes & almost flat, within 1% of its smallest value from 30
  seconds to 15 minutes". The paper then uses 4 minutes, as here.
* **Price filter.** The k-trade-ahead Huber forecast loss
  `Σ H_{δ_{t−k}}(Y_t − θ_{t−k})`, thresholds held at the day's path, is
  smallest at a half-life of **2 trades** for every k = 2, 5, 10, 25, 50,
  100 on the grid 1.5, 2, 2.5, 3, 4, 6, with 2.5 the runner-up for k ≥ 5.
  The loss at 3 trades is 2.4%, 1.2%, 0.8%, 0.3%, 0.2% and 0.1% above the
  loss at 2. The paper: "slightly more than 2 trades for every k from 2 to
  100, and the loss changes by less than 2.5% between half-lives of 2 and
  3". The paper then uses 3 trades, as here.

### Diurnal spot volatility, Sections 4.5.2–4.6

`exp3_spot_summary.json`. The diurnal curve is a quadratic plus four Fourier
pairs fit to the log of the previous ten sessions' 10-second uniform
variances of Huber prices. Spot variance uses Mandelbrot weights `1/(j + a)`
in trade time on the diurnally adjusted 1-second increments; the first 15
seconds blend towards the median first-minute variance of the previous five
sessions. QLIKE of `σ̂²(u − 60; a)` against the 10-second uniform estimate
`σ̄²(u)`, u from 75 s to the close, summed over the twelve sessions, is
minimized at **a = 566 trades, 8.1 seconds** (the paper: 563 trades,
roughly 8 seconds).

* **Figure 11** (`exp3_figure11_huber_linear.png`). The linear-based path lies
  23.2% above the Huber-based one at the typical trade: the median linear
  spot volatility divided by the median Huber one, on every 300th trade of
  the twelve days (the paper: "about a quarter above"). At the $118.85
  print it goes from 23.4% to 269% (the paper: about 23% to about 273%) and
  fades over the following half hour; the Huber path stays near 20%.
* **Figure 10** (`exp3_figure10_hyperbolic_hepx.png`). The Example 5
  superposition with half-lives of 2, 20, 200 and 2,000 seconds, converted
  to trades at NVDA's 69.6 trades per second; QLIKE chooses the mixture
  0.00, 0.24, 0.47, 0.30. Its log-path correlation with the hyperbolic
  filter is **0.998** (the paper: 0.999). The same half-lives in clock time
  correlate 0.994.
* **Figure 9** (`exp3_figure9_signature.png`). Twelve-day signatures (square
  root of the mean of squared daily curves): Huber 50.7, 35.2, 34.5 and
  37.0 at 0.1 s, 1 s, 1 min and 15 min; linear 115.5, 44.8, 34.5, 37.0;
  trades 218.0, 85.8, 35.8, 37.0. Raw trades over Huber at 0.1 s range from
  2.5 to 7.0 across the days (the paper: "between about 2.5 & 7 times").

---

## Experiment 4. Empirical Robustness and Appendix Replication: Tail Index, Alternative Volatility Lags, Simulation Diagnostics, and Cross-Asset Extension

### A. Table A.1, Hill index of the Huber residual

| screen | 16 Oct, this run | paper | 12-day range, this run | paper |
|--------|------------------|-------|------------------------|-------|
| all trades | 1.43 | 1.43 | 1.22 to 1.80 | 1.23 to 1.81 |
| residual / δ | 1.51 | 1.51 | 1.30 to 1.85 | 1.31 to 1.85 |
| repricing windows removed | 1.44 | 1.43 | 1.16 to 1.64 | 1.14 to 1.64 |
| every 10th trade | 1.40 | 1.41 | 1.26 to 1.81 | 1.26 to 1.78 |
| six-trade half-life | 1.47 | 1.47 | 1.28 to 1.92 | 1.30 to 1.92 |

The repricing screen drops trades whose filtered price ranges over more than
10 cents within ±1 second of the trade: 9.3% of 16 October, inside the
paper's 7% to 25%. Every screen leaves the index below 2, so the noise
variance is infinite under each of them.

### B. s = 10 seconds, Appendix A.2–A.3

The spot volatility of Figure 11 rebuilt from `s = 10` second returns, with
the same `a`, the same diurnal curve and each lag's own damping factor and
opening background (`results/exp4_summary.json`, `B.spot`). A ratio at the
typical trade is the median of one path divided by the median of the other,
on every 300th trade of the twelve days (Table B.2).

| | replication | paper |
|---|---|---|
| linear over Huber, s = 1 | 1.232 | about 1.23 |
| linear over Huber, s = 10 | 1.023 | about 1.025 |
| Huber s = 10 over Huber s = 1 | 0.988 | about 0.99 |
| linear path at the $118.85 trade, s = 10 | 19.8% to 85.0% | about 20% to 85% |

The medians of the trade-by-trade ratios are lower (1.091, 1.004 and 0.970),
because the two paths differ most where volatility is high. Figures:
`exp4_B_figure_a1_s10.png` (Figure A.1) and `exp4_B_figure_a2_huber_s1_s10.png`
(Figure A.2).

The integrated volatility on 16 October with the closed-form factor in place
of the simulated one is, Huber, 34.4 at 1 s and 34.1 at 10 s; linear 43.1
and 35.1; raw trades (no factor) 77.9 and 46.6. The Huber 1-second cell
matches Table 2 (34.4). The linear 1-second cell is the annualized volatility
of the linear filter: 42.9 with the simulated factor (Table 2) and 43.1 with
the closed form (the paper: 43.1).

### C. Time-varying ψ*_t, Appendix A.4

The Table 3 standard error on windows of the most recent 100,000 trades,
moved forward 1,000 trades at a time (1,422 windows on 16 October,
`exp4_C_figure_a3_psi_rolling.png`). It is highest at the open, 0.37 cents for
the first window (ending at 09:37:54), averages 0.28 cents over the first
hour and 0.155 cents over the second half of the day; the full-day value is
0.22 cents. The second-half level is **0.55** times the first-hour level;
the median threshold δ_t falls by **0.54** over the same stretch. The
paper: "in the second half of the day the standard error is around one half
the value seen in the first hour … roughly the same as the change in values
of the Huber threshold δ_t".

### D. First-minute startup rule, Appendix A.5

The presample of the previous five openings is swapped out by the clock
(the paper's rule, masses `(60 − τ_t)/60` and `τ_t/60`) or by trade count
(`(T60 − t)/T60` and `t/T60`), on the six stocks of Figure B.1 and the twelve
sessions; both rules share the seeds, so only the scale differs. For each
stock the panel of `exp4_D_figure_a4_startup.png` is the day with the largest
gap between the two thresholds.

| ticker | trades/s | day | δ0 (cents) | largest δ gap (cents) | outliers in the first minute, clock / count | largest Huber gap, any day (cents) |
|---|---|---|---|---|---|---|
| NVDA | 70 | 10-11 | 3.17 | 1.59 | 971 / 850 | 1.75 |
| AAPL | 24 | 10-03 | 3.52 | 3.03 | 495 / 475 | 3.01 |
| INTC | 8.9 | 10-01 | 2.71 | 0.33 | 91 / 88 | 0.27 |
| COST | 3 | 10-02 | 60.23 | 24.76 | 96 / 78 | 15.84 |
| CROX | 0.87 | 10-16 | 29.97 | 42.47 | 8 / 5 | 15.59 |
| IPAR | 0.19 | 10-14 | 56.19 | 1185.59 | 4 / 2 | 106.93 |

On every stock and day the two rules give identical thresholds, outliers
and prices after the first minute, as the paper says ("after the first
minute the two coincide, so nothing later in the day depends on the
choice"). By trade count δ_t jumps in the opening seconds and falls back;
by the clock it moves gradually from δ0 to the day's own level. NVDA on
11 October and INTC on 1 October are also the paper's panels; for AAPL,
COST, CROX and IPAR the paper shows 10, 9, 14 and 10 October; on those days
the gap here is 1.43, 14.8, 11.5 and 121 cents, the 10th, 9th, 10th and 3rd
largest of the twelve. On the slowest stocks
the trade-count rule puts a weight of `1/T60` on each of a handful of
opening residuals, which drives IPAR's threshold above $10 for a few
seconds on 14 October.

### E. Simulated signatures, Appendix A.7 and Table A.2

The four cases of Table A.2 on the 16 October trade times, 100 paths each:
Brownian or variance-gamma prices with 40% annual volatility, with or without
Student-t noise (1.2 degrees of freedom, scale 0.24 cents at the opening
price), added to the same paths. The variance-gamma price has increments
`U_t sqrt(Z(τ_t) − Z(τ_{t−1}))` with gamma increments of shape `σ²β Δτ` and
rate β = 1 (price in percent). Every filtered path is the full pipeline at
B = 100 (median filter, scale filter, Huber filter, linear filter);
`c_{t,s}` of each filter is estimated on the noise-free Brownian paths and
used in all four cases. Figures: `exp4_E_figure_a5_brownian.png` (Figure A.5),
`exp4_E_figure_a6_variance_gamma.png` (Figure A.6).

Huber signature over each path's own truth (c = 1), and with `c_{t,s}`
minus the truth, in annualized percentage points:

| case | c = 1: 0.1 s / 1 s / 10 s | with c: 0.1 s / 1 s / 10 s | median δ (cents) | lag-1 ACF, Huber |
|---|---|---|---|---|
| Brownian | 0.625 / 0.928 / 0.993 | 0.0 / 0.0 / 0.0 | 0.15 | +0.36 |
| Brownian + noise | 0.615 / 0.927 / 0.998 | +29.6 / −0.1 / +0.2 | 1.00 | +0.19 |
| variance gamma | 0.781 / 0.965 / 0.995 | +23.5 / +2.0 / +0.1 | 0.00 | 0.00 |
| variance gamma + noise | 0.830 / 0.974 / 0.996 | +26.2 / +2.3 / +0.1 | 0.74 | −0.02 |

Against the paper:

* Brownian, c = 1: 37.5% low at 0.1 s and 7% at 1 s (the paper: 38% and
  7%). The correction estimated on the same paths removes the bias at every
  lag, so the corrected Huber, median and linear signatures coincide with
  the simulated price (the paper's red and purple lines).
* Brownian with noise, c = 1: 38.5% low at 0.1 s (the paper: 39%, "barely
  responds to the noise"). The scale filter raises δ_t from 0.15 to 1.00
  cents (the paper: 0.16 to 1.00). With the correction the estimate is
  30 points above the truth at 0.1 s and on it from 1 s: the noise the
  filter leaves in the price.
* Variance gamma, c = 1: 3.5% low at 1 s and within 0.7% from 5 to 10 s
  (the paper: 3.3% low against 7.4% for the Brownian price, within 0.6%).
  At 0.1 s the filter keeps 78% of the estimate against 62% for the
  Brownian price (the paper: 80% and 62%). δ_t is in effect zero, so the
  Huber filter is a weighted median. About 109 of the 1.5 million
  increments exceed 0.01 cents, and on a typical path 20 of them hold 99%
  of the quadratic variation and 2 hold half (the paper: roughly 100, 20
  and 2).
* Variance gamma with the Brownian correction: **above** the truth at every
  short lag, by 23.5 points at 0.1 s and 2.0 at 1 s. The paper says the
  corrected estimator is "below the truth at the shortest s and above it
  just beyond". See "What still differs".
* Variance gamma with noise: far above the truth at the shortest s with the
  correction (+26 points at 0.1 s), as the paper says.
* Lag-1 ACF of the one-trade Huber price change: +0.36 on the noise-free
  Brownian price and 0.00 on the noise-free variance-gamma price (the paper:
  +0.39 and 0.00); the noise turns it to +0.19 and −0.02.
* Linear and median filters (beyond the paper). Without noise both behave
  like the Huber filter (linear 46% low at 0.1 s with c = 1, median 34%
  low). With noise the linear signature is useless at every lag: corrected,
  it is 1,572% at 1 s and 204% at one minute on the Brownian price, while
  raw noisy prices give 8,114% at 0.1 s and 452% at one minute. The median
  filter stays close to the Huber filter from 1 s on.

### F. Cross-asset, Appendix B

NVDA and the 54 other assets of Table B.2 over the twelve sessions, each
session filtered once (`src/spot_pipeline.py`) with its own five-opening
presample, and each asset's own QLIKE `a`. `exp4_F_table_b2.json` has every row;
figures `exp4_F_figure_b1_signatures.png` (Figure B.1, the paper's six stocks
NVDA, AAPL, INTC, COST, CROX and IPAR) and `exp4_F_figure_b2_cross_section.png`
(Figure B.2, the six statistics against the trading rate, replication and
paper).

#### Table B.2, replication / paper

The definitions are the paper's (see below). The jump is measured around the
largest absolute Huber residual among trades at least 15 minutes after the
open; on 15 slow names the day's largest residual is in the first few
seconds, where the baseline window "15 to 3 minutes before" does not exist.

| ticker | trades/s | outlier % | Hill | ψ* cents | ψ* bp | linear/Huber | sig 60s/1s | linear jump | ACF 1 | ACF 2 | a (trades) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NVDA | 69.63 / 69.63 | 1.07 / 1.07 | 1.43 / 1.43 | 0.22 / 0.22 | 0.16 / 0.16 | 1.232 / 1.223 | 0.98 / 0.98 | 11.5 / 11.9 | −0.08 / −0.08 | 0.03 / 0.03 | 566 |
| TSLA | 44.40 / 44.40 | 2.40 / 2.43 | 1.84 / 1.83 | 0.61 / 0.61 | 0.28 / 0.28 | 1.083 / 1.074 | 1.02 / 1.02 | 1.2 / 1.3 | −0.03 / −0.02 | 0.05 / 0.05 | 444 |
| AAPL | 23.94 / 23.94 | 1.31 / 1.30 | 2.05 / 2.10 | 0.29 / 0.29 | 0.13 / 0.13 | 1.240 / 1.211 | 0.91 / 0.91 | 3.7 / 4.6 | −0.09 / −0.09 | 0.05 / 0.05 | 178 |
| SPY | 17.06 / 17.06 | 0.93 / 0.92 | 1.62 / 1.63 | 0.35 / 0.35 | 0.06 / 0.06 | 1.396 / 1.322 | 0.99 / 0.99 | 146.6 / 137.6 | 0.24 / 0.25 | 0.19 / 0.20 | 71 |
| AMD | 17.00 / 17.00 | 1.62 / 1.60 | 1.83 / 1.85 | 0.47 / 0.47 | 0.30 / 0.30 | 1.226 / 1.171 | 0.94 / 0.94 | 5.5 / 4.1 | −0.05 / −0.06 | 0.06 / 0.06 | 144 |
| MSFT | 14.64 / 14.64 | 2.15 / 2.15 | 1.73 / 1.68 | 0.70 / 0.70 | 0.17 / 0.17 | 1.278 / 1.236 | 0.90 / 0.89 | 20.5 / 16.2 | 0.01 / 0.01 | 0.06 / 0.07 | 97 |
| AMZN | 15.60 / 15.60 | 2.01 / 2.00 | 2.01 / 2.03 | 0.32 / 0.32 | 0.17 / 0.17 | 1.301 / 1.254 | 0.92 / 0.92 | 2.3 / 2.7 | −0.07 / −0.07 | 0.03 / 0.04 | 192 |
| AVGO | 10.14 / 10.14 | 3.01 / 3.01 | 1.89 / 1.90 | 0.68 / 0.68 | 0.38 / 0.38 | 1.377 / 1.241 | 0.84 / 0.83 | 8.6 / 9.7 | −0.03 / −0.03 | 0.06 / 0.06 | 123 |
| MU | 9.19 / 9.19 | 1.61 / 1.61 | 1.99 / 2.01 | 0.42 / 0.42 | 0.38 / 0.38 | 1.352 / 1.262 | 0.89 / 0.89 | 9.2 / 13.0 | −0.01 / −0.01 | 0.06 / 0.06 | 128 |
| INTC | 8.94 / 8.94 | 0.96 / 0.96 | 2.16 / 2.19 | 0.12 / 0.12 | 0.52 / 0.52 | 1.190 / 1.143 | 0.71 / 0.71 | 3.2 / 2.6 | 0.08 / 0.08 | 0.03 / 0.03 | 78 |
| META | 8.74 / 8.74 | 4.32 / 4.33 | 2.53 / 2.51 | 1.64 / 1.65 | 0.28 / 0.29 | 1.147 / 1.113 | 0.72 / 0.72 | 3.8 / 3.1 | 0.00 / 0.01 | 0.02 / 0.02 | 94 |
| GOOG | 8.25 / 8.25 | 0.93 / 0.93 | 1.64 / 1.65 | 0.36 / 0.36 | 0.21 / 0.21 | 1.281 / 1.171 | 0.95 / 0.94 | 5.0 / 7.4 | 0.02 / 0.02 | 0.10 / 0.10 | 30000 |
| COIN | 5.81 / 5.81 | 5.02 / 5.03 | 2.67 / 2.67 | 1.74 / 1.74 | 0.83 / 0.83 | 1.140 / 1.089 | 0.76 / 0.74 | 1.4 / 1.1 | 0.08 / 0.10 | 0.04 / 0.04 | 81 |
| MRVL | 5.39 / 5.39 | 2.12 / 2.10 | 2.38 / 2.40 | 0.43 / 0.43 | 0.52 / 0.52 | 1.440 / 1.268 | 0.80 / 0.78 | 10.4 / 12.3 | 0.07 / 0.08 | 0.07 / 0.08 | 96 |
| RIVN | 4.62 / 4.62 | 0.68 / 0.69 | 4.65 / 4.70 | 0.10 / 0.10 | 0.94 / 0.94 | 1.104 / 1.118 | 0.67 / 0.67 | 1.8 / 1.9 | 0.11 / 0.11 | 0.04 / 0.04 | 30 |
| HOOD | 4.47 / 4.47 | 1.30 / 1.30 | 2.54 / 2.51 | 0.19 / 0.19 | 0.72 / 0.72 | 1.272 / 1.183 | 0.73 / 0.72 | 3.8 / 1.8 | 0.07 / 0.09 | 0.07 / 0.07 | 27 |
| CELH | 4.43 / 4.43 | 2.18 / 2.17 | 2.28 / 2.28 | 0.45 / 0.45 | 1.33 / 1.33 | 1.395 / 1.235 | 0.71 / 0.71 | 22.6 / 11.4 | 0.07 / 0.07 | 0.07 / 0.07 | 34 |
| QCOM | 4.33 / 4.33 | 3.88 / 3.89 | 2.01 / 2.00 | 0.98 / 0.98 | 0.57 / 0.57 | 1.238 / 1.148 | 0.75 / 0.73 | 2.9 / 2.1 | 0.09 / 0.09 | 0.07 / 0.07 | 33 |
| PYPL | 4.30 / 4.30 | 3.03 / 3.02 | 2.37 / 2.38 | 0.26 / 0.27 | 0.33 / 0.33 | 1.388 / 1.255 | 0.90 / 0.88 | 3.1 / 1.5 | 0.05 / 0.04 | 0.04 / 0.04 | 53 |
| CSCO | 3.96 / 3.96 | 1.37 / 1.37 | 1.92 / 1.91 | 0.17 / 0.17 | 0.30 / 0.30 | 1.367 / 1.277 | 0.67 / 0.67 | 22.1 / 17.2 | 0.12 / 0.13 | 0.05 / 0.05 | 30 |
| PEP | 3.96 / 3.96 | 2.13 / 2.11 | 2.74 / 2.73 | 0.48 / 0.48 | 0.27 / 0.27 | 1.436 / 1.318 | 0.69 / 0.68 | 22.1 / 13.4 | 0.04 / 0.04 | 0.04 / 0.07 | 85 |
| AMAT | 3.93 / 3.93 | 3.78 / 3.77 | 1.81 / 1.80 | 0.91 / 0.91 | 0.49 / 0.49 | 1.378 / 1.191 | 0.74 / 0.72 | 36.2 / 25.5 | 0.13 / 0.13 | 0.09 / 0.09 | 33 |
| SBUX | 3.71 / 3.71 | 1.62 / 1.61 | 2.19 / 2.19 | 0.34 / 0.34 | 0.36 / 0.36 | 1.283 / 1.172 | 0.75 / 0.75 | 10.4 / 6.7 | 0.11 / 0.13 | 0.06 / 0.07 | 21 |
| NFLX | 3.71 / 3.71 | 5.15 / 5.15 | 3.42 / 3.40 | 3.30 / 3.31 | 0.47 / 0.47 | 1.112 / 1.072 | 0.54 / 0.54 | 1.4 / 1.6 | 0.09 / 0.10 | 0.01 / 0.01 | 131 |
| CRWD | 3.41 / 3.41 | 5.36 / 5.36 | 2.91 / 2.91 | 2.17 / 2.17 | 0.71 / 0.71 | 1.158 / 1.102 | 0.62 / 0.62 | 1.1 / 1.2 | 0.08 / 0.08 | 0.03 / 0.03 | 84 |
| ADBE | 3.10 / 3.10 | 4.83 / 4.89 | 3.03 / 3.05 | 2.86 / 2.86 | 0.57 / 0.57 | 1.156 / 1.115 | 0.51 / 0.51 | 1.1 / 1.4 | 0.06 / 0.06 | 0.02 / 0.02 | 70 |
| COST | 3.02 / 3.02 | 5.55 / 5.60 | 3.71 / 3.70 | 3.82 / 3.82 | 0.43 / 0.43 | 1.023 / 1.016 | 0.41 / 0.41 | 1.3 / 1.4 | 0.04 / 0.04 | −0.01 / −0.01 | 36 |
| DKNG | 2.75 / 2.75 | 2.02 / 2.01 | 3.15 / 3.13 | 0.25 / 0.25 | 0.67 / 0.67 | 1.265 / 1.189 | 0.70 / 0.70 | 1.8 / 1.1 | 0.08 / 0.09 | 0.09 / 0.09 | 18 |
| TXN | 2.69 / 2.69 | 4.02 / 4.00 | 2.52 / 2.51 | 1.08 / 1.09 | 0.54 / 0.54 | 1.243 / 1.137 | 0.71 / 0.70 | 1.3 / 1.5 | 0.16 / 0.16 | 0.12 / 0.12 | 27 |
| PANW | 2.38 / 2.38 | 4.91 / 4.95 | 3.14 / 3.14 | 2.69 / 2.69 | 0.72 / 0.72 | 1.162 / 1.133 | 0.59 / 0.59 | 1.3 / 1.6 | 0.07 / 0.07 | 0.02 / 0.02 | 52 |
| ABNB | 2.33 / 2.33 | 3.49 / 3.48 | 2.05 / 2.03 | 0.84 / 0.84 | 0.62 / 0.62 | 1.213 / 1.149 | 0.71 / 0.69 | 9.7 / 8.4 | 0.10 / 0.10 | 0.07 / 0.07 | 22 |
| GILD | 2.25 / 2.25 | 1.67 / 1.67 | 3.07 / 3.07 | 0.32 / 0.32 | 0.36 / 0.36 | 1.266 / 1.201 | 0.75 / 0.75 | 2.8 / 2.5 | 0.15 / 0.15 | 0.11 / 0.12 | 13 |
| DDOG | 2.11 / 2.11 | 2.87 / 2.86 | 3.20 / 3.18 | 0.82 / 0.82 | 0.65 / 0.65 | 1.285 / 1.171 | 0.72 / 0.70 | 1.3 / 1.4 | 0.16 / 0.15 | 0.09 / 0.09 | 42 |
| ETSY | 1.88 / 1.88 | 2.64 / 2.65 | 3.04 / 2.98 | 0.46 / 0.46 | 0.87 / 0.87 | 1.142 / 1.099 | 0.64 / 0.64 | 3.6 / 3.2 | 0.12 / 0.12 | 0.07 / 0.07 | 13 |
| LCID | 1.87 / 1.87 | 0.03 / 0.03 | 5.65 / 5.60 | 0.08 / 0.08 | 2.37 / 2.37 | 1.002 / 1.004 | 0.40 / 0.40 | 1.1 / 1.1 | 0.11 / 0.11 | 0.00 / 0.00 | 40 |
| TTD | 1.61 / 1.61 | 3.31 / 3.32 | 2.77 / 2.78 | 0.82 / 0.82 | 0.69 / 0.69 | 1.184 / 1.123 | 0.69 / 0.69 | 2.7 / 3.2 | 0.15 / 0.15 | 0.09 / 0.09 | 14 |
| ROKU | 1.51 / 1.51 | 3.44 / 3.42 | 2.72 / 2.63 | 0.81 / 0.81 | 1.04 / 1.05 | 1.180 / 1.113 | 0.68 / 0.68 | 1.3 / 1.0 | 0.13 / 0.13 | 0.07 / 0.07 | 24 |
| TEAM | 1.49 / 1.49 | 4.82 / 4.86 | 3.00 / 2.91 | 1.95 / 1.95 | 1.03 / 1.03 | 1.219 / 1.154 | 0.71 / 0.68 | 2.9 / 2.2 | 0.12 / 0.11 | 0.06 / 0.06 | 9 |
| INTU | 1.43 / 1.43 | 5.18 / 5.13 | 2.50 / 2.49 | 4.12 / 4.12 | 0.68 / 0.68 | 1.139 / 1.115 | 0.58 / 0.58 | 15.4 / 9.3 | 0.13 / 0.13 | 0.04 / 0.04 | 20 |
| AMGN | 1.38 / 1.38 | 4.11 / 4.16 | 3.05 / 3.04 | 1.82 / 1.82 | 0.57 / 0.57 | 1.054 / 1.023 | 0.52 / 0.52 | 3.2 / 3.9 | 0.13 / 0.13 | 0.05 / 0.05 | 17 |
| ISRG | 1.28 / 1.28 | 4.76 / 4.79 | 2.31 / 2.30 | 3.18 / 3.19 | 0.67 / 0.67 | 1.073 / 1.049 | 0.48 / 0.48 | 3.4 / 2.5 | 0.10 / 0.10 | 0.02 / 0.02 | 16 |
| VRTX | 1.08 / 1.08 | 5.12 / 5.11 | 3.78 / 3.61 | 3.95 / 3.95 | 0.81 / 0.81 | 1.075 / 1.044 | 0.53 / 0.53 | 2.0 / 1.8 | 0.14 / 0.14 | 0.05 / 0.05 | 21 |
| REGN | 0.91 / 0.91 | 5.73 / 5.78 | 2.21 / 2.25 | 10.68 / 10.69 | 1.06 / 1.06 | 1.082 / 1.036 | 0.53 / 0.53 | 1.8 / 1.7 | 0.10 / 0.10 | 0.03 / 0.03 | 14 |
| CROX | 0.87 / 0.87 | 4.92 / 4.91 | 4.40 / 4.18 | 1.35 / 1.35 | 0.98 / 0.98 | 1.120 / 1.052 | 0.58 / 0.58 | 1.8 / 1.4 | 0.12 / 0.12 | 0.05 / 0.06 | 10 |
| MELI | 0.83 / 0.83 | 6.91 / 6.90 | 3.94 / 3.71 | 32.23 / 32.27 | 1.58 / 1.58 | 1.023 / 1.004 | 0.38 / 0.38 | 2.8 / 2.2 | 0.01 / 0.01 | −0.01 / −0.01 | 14 |
| CALM | 0.71 / 0.71 | 4.79 / 4.69 | 3.38 / 3.32 | 1.13 / 1.13 | 1.20 / 1.20 | 1.017 / 0.982 | 0.62 / 0.62 | 1.0 / 0.9 | 0.12 / 0.12 | 0.09 / 0.09 | 7 |
| ORLY | 0.60 / 0.60 | 7.13 / 7.11 | 2.75 / 2.70 | 12.74 / 12.75 | 1.06 / 1.06 | 1.072 / 1.071 | 0.52 / 0.52 | 1.1 / 1.0 | 0.10 / 0.10 | 0.06 / 0.06 | 19 |
| WING | 0.57 / 0.57 | 6.32 / 6.32 | 3.18 / 3.09 | 6.19 / 6.20 | 1.64 / 1.64 | 1.032 / 1.011 | 0.54 / 0.54 | 1.8 / 1.2 | 0.12 / 0.12 | 0.05 / 0.05 | 15 |
| BKNG | 0.56 / 0.56 | 8.70 / 8.67 | 3.14 / 2.72 | 70.22 / 70.21 | 1.62 / 1.62 | 1.085 / 1.081 | 0.45 / 0.45 | 1.7 / 1.3 | 0.03 / 0.03 | −0.00 / 0.00 | 21 |
| DUOL | 0.54 / 0.54 | 6.50 / 6.53 | 4.52 / 4.07 | 5.06 / 5.08 | 1.77 / 1.77 | 1.049 / 1.020 | 0.63 / 0.63 | 1.5 / 1.2 | 0.15 / 0.15 | 0.08 / 0.08 | 17 |
| MIDD | 0.46 / 0.46 | 4.91 / 5.01 | 5.44 / 2.96 | 1.52 / 1.52 | 1.10 / 1.10 | 1.017 / 0.967 | 0.74 / 0.73 | 2.9 / 1.8 | 0.20 / 0.20 | 0.11 / 0.11 | 6 |
| SAIA | 0.39 / 0.39 | 6.03 / 5.96 | 5.35 / 5.04 | 11.51 / 11.51 | 2.50 / 2.50 | 1.052 / 1.014 | 0.66 / 0.66 | 1.3 / 1.0 | 0.12 / 0.12 | 0.05 / 0.05 | 12 |
| FIZZ | 0.22 / 0.22 | 5.92 / 5.90 | 2.80 / 2.53 | 0.68 / 0.68 | 1.47 / 1.47 | 0.970 / 0.930 | 0.74 / 0.75 | 1.0 / 1.0 | 0.19 / 0.20 | 0.11 / 0.11 | 5 |
| LANC | 0.19 / 0.19 | 5.25 / 5.35 | 2.90 / 2.52 | 3.29 / 3.29 | 1.83 / 1.83 | 0.955 / 0.972 | 0.75 / 0.76 | 1.5 / 1.0 | 0.11 / 0.11 | 0.02 / 0.01 | 16 |
| IPAR | 0.19 / 0.19 | 5.36 / 5.42 | 4.24 / 4.07 | 2.29 / 2.29 | 1.94 / 1.94 | 1.004 / 0.974 | 0.79 / 0.80 | 3.0 / 1.6 | 0.23 / 0.22 | 0.10 / 0.10 | 11 |

Definitions: trades per second over the twelve sessions; outlier share,
Hill and ψ* on 16 October; ψ* in basis points is `10⁴ ψ*/X_T`; linear/Huber
is the median of the linear path's spot volatility divided by the median of
the Huber path's, on every 300th trade of the twelve days; the signature
ratio is the square root of the twelve-day mean of squared Huber signatures
at 60 s over
the same at 1 s; the linear jump is the largest linear spot volatility from
6 minutes before to 12 minutes after the trade over its median from 15 to
3 minutes before; the ACF is of one-trade Huber price changes, median over
the twelve days.

Median absolute gap to the paper over the 55 rows: trades per second 0.002,
outlier share 0.012 points, Hill 0.017, ψ* 0.003 cents, signature ratio
0.004, ACF 0.003 at both lags; linear/Huber 0.047 and the jump 0.59 (the
two columns built on the linear spot path, see "What still differs").

The SPY jump depends on SPY's own memory. SPY's worst print on 16 October is
6 shares at $522.93 against a Huber price of $580.41; the linear spot path
goes from 7.7% to 1,136%. With the NVDA value `a = 563` in place of SPY's
71 trades, the same print gives a jump of 67 instead of 147
(`exp4_summary.json`, `G.jump`).

#### Table B.1, replication / paper

Lag-1 and lag-2 ACF of the one-trade price change, medians over the stocks
(SPY excluded) of each trading-rate group; "12 days" is each stock's median
over the twelve days, then the median over the group.

| trades per second | stocks | lag 1 Huber 16 Oct | lag 1 Huber 12 days | lag 1 trades 16 Oct | lag 2 Huber 16 Oct | lag 2 Huber 12 days | lag 2 trades 16 Oct |
|---|---|---|---|---|---|---|---|
| above 10 | 7 / 7 | −0.03 / −0.03 | −0.05 / −0.06 | −0.46 / −0.46 | 0.05 / 0.05 | 0.05 / 0.05 | −0.01 / −0.01 |
| 3 to 10 | 19 / 19 | 0.07 / 0.09 | 0.07 / 0.08 | −0.46 / −0.46 | 0.03 / 0.03 | 0.04 / 0.05 | −0.02 / −0.02 |
| 1 to 3 | 15 / 15 | 0.13 / 0.13 | 0.13 / 0.13 | −0.43 / −0.43 | 0.06 / 0.06 | 0.07 / 0.07 | −0.03 / −0.03 |
| below 1 | 13 / 13 | 0.10 / 0.09 | 0.12 / 0.12 | −0.43 / −0.43 | 0.01 / 0.01 | 0.05 / 0.06 | −0.03 / −0.03 |

#### The cross-sectional statements of Appendix B

Each statement computed on the replication and, as a check of the
definitions, on the published Table B.2 (which reproduces the paper's text
in every case):

| statement | replication | paper's table | paper's text |
|---|---|---|---|
| Hill of the 53 other stocks in [1, 2) / [2, 3) / [3, 4) / ≥ 4 | 8 / 24 / 14 / 7 | 7 / 28 / 12 / 6 | 7, 28, 12, 6 |
| largest Hill above 10 trades/s | 2.05 | 2.10 | "at most 2.1" |
| median jump above 10 / below 1 trades/s | 5.5 / 1.7 | 4.6 / 1.2 | 4.6, 1.2 |
| stocks jumping 3× or more, of 54 | 23 | 19 | 19 |
| median linear/Huber above 10 / below 1 trades/s | 1.24 / 1.03 | 1.22 / 1.01 | 1.22, 1.01 |
| median outlier share above 10 / below 1 trades/s | 2.0% / 5.9% | 2.0% / 5.9% | "2%", "6%" |
| median signature 1 min / 1 s above 10 / below 3 trades/s | 0.92 / 0.64 | 0.92 / 0.64 | 0.92, "roughly 0.6" |
| stocks with a negative 12-day lag-1 ACF | NVDA, AAPL, AMZN, TSLA, AVGO, AMD, MU | the same | the same seven |

The Mandelbrot `a` chosen by QLIKE ranges from 5 trades (FIZZ) to 566
(NVDA). For GOOG the QLIKE loss keeps falling as `a` grows, up to 10⁶; the
search stops at the top of its grid, 30,000 trades. The paper publishes `a`
for NVDA only.

---

## What still differs

* **Huber rms at k = 1 and 2** is 0.435 and 0.391 against 0.431 and 0.389,
  and the median filter mad is 0.001 cents high at k = 1, 2 and 25. The
  other mean and rms cells of Table 1 match; the lag-1 ACF of the median and
  Huber paths is 0.01 more negative than published. These cells are
  dominated by repricing jumps. With a 3-trade half-life `λ³ = 1/2`, so right after a jump the three newest
  trades hold exactly half the weight and the median is a tie between the old
  and the new level. This replication resolves the tie exactly, in favour of
  the newest trade's side. Leaving the tie to floating-point rounding of the
  summed weights gives a Huber rms of 0.429 at k = 1 instead; the published
  0.431 lies between the two, which is what a rounding-dependent tie in the
  original implementation would produce. At k = 2 the two rules give 0.391
  and 0.380 against the published 0.389 (`exp3_sensitivity.json`,
  `median_tie`).
* **Outlier count** 16,341 against 16,314 (1.074% against 1.07%). The
  head-only filter (B = 0) gives 16,312; the tail draws move the count by a
  few dozen (`exp3_sensitivity.json`, `head_only_B0`).
* **Opening threshold** 3.01 cents against 2.97. It is the median of the five
  previous openings' nonzero median residuals, which sit on a few discrete
  values (one cent divided by each day's price); the median falls on a
  different day's one-cent atom (b = 0.01 / 132.0 here, 0.01 / 133.8 for the
  published value; no presample opening is at $133.8, and $132.00 is the
  10 October opening). It stays at 3.010 cents for eight seeds of the tail
  draws of the presample median filter and moves by less than 0.001 cents
  with a first minute counted from 09:30 or from the opening print, or of 59
  to 61 seconds; it is set by how many one-cent residuals each of the five
  openings contributes (`exp3_sensitivity.json`, `delta0`). Only the first
  minute is affected.
* **Huber 0.1-second volatility** is 54.1 with the simulated factor against
  53.3. This is within the paper's own 1/M bias; the analytic factor gives
  53.3.
* **linear/Huber spot ratio (Table B.2).** Above the published value on 52
  of the 55 assets, by a median of 0.047 (SPY 1.396 against 1.322, MRVL
  1.440 against 1.268); the correlation with the published column is 0.96
  and the group medians of Appendix B are close (1.24 against 1.22 above 10
  trades a second). Two explanations are ruled out. The memory `a`: on
  MRVL, PEP and AMAT the ratio stays at 1.42–1.45, 1.42–1.44 and 1.36–1.48
  for every `a` from 10 to 3,000 trades. The damping factor: on 16 October
  the median over trades of the ratio of the simulated Huber factor at
  s = 1 to the simulated linear one is within 0.07%, 0.33% and 0.30% of one
  on NVDA, MRVL and PEP, and using one in place of the other moves the Huber 1-second volatility
  by 0.1%, 1.4% and 0.1%. Nor do the aggregation or the diurnal curve: on
  MRVL the median of the twelve daily ratios is 1.429 and their mean 1.425,
  against the pooled 1.440, and a diurnal curve fit to the linear path's own
  10-second estimates gives 1.458 (`exp4_summary.json`, `G.ratio` and
  `G.damping`). Every column built on the Huber path alone matches
  (outliers, Hill, ψ*, signature, ACF), so the gap sits in the linear
  filter's spot path; its cause was not identified.
* **Linear jump (Table B.2).** Built on the same linear path: a median of
  15% above the published values, 65% of the rows within 30%, a log
  correlation of 0.97 with the published column (`exp4_summary.json`,
  `F.paper_comparison`). It is a ratio of a maximum
  to a median around one print, so it is also sensitive to the window edges
  and to which print is taken (see the interpretation below).
* **Hill index of slow assets (Table B.2).** Within 0.05 of the paper on 39
  of the 55 assets, but higher on 11 of the 13 stocks below one trade a
  second, by 0.07 to 0.45 (BKNG 3.14 against 2.72, LANC 2.90 against 2.52)
  and by 2.48 on MIDD (5.44 against 2.96), and on 4 of the 15 stocks at 1
  to 3 trades a second, by 0.06 to 0.17 (VRTX 3.78 against 3.61); every
  stock above 3 trades a second is within 0.05 (`exp4_summary.json`,
  `F.paper_comparison.hill`). With 5,000 to 25,000 trades a day the 0.1% to 1% tail is 5
  to 250 residuals, and the result depends on how the Hill estimates are
  averaged over k, which the paper does not specify. An arithmetic average
  over k reproduces the slow stocks (BKNG 2.73 against 2.72, LANC 2.50
  against 2.52, FIZZ 2.59 against 2.53, MIDD 3.93 against 2.96) but not the
  liquid ones (NVDA 1.25 against 1.43, AAPL 1.76 against 2.10), while the
  log-uniform average used here reproduces the liquid ones (1.43, 2.05).
  The median absolute gap below one trade a second is 0.06 with the
  arithmetic average and 0.23 with the log-uniform one; above 10 trades a
  second it is 0.22 and 0.02 (`exp4_summary.json`, `G.hill_average`).
  In the Appendix B count [2, 3) holds 24 stocks here against 28.
* **GOOG's `a`.** QLIKE has no interior minimum for GOOG; the search stops
  at the top of the grid, 30,000 trades. GOOG's linear/Huber (1.281 against
  1.171) and jump (5.0 against 7.4) depend on that bound.
* **Corrected variance-gamma signature (Appendix A.7).** With the Brownian
  correction the noise-free variance-gamma estimate is above the truth at
  every short lag (+23.5 points at 0.1 s, +2.0 at 1 s); the paper says it is
  below at the shortest s and above just beyond. The uncorrected estimate
  agrees with the paper (78% of the truth at 0.1 s against the paper's 80%),
  and the Brownian correction at 0.1 s is about 1/0.625 = 1.6, so the
  product, about 1.25, is above one; the paper's statement does not follow
  from its own two numbers.
* **Days of Figure A.4.** The day with the largest gap between the two
  startup rules is the paper's day for NVDA and INTC, not for AAPL, COST,
  CROX and IPAR, where the paper's day ranks 10th, 9th, 10th and 3rd of the
  twelve (`exp4_summary.json`, `D`; the paper does not define "moves δ_t
  most").
* **Hill indices of filtered returns** (Table 1's lower panel, Table 2's
  Hill row, the ends of the Table A.1 ranges) differ by 0.01 to 0.06. The
  returns overlap heavily, so a few extreme moves enter many order
  statistics, and the estimates depend on the averaging choice below.
* **Hill index on 12 random subsamples of 25,000 trades** ranges from 1.25
  to 1.77 against 1.33 to 1.49. With 25,000 trades the 0.1% to 1% tail is
  25 to 250 residuals, and the paper does not say how its subsamples are
  drawn. The range depends on that choice: 1.25 to 1.77 for trades drawn
  without replacement from the whole day, 1.26 to 1.76 for every 60th trade
  at twelve evenly spaced offsets, 1.21 to 2.25 for eleven of twelve evenly
  spaced contiguous blocks, and 10.86 for the block of the first 25,000
  trades after the open (`exp3_sensitivity.json`, `hill_subsamples`). The
  published range lies inside each of them.

The Hill index averages ζ(k) uniformly in log k, for k from 0.1% to 1% of all
observations, with zeros counted in n. This choice of average is
interpretation, not text. It reproduces about 30 independent published cells
across Tables 1, 2, 3, A.1 and B.2.

Further readings of the text that are choices, not facts:

* The spot filter drops a return until the trade is `s` seconds after the
  open. `s` is the return lag: 1 second in Experiment 3 (Section 4.6) and
  10 seconds in Appendix A.2–A.3. Section 4.6 uses the opening-print return
  "only once that trade is at least 1 second old".
* The opening auction is the first print with the opening condition
  (Massive 17), which on 16 October is the paper's 1,857,756 shares at
  $134.01.
* The Table B.2 jump is taken around the largest Huber residual at least 15
  minutes after the open, so that its baseline window lies in the session.
  For NVDA, SPY and every asset whose largest residual is later in the day
  this is the day's largest residual.
* The QLIKE search for `a` uses a log grid from 10 to 30,000 trades, refined
  by golden section, and extends the grid downwards when the minimum is at
  its lower end (TEAM, CALM, MIDD, FIZZ choose 5 to 9 trades).
* In Appendix A.7 the noise is added to the same 100 paths as the
  noise-free case, and the variance-gamma rate is β = 1 with the price in
  percent, which makes the standard deviation of `Z(23,400)` about 40% of its
  mean, as the paper says.
* In Appendix A.5 the trade-count rule uses masses `(T60 − t)/T60` and
  `t/T60`, the text's formula, and Figure A.4 shows each stock's day with the
  largest gap between the two thresholds.
