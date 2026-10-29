"""Fast filters for the paper's trade-level pipeline.

The median and Huber filters use exponential trade-time weights with a
half-life of a few trades. The 99.9% head is ``h = 29`` at a 3-trade
half-life (lags 0 to 29), so each update looks at a fixed window. The noise-scale filter is a clock-time exponentially
weighted median of nonzero absolute residuals, maintained in an order-statistic
treap (the paper's Fenwick tree plays the same role).

The finite-sample factor ``c_{t,s}`` is the ratio of a Brownian-motion
return's variance to the variance of the same return after filtering. Two
versions are provided: ``simulated_damping`` is the Monte Carlo of
Appendix A.6, one factor per filter (Table 2); ``damping_correction`` is the
closed form for the 30-lag linear filter, which Appendix A.6 puts within 1%
of the Huber factor on average (used for the multi-day signatures and the
spot volatility).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from .config import MAD_TO_SD, STRATIFIED_HEAD_MASS, THRESHOLD_MULTIPLIER


# ---------------------------------------------------------------------------
# Trade-time exponential median and Huber
# ---------------------------------------------------------------------------

@njit(cache=True)
def _head_lag(n_avail: int, lam: float, head_mass: float) -> int:
    """Smallest lag h with cumulative weight at least ``head_mass``, or n_avail-1."""
    if n_avail <= 1:
        return 0
    norm = (1.0 - lam ** n_avail) / (1.0 - lam)
    acc = 0.0
    wj = 1.0
    h = n_avail - 1
    for j in range(n_avail):
        acc += wj / norm
        if acc >= head_mass:
            return j
        wj *= lam
    return h


@njit(cache=True)
def _tail_lag(h: int, t: int, lam: float) -> int:
    """One draw of the lag ``k`` in ``{h+1, ..., t}`` with ``P(k) ∝ lam**k``.

    Inverse transform of the truncated geometric law, O(1) in ``t``
    (Remark 4(d) and Section 3.4).
    """
    n_tail = t - h
    u = np.random.random()
    g = int(math.floor(math.log(1.0 - u * (1.0 - lam ** n_tail)) / math.log(lam)))
    if g < 0:
        g = 0
    if g > n_tail - 1:
        g = n_tail - 1
    return h + 1 + g


@njit(cache=True)
def _fill_stratified(y, t, lam, n_tail_draws, head_mass, vals, wts, lags):
    """Head lags ``0..h`` with weights ``lam**j``, then the tail draws.

    Returns the number of filled entries. Tail draws share the tail mass
    ``sum_{k>h} lam**k`` equally. Weights are unnormalised; every caller
    only needs ratios.
    """
    n_avail = t + 1
    h = _head_lag(n_avail, lam, head_mass)
    m = h + 1
    wj = 1.0
    for j in range(m):
        vals[j] = y[t - j]
        wts[j] = wj
        lags[j] = j
        wj *= lam
    if n_tail_draws > 0 and h < t:
        tail_mass = (lam ** (h + 1) - lam ** (t + 1)) / (1.0 - lam)
        wb = tail_mass / n_tail_draws
        for b in range(n_tail_draws):
            k = _tail_lag(h, t, lam)
            vals[m + b] = y[t - k]
            wts[m + b] = wb
            lags[m + b] = k
        m += n_tail_draws
    return m


def _buffer_size(halflife: float, n_tail_draws: int, head_mass: float) -> int:
    lam = 2.0 ** (-1.0 / halflife)
    return int(math.ceil(math.log(1.0 - head_mass) / math.log(lam))) + 8 + int(n_tail_draws)


@njit(cache=True)
def _exponential_median(y, halflife, n_tail_draws, head_mass, exact_tie, seed, size):
    n = y.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = y[0]
    if n == 1:
        return out
    out[n - 1] = y[n - 1]
    np.random.seed(seed)
    lam = 2.0 ** (-1.0 / halflife)
    vals = np.empty(size, dtype=np.float64)
    wts = np.empty(size, dtype=np.float64)
    lags = np.empty(size, dtype=np.int64)
    for t in range(1, n - 1):
        m = _fill_stratified(y, t, lam, n_tail_draws, head_mass, vals, wts, lags)
        order = np.argsort(vals[:m])
        half = 0.0
        for j in range(m):
            half += wts[j]
        # At a 3-trade half-life lam**3 = 1/2: after a jump the newest three
        # trades carry half of the weight up to a term of order lam**t, so
        # rounding decides. In exact arithmetic the side holding the newest
        # trade is the heavier one, and the tie is broken that way.
        # ``exact_tie=False`` leaves the tie to the rounding of the sums.
        half *= 0.5
        tol = 1e-12 * half
        cw = 0.0
        med = vals[order[m - 1]]
        for j in range(m):
            cw += wts[order[j]]
            v = vals[order[j]]
            if exact_tie:
                if cw > half + tol or (cw >= half - tol and y[t] <= v):
                    med = v
                    break
            elif cw >= half:
                med = v
                break
        out[t] = med
    return out


def exponential_median(y: np.ndarray, halflife: float = 3, B: int = 0, seed: int = 0,
                       head_mass: float = STRATIFIED_HEAD_MASS, exact_tie: bool = True) -> np.ndarray:
    """Exponentially weighted median. Endpoints are the observations themselves.

    The median is the smallest value whose cumulative weight reaches one half
    (the convention of Section 4.3). ``B > 0`` adds the stratified tail of
    Remark 4(d): the head ``h = W_t^{-1}(0.999)`` is exact and ``B`` lags
    beyond it are drawn with probability proportional to their weights.
    ``B = 0`` drops the 0.1% tail mass. A tie at one half goes to the side of
    the newest trade; ``exact_tie=False`` leaves it to floating-point rounding.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    return _exponential_median(y, float(halflife), int(B), float(head_mass), bool(exact_tie), _numba_seed(seed),
                               _buffer_size(halflife, B, head_mass))


def _numba_seed(seed: int) -> int:
    """Numba keeps its own Mersenne Twister; seed it from ``default_rng``."""
    return int(np.random.default_rng(seed).integers(0, 2**31 - 1))


def exponential_huber(
    y: np.ndarray,
    delta: np.ndarray,
    init: np.ndarray,
    halflife: float = 3,
    B: int = 0,
    seed: int = 0,
    delta0: float | None = None,
    head_mass: float = STRATIFIED_HEAD_MASS,
) -> np.ndarray:
    """Huber filter by scalar IRLS, started at ``init`` (the median filter).

    Lag ``j`` at time ``t`` is penalised with threshold ``delta[t-j-1]``; the
    opening print uses ``delta0`` (the presample threshold of Section 4.6,
    which equals ``delta[0]`` when the first median residual is zero).
    Iteration stops at 1e-12. ``B`` tail draws as in ``exponential_median``;
    each draw carries its own trade's threshold.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    delta = np.ascontiguousarray(delta, dtype=np.float64)
    init = np.ascontiguousarray(init, dtype=np.float64)
    d0 = float(delta[0]) if delta0 is None else float(delta0)
    return _exponential_huber(y, delta, init, float(halflife), int(B), float(head_mass), _numba_seed(seed),
                              _buffer_size(halflife, B, head_mass), d0)


@njit(cache=True)
def _exponential_huber(y, delta, init, halflife, n_tail_draws, head_mass, seed, size, delta0):
    n = y.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = y[0]
    if n == 1:
        return out
    out[n - 1] = y[n - 1]
    np.random.seed(seed)
    lam = 2.0 ** (-1.0 / halflife)
    vals = np.empty(size, dtype=np.float64)
    wts = np.empty(size, dtype=np.float64)
    dlts = np.empty(size, dtype=np.float64)
    lags = np.empty(size, dtype=np.int64)
    for t in range(1, n - 1):
        m = _fill_stratified(y, t, lam, n_tail_draws, head_mass, vals, wts, lags)
        for j in range(m):
            i = t - lags[j]
            if i > 0:
                dlts[j] = delta[i - 1]
            else:
                dlts[j] = delta0
        theta = init[t]
        for _it in range(60):
            sw = 0.0
            sy = 0.0
            for j in range(m):
                r = abs(vals[j] - theta)
                dlt = dlts[j]
                if r <= dlt or dlt <= 0.0:
                    fac = 1.0
                else:
                    fac = dlt / r
                ww = wts[j] * fac
                sw += ww
                sy += ww * vals[j]
            if sw <= 0.0:
                break
            new_theta = sy / sw
            if abs(new_theta - theta) < 1e-12:
                theta = new_theta
                break
            theta = new_theta
        out[t] = theta
    return out


@njit(cache=True)
def exponential_mean(y: np.ndarray, halflife: int = 3) -> np.ndarray:
    """Exact finite-sample exponential weighted mean, endpoints fixed at y."""
    n = y.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    lam = 2.0 ** (-1.0 / halflife)
    out[0] = y[0]
    if n == 1:
        return out
    acc_y = y[0]
    acc_w = 1.0
    for t in range(1, n - 1):
        acc_y = y[t] + lam * acc_y
        acc_w = 1.0 + lam * acc_w
        out[t] = acc_y / acc_w
    out[n - 1] = y[n - 1]
    return out


# ---------------------------------------------------------------------------
# Clock-time weighted median via a treap
# ---------------------------------------------------------------------------

@njit(cache=True)
def _pull(i, left, right, wt, sub):
    s = wt[i]
    li = left[i]
    ri = right[i]
    if li >= 0:
        s += sub[li]
    if ri >= 0:
        s += sub[ri]
    sub[i] = s


@njit(cache=True)
def _merge(a, b, left, right, prio, wt, sub):
    if a < 0:
        return b
    if b < 0:
        return a
    if prio[a] > prio[b]:
        right[a] = _merge(right[a], b, left, right, prio, wt, sub)
        _pull(a, left, right, wt, sub)
        return a
    left[b] = _merge(a, left[b], left, right, prio, wt, sub)
    _pull(b, left, right, wt, sub)
    return b


@njit(cache=True)
def _split(root, key, left, right, prio, val, wt, sub):
    """Split into (keys <= key, keys > key)."""
    if root < 0:
        return -1, -1
    if val[root] <= key:
        rl, rr = _split(right[root], key, left, right, prio, val, wt, sub)
        right[root] = rl
        _pull(root, left, right, wt, sub)
        return root, rr
    ll, lr = _split(left[root], key, left, right, prio, val, wt, sub)
    left[root] = lr
    _pull(root, left, right, wt, sub)
    return ll, root


@njit(cache=True)
def _prefix(root, key, left, right, val, wt, sub):
    """Total weight of nodes with value <= key."""
    if root < 0:
        return 0.0
    if val[root] <= key:
        lw = 0.0
        if left[root] >= 0:
            lw = sub[left[root]]
        return lw + wt[root] + _prefix(right[root], key, left, right, val, wt, sub)
    return _prefix(left[root], key, left, right, val, wt, sub)


@njit(cache=True)
def _quantile(root, target, left, right, val, wt, sub):
    node = root
    while node >= 0:
        lw = 0.0
        if left[node] >= 0:
            lw = sub[left[node]]
        if target <= lw:
            node = left[node]
        elif target <= lw + wt[node]:
            return val[node]
        else:
            target -= lw + wt[node]
            node = right[node]
    return np.nan


@njit(cache=True)
def _successor(root, key, left, right, val):
    best = 1e300
    node = root
    while node >= 0:
        if val[node] > key:
            if val[node] < best:
                best = val[node]
            node = left[node]
        else:
            node = right[node]
    return best


@njit(cache=True)
def _xorshift(state):
    x = state[0]
    x ^= (x << np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= x >> np.uint64(7)
    x ^= (x << np.uint64(17)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    state[0] = x
    return x


@njit(cache=True)
def _insert(root, nodes, value, weight, left, right, prio, val, wt, sub, state):
    node = nodes[0]
    nodes[0] = node + 1
    left[node] = -1
    right[node] = -1
    prio[node] = _xorshift(state)
    val[node] = value
    wt[node] = weight
    sub[node] = weight
    lo, hi = _split(root, value, left, right, prio, val, wt, sub)
    root = _merge(lo, node, left, right, prio, wt, sub)
    root = _merge(root, hi, left, right, prio, wt, sub)
    return root


@njit(cache=True)
def _build_equal(values, left, right, prio, val, wt, sub, nodes, state):
    root = -1
    for i in range(values.shape[0]):
        root = _insert(root, nodes, values[i], 1.0, left, right, prio, val, wt, sub, state)
    return root


@njit(cache=True)
def clock_time_scale(
    tau_open: np.ndarray,
    abs_resid: np.ndarray,
    presample: np.ndarray,
    halflife_s: float = 240.0,
    startup_s: float = 60.0,
    by_count: bool = False,
    T60: int = 0,
) -> np.ndarray:
    """Clock-time exponentially weighted median of nonzero |residuals|.

    For ``tau_open <= startup_s`` the estimate mixes the previous openings
    (equal weights, total mass ``(startup - tau) / startup``) with the day's
    own exponential weights (total mass ``tau / startup``), which is the
    Section 4.6 startup rule. After that minute only the day's weights remain.
    ``by_count=True`` is the Appendix A.5 alternative: the masses are
    ``(T60 - t) / T60`` and ``t / T60`` for trade ``t = 1, ..., T60``.
    """
    n = abs_resid.shape[0]
    out = np.empty(n, dtype=np.float64)
    n_pre = presample.shape[0]
    cap = n + n_pre + 2
    left = np.empty(cap, dtype=np.int64)
    right = np.empty(cap, dtype=np.int64)
    prio = np.empty(cap, dtype=np.uint64)
    val = np.empty(cap, dtype=np.float64)
    wt = np.empty(cap, dtype=np.float64)
    sub = np.empty(cap, dtype=np.float64)
    left[:] = -1
    right[:] = -1
    nodes = np.zeros(1, dtype=np.int64)
    state = np.zeros(1, dtype=np.uint64)
    state[0] = np.uint64(88172645463325252)

    pre_root = -1
    if n_pre > 0:
        pre_root = _build_equal(presample, left, right, prio, val, wt, sub, nodes, state)
    pre_total = float(n_pre)

    cur_root = -1
    alpha = math.log(2.0) / halflife_s
    last = 0.0
    if n_pre > 0:
        last = _quantile(pre_root, 0.5 * pre_total, left, right, val, wt, sub)

    for t in range(n):
        if abs_resid[t] > 0.0:
            # Stored weight exp(alpha * tau) so relative weights match the kernel.
            cur_root = _insert(
                cur_root,
                nodes,
                abs_resid[t],
                math.exp(alpha * tau_open[t]),
                left,
                right,
                prio,
                val,
                wt,
                sub,
                state,
            )
        tau = tau_open[t]
        if tau < 0.0:
            tau = 0.0
        cur_total = 0.0
        if cur_root >= 0:
            cur_total = sub[cur_root]
        if by_count:
            done = t + 1 >= T60
        else:
            done = tau >= startup_s
        if done or n_pre == 0:
            if cur_total > 0.0:
                last = _quantile(cur_root, 0.5 * cur_total, left, right, val, wt, sub)
        else:
            if by_count:
                mass_pre = (T60 - (t + 1)) / T60
                mass_cur = (t + 1) / T60
            else:
                mass_pre = (startup_s - tau) / startup_s
                mass_cur = tau / startup_s
            if cur_total <= 0.0:
                last = _quantile(pre_root, 0.5 * pre_total, left, right, val, wt, sub)
            else:
                # Binary search the smallest value whose mixture CDF reaches 1/2.
                lo = 0.0
                hi = 1.0
                # Expand hi to cover both trees.
                if pre_root >= 0:
                    hi = max(hi, _max_val(pre_root, right, val))
                hi = max(hi, _max_val(cur_root, right, val))
                for _ in range(64):
                    mid = 0.5 * (lo + hi)
                    cdf = mass_pre * _prefix(pre_root, mid, left, right, val, wt, sub) / pre_total
                    cdf += mass_cur * _prefix(cur_root, mid, left, right, val, wt, sub) / cur_total
                    if cdf >= 0.5:
                        hi = mid
                    else:
                        lo = mid
                sp = _successor(pre_root, lo, left, right, val)
                sc = _successor(cur_root, lo, left, right, val)
                snap = sp if sp < sc else sc
                if snap < 1e299:
                    last = snap
        if last <= 0.0:
            last = 1e-8
        out[t] = last
    return out


@njit(cache=True)
def _max_val(root, right, val):
    node = root
    while right[node] >= 0:
        node = right[node]
    return val[node]


@njit(cache=True)
def directional_probability(
    tau: np.ndarray,
    d_pos: np.ndarray,
    d_neg: np.ndarray,
    halflife_s: float,
    warmup: int,
) -> np.ndarray:
    """Clock-time exponential mean of the two outlier indicators, as a ratio.

    ``q_t = p+_t / (p+_t + p-_t)`` with weights proportional to
    ``2^{-(tau_t - tau_{t-j}) / halflife}``. Undefined on the opening minute
    and at the terminal auction print.
    """
    n = tau.shape[0]
    q = np.empty(n, dtype=np.float64)
    alpha = math.log(2.0) / halflife_s
    sp = 0.0
    sn = 0.0
    zp = 0.0
    zn = 0.0
    for t in range(n):
        if t == 0:
            decay = 0.0
        else:
            dt = tau[t] - tau[t - 1]
            if dt < 0.0:
                dt = 0.0
            decay = math.exp(-alpha * dt)
        sp = decay * sp + d_pos[t]
        sn = decay * sn + d_neg[t]
        zp = decay * zp + 1.0
        zn = decay * zn + 1.0
        pp = sp / zp
        pn = sn / zn
        denom = pp + pn
        if denom > 0.0:
            q[t] = pp / denom
        else:
            q[t] = 0.5
    for t in range(warmup):
        q[t] = np.nan
    if n > 0:
        q[n - 1] = np.nan
    return q


@njit(cache=True)
def clock_binomial_filter(tau: np.ndarray, d: np.ndarray, halflife_s: float) -> np.ndarray:
    """``p_t = sum_j w_{t,j} d_{t-j}`` with ``w_{t,j} ∝ 2^{-(tau_t - tau_{t-j})/h}``.

    The Bernoulli-loss filter of Section 4.7, normalised at every ``t``.
    """
    n = tau.shape[0]
    p = np.empty(n, dtype=np.float64)
    alpha = math.log(2.0) / halflife_s
    s = 0.0
    z = 0.0
    for t in range(n):
        decay = 0.0
        if t > 0:
            dt = tau[t] - tau[t - 1]
            if dt < 0.0:
                dt = 0.0
            decay = math.exp(-alpha * dt)
        s = decay * s + d[t]
        z = decay * z + 1.0
        p[t] = s / z
    return p


def outlier_predictive_loglik(tau, d_pos, d_neg, halflife_s: float, T60: int) -> float:
    """Trade-by-trade Bernoulli log-likelihood of ``d±_t`` given ``p±_{t-1}``.

    Summed over ``t ∈ {T60+1, ..., T−1}`` and the two series, as used in
    Section 4.7 to choose the 13-second half-life.
    """
    tau = np.ascontiguousarray(tau, dtype=np.float64)
    total = 0.0
    n = len(tau)
    for d in (d_pos, d_neg):
        d = np.ascontiguousarray(d, dtype=np.float64)
        p = np.clip(clock_binomial_filter(tau, d, float(halflife_s)), 1e-12, 1.0 - 1e-12)
        idx = np.arange(max(int(T60), 1), n - 1)
        prev = p[idx - 1]
        total += float(np.sum(d[idx] * np.log(prev) + (1.0 - d[idx]) * np.log(1.0 - prev)))
    return total


# ---------------------------------------------------------------------------
# Finite-sample damping correction c_{t,s} for a 30-lag linear filter
# ---------------------------------------------------------------------------

@njit(cache=True)
def damping_correction(
    tau: np.ndarray,
    s: float,
    halflife: int = 3,
    window: int = 30,
) -> np.ndarray:
    """``c_{t,s} = (tau_t - tau_u) / Q_t`` for the truncated linear filter.

    ``u`` is the latest trade at least ``s`` seconds earlier, or the opening
    print when no such trade exists. ``Q`` is ``Var(filter_t - filter_u) / sigma^2``
    for a Brownian log-price. Raw returns use ``c = 1``; callers should not
    apply this array to the unfiltered price.
    """
    n = tau.shape[0]
    c = np.ones(n, dtype=np.float64)
    if n < 3 or window < 2:
        return c
    lam = 2.0 ** (-1.0 / halflife)
    w = np.empty(window, dtype=np.float64)
    acc = 0.0
    wj = 1.0
    for j in range(window):
        w[j] = wj
        acc += wj
        wj *= lam
    for j in range(window):
        w[j] /= acc
    wcum = np.empty(window, dtype=np.float64)
    run = 0.0
    for j in range(window):
        run += w[j]
        wcum[j] = run
    h = window  # observations in the window
    # Precompute the two boundary kernels of length h-1.
    edge_new = np.empty(h - 1, dtype=np.float64)
    edge_old = np.empty(h - 1, dtype=np.float64)
    for k in range(h - 1):
        edge_new[k] = wcum[k] * wcum[k]
        edge_old[k] = (1.0 - wcum[k]) * (1.0 - wcum[k])

    for t in range(1, n):
        # Last index u with tau[u] <= tau[t] - s.
        # Linear scan from a moving pointer would be faster; searchsorted is
        # not available as a method here, so walk back from t. The gap in
        # trades is small except at long s, where we jump with an exponential
        # search.
        target = tau[t] - s
        if tau[0] > target:
            u = 0
        else:
            lo = 0
            hi = t
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if tau[mid] <= target:
                    lo = mid
                else:
                    hi = mid
            u = lo
        lag = tau[t] - tau[u]
        if lag <= 0.0:
            c[t] = 1.0
            continue
        q = _filter_var_gap(tau, t, u, w, wcum, edge_new, edge_old, h)
        if q <= 1e-18:
            c[t] = 1.0
        else:
            c[t] = lag / q
    return c


@njit(cache=True)
def _filter_var_gap(tau, t, u, w, wcum, edge_new, edge_old, h):
    """Variance / sigma^2 of filter(B)_t - filter(B)_u, window length h.

    When ``u == 0`` the past filter is the opening observation itself
    (the paper fixes the first print). When ``u < h-1`` or ``t < h-1`` the
    filter is the shorter exact weighted mean; that case is summed directly.
    """
    if u == 0 and t >= h - 1:
        # theta_0 = B_0, so coefficients of increments i>=1 are alpha_t only.
        q = tau[t - (h - 1)] - tau[0]
        for k in range(h - 1):
            i = t - k
            dt = tau[i] - tau[i - 1]
            q += edge_new[k] * dt
        return q
    if t >= h - 1 and u >= h - 1 and (t - u) >= (h - 1):
        q = tau[t - (h - 1)] - tau[u]
        for k in range(h - 1):
            i_new = t - k
            dt_new = tau[i_new] - tau[i_new - 1]
            q += edge_new[k] * dt_new
            i_old = u - k
            dt_old = tau[i_old] - tau[i_old - 1]
            q += edge_old[k] * dt_old
        return q
    return _short_var(tau, t, u, w, h)


@njit(cache=True)
def _short_var(tau, t, u, w_full, h):
    """Quadratic form when a 30-lag window is cut short or the two windows overlap.

    Coefficients match ``_alpha``: geometric weights on at most ``h`` lags,
    renormalised, with the opening print left unfiltered.
    """
    if h >= 2 and w_full[0] > 0.0:
        lam = w_full[1] / w_full[0]
    else:
        lam = 0.8
    start = u - (h - 1)
    if start < 1:
        start = 1
    q = 0.0
    for i in range(start, t + 1):
        bt = _alpha(t, i, h, lam)
        if u == 0:
            bu = 0.0
        else:
            bu = _alpha(u, i, h, lam)
        beta = bt - bu
        dt = tau[i] - tau[i - 1]
        if dt < 0.0:
            dt = 0.0
        q += beta * beta * dt
    return q


@njit(cache=True)
def _alpha(t_idx, i_idx, h, lam):
    """Coefficient of level B_i in the filtered value at t, 0 if i > t.

    History length is min(h, t_idx+1). Weights are geometric, newest first,
    renormalised on that history.
    """
    if i_idx > t_idx or i_idx < 0:
        return 0.0
    n_obs = t_idx + 1
    if n_obs > h:
        n_obs = h
    # Oldest observation inside the window.
    oldest = t_idx - (n_obs - 1)
    if i_idx < oldest:
        return 1.0
    # Weight sum on observations with index >= i_idx, i.e. lags j <= t_idx - i_idx.
    lag_max = t_idx - i_idx
    num = 0.0
    den = 0.0
    wj = 1.0
    for j in range(n_obs):
        den += wj
        if j <= lag_max:
            num += wj
        wj *= lam
    if den <= 0.0:
        return 0.0
    return num / den


def threshold_from_scale(b: np.ndarray) -> np.ndarray:
    """``delta = 2 * 1.4826 * b`` (Section 4.3)."""
    return (THRESHOLD_MULTIPLIER * MAD_TO_SD) * np.asarray(b, dtype=np.float64)


def simulated_damping(
    tau: np.ndarray,
    s_list,
    sigma: float,
    delta: np.ndarray,
    halflife: int = 3,
    n_paths: int = 100,
    seed: int = 0,
) -> dict:
    """Section 4.4 Monte Carlo ``c_{t,s}`` for the median, Huber and linear filters.

    Brownian log-prices with per-second volatility ``sigma`` are observed at the
    trade times; ``c_{t,s}`` is the ratio of the path averages of
    ``(B_t - B_{t,s})^2`` and ``(B^_t - B^_{t,s})^2``. All filters share the
    paths, and the Huber uses the day's own threshold path ``delta``. The
    ratio of averages carries the paper's O(1/M) bias.
    """
    tau = np.asarray(tau, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    n = len(tau)
    rng = np.random.default_rng(seed)
    ref = {s: np.maximum(np.searchsorted(tau, tau - s, side="right") - 1, 0) for s in s_list}
    names = ("median", "huber", "linear")
    num = {s: np.zeros(n) for s in s_list}
    den = {f: {s: np.zeros(n) for s in s_list} for f in names}
    sd = sigma * np.sqrt(np.maximum(np.diff(tau, prepend=tau[0]), 0.0))
    for _ in range(n_paths):
        B = np.cumsum(sd * rng.standard_normal(n))
        m = exponential_median(B, halflife)
        paths = {"median": m, "huber": exponential_huber(B, delta, m, halflife),
                 "linear": exponential_mean(B, halflife)}
        for s in s_list:
            u = ref[s]
            num[s] += (B - B[u]) ** 2
            for f in names:
                den[f][s] += (paths[f] - paths[f][u]) ** 2
    return {f: {s: np.where(den[f][s] > 0.0, num[s] / np.maximum(den[f][s], 1e-300), 1.0)
                for s in s_list} for f in names}
