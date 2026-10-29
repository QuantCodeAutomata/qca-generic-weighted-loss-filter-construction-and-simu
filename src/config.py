"""Parameters of the four experiments.

Module-level constants only, grouped by experiment. Each comment gives the
meaning, the unit and the source: a section, equation or table of the paper,
or "not specified in the paper". Every experiment runs at these values, which
are the paper's full configuration.
"""

# ---------------------------------------------------------------------------
# Sample and data (Section 4, Appendix B)
# ---------------------------------------------------------------------------

# Ticker of the main empirical study. Section 4.
NVDA_TICKER = "NVDA"
# The twelve trading days of the paper, 1-16 October 2024. Section 4.
PAPER_DATES = [
    "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-07", "2024-10-08",
    "2024-10-09", "2024-10-10", "2024-10-11", "2024-10-14", "2024-10-15", "2024-10-16",
]
# The five sessions before 1 October: the opening presample of 1 October and
# the second half of its diurnal presample. Section 4.6.
PRESAMPLE_DATES = ["2024-09-24", "2024-09-25", "2024-09-26", "2024-09-27", "2024-09-30"]
# The five sessions before those: the rest of the ten-session diurnal
# presample of the first October days. Section 4.6.
DIURNAL_PRESAMPLE_DATES = ["2024-09-17", "2024-09-18", "2024-09-19", "2024-09-20", "2024-09-23"]
# Day of Tables 1-3, Figures 6, 8, 11-13 and the 16 October columns of Table B.2.
# Sections 4.3-4.7, Appendix B.
OCT16 = "2024-10-16"
# NVDA and the 54 assets of Table B.2, in the table's order of trading rate. The
# position of an asset sets its session seeds (SESSION_SEED), so new assets go at
# the end. Appendix B, Table B.2.
ASSETS = [
    "NVDA", "TSLA", "AAPL", "SPY", "AMD", "MSFT",
    "AMZN", "AVGO", "MU", "INTC", "META", "GOOG", "COIN", "MRVL", "RIVN", "HOOD",
    "CELH", "QCOM", "PYPL", "CSCO", "PEP", "AMAT", "SBUX", "NFLX", "CRWD", "ADBE",
    "COST", "DKNG", "TXN", "PANW", "ABNB", "GILD", "DDOG", "ETSY", "LCID", "TTD",
    "ROKU", "TEAM", "INTU", "AMGN", "ISRG", "VRTX", "REGN", "CROX", "MELI", "CALM",
    "ORLY", "WING", "BKNG", "DUOL", "MIDD", "SAIA", "FIZZ", "LANC", "IPAR",
]
# Regular session, seconds after midnight ET: 09:30:00 to 16:00:00. Section 4, footnote.
REGULAR_OPEN_SECONDS = 9 * 3600 + 30 * 60
REGULAR_CLOSE_SECONDS = 16 * 3600
# Session length in seconds (23,400). Section 4.
SESSION_SECONDS = 23_400.0
# Massive sale-condition id of the opening print (TAQ "O"), which becomes trade 1. Section 4, footnote.
OPENING_AUCTION_CONDITION = 17
# Massive sale-condition id of the closing print (TAQ "6"), appended at 16:00:00. Section 4, footnote.
CLOSING_AUCTION_CONDITION = 8
# Massive sale-condition ids of extended-hours trades, dropped. Section 4, footnote.
EXTENDED_HOURS_CONDITIONS = (12, 13)
# Massive exchange id of the FINRA trade reporting facilities (off-exchange prints). Section 4.2.
OFF_EXCHANGE_ID = 4
# Page size of the Massive trades endpoint, prints per request. Not specified in the paper.
TRADES_PAGE_LIMIT = 50_000

# ---------------------------------------------------------------------------
# The five filters of Section 4 (experiments 3 and 4)
# ---------------------------------------------------------------------------

# Half-life of the median, Huber and linear filters, in trades. Section 4.3-4.4.
PRICE_HALF_LIFE_TRADES = 3
# Half-life of the clock-time noise scale b_t, in seconds (4 minutes). Section 4.3.
SCALE_HALF_LIFE_S = 240.0
# Half-life of the directional outlier probability q_t, in seconds. Section 4.7.
OUTLIER_HALF_LIFE_S = 13.0
# Length of the opening minute over which the presample is swapped out, seconds. Section 4.6.
STARTUP_S = 60.0
# Number of previous openings in the presample of b_t. Section 4.6.
PRESAMPLE_OPENINGS = 5
# Weight mass of the exact head of the stratified estimator, h = W_t^{-1}(0.999). Remark 4(d), Section 4.3.
STRATIFIED_HEAD_MASS = 0.999
# Tail draws B per trade of the stratified estimator. Remark 4(d), Section 4.3.
TAIL_DRAWS = 100
# Huber threshold delta_t = THRESHOLD_MULTIPLIER * MAD_TO_SD * b_t. Section 4.3.
THRESHOLD_MULTIPLIER = 2.0
MAD_TO_SD = 1.4826
# Lags of the linear filter in the closed-form damping factor c_{t,s}. Section 4.4, Appendix A.6.
DAMPING_WINDOW_LAGS = 30
# Brownian paths M of the simulated damping factor of Table 2. Section 4.4.
DAMPING_PATHS = 100
# The Hill index averages the estimates over the largest 0.1% to 1% of observations. Section 4.4.1.
HILL_K_FRACTION_LOW = 0.001
HILL_K_FRACTION_HIGH = 0.01
# Base seed of the NVDA filters (exp_3, exp_4 A-E). Not specified in the paper.
NVDA_SEED = 20241016
# Base seed of the per-session filtering cache; asset k, session i uses
# SESSION_SEED + 1000 k + i. Not specified in the paper.
SESSION_SEED = 20241001

# ---------------------------------------------------------------------------
# Spot volatility (Sections 4.5.2-4.6, Appendix B)
# ---------------------------------------------------------------------------

# Return lag s of the spot volatility, seconds. Section 4.6.
SPOT_LAG_S = 1.0
# Sessions in the diurnal fit (the previous ten). Section 4.6.
DIURNAL_SESSIONS = 10
# Cosine-sine pairs of the diurnal Fourier fit (plus linear and quadratic terms). Section 4.6.
DIURNAL_FOURIER_PAIRS = 4
# Window of the uniform spot estimator that is the diurnal and QLIKE target, seconds. Section 4.6.
UNIFORM_SPOT_WINDOW_S = 10.0
# The first seconds blend towards the median first-minute variance of the previous openings. Section 4.6.
OPEN_BLEND_S = 15.0
# Forecast horizon of the QLIKE choice of a: sigma(u - 60) forecasts sigma_bar(u), seconds. Section 4.6.
QLIKE_HORIZON_S = 60.0
# First second u of the QLIKE sum. Not specified in the paper.
QLIKE_START_S = 75.0
# Log grid of the Mandelbrot a searched by QLIKE, in trades, then golden-section
# refinement steps. Not specified in the paper.
MANDELBROT_A_MIN = 10.0
MANDELBROT_A_MAX = 30_000.0
MANDELBROT_A_GRID_POINTS = 25
MANDELBROT_A_REFINE_STEPS = 14
# The paper's QLIKE value of a for NVDA, in trades (Section 4.6); used only where a check fixes a at it.
NVDA_A_PUBLISHED = 563.0
# Half-lives of the four exponential components of the Example 5 superposition, seconds. Figure 10.
HEXP_HALF_LIVES_S = (2.0, 20.0, 200.0, 2_000.0)
# Lags of the multi-day volatility signatures, seconds. Figure 9, Figure B.1.
SIGNATURE_S = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 900.0)
# Spot paths are compared every SPOT_SAMPLE_STEP-th trade (the linear/Huber ratio of
# Table B.2 and Figure 11). Not specified in the paper.
SPOT_SAMPLE_STEP = 300
# Linear jump of Table B.2: the peak from 6 minutes before to 12 minutes after the
# trade over the median from 15 to 3 minutes before it, seconds. Appendix B.
JUMP_PEAK_FROM_S = -6.0 * 60.0
JUMP_PEAK_TO_S = 12.0 * 60.0
JUMP_BASE_FROM_S = -15.0 * 60.0
JUMP_BASE_TO_S = -3.0 * 60.0
# The jump trade is the largest Huber residual at least this long after the open,
# so that the baseline window lies in the session, seconds. Not specified in the paper.
JUMP_MIN_TAU_S = 15.0 * 60.0

# ---------------------------------------------------------------------------
# Experiment 1: generic weighted-loss filter (Sections 2-3, Theorem 1)
# ---------------------------------------------------------------------------

# Path length T and seed of the Example 7(a) path. Not specified in the paper.
EXP1_T = 1000
EXP1_SEED = 1
# Monte Carlo sizes B of the simulated filter (Definition 3). Section 3.
EXP1_B_GRID = (100, 500, 5_000, 50_000)
# Seed of the lag draws of the simulated filter. Not specified in the paper.
EXP1_LAG_SEED = 123
# Exponential weights lambda, uniform window ell and hyperbolic a of the scenarios. Section 2.2.
EXP1_LAMBDA = 0.9
EXP1_UNIFORM_ELL = 15
EXP1_HYPERBOLIC_A = 10.0
# Huber threshold of the Huber-loss scenario, in units of Y. Not specified in the paper.
EXP1_HUBER_DELTA = 1.345
# Tail draws and seed of the stratified estimator check. Remark 4(d); seed not specified in the paper.
EXP1_STRATIFIED_B = 100
EXP1_STRATIFIED_SEED = 42
# Path lengths, B and the largest T with the O(T^2) exact median, for the runtime study. Section 3.4.
EXP1_RUNTIME_T = (1_000, 3_000, 10_000, 30_000, 100_000)
EXP1_RUNTIME_B = 100
EXP1_RUNTIME_EXACT_MAX_T = 10_000
# Seed of the iid Cauchy paths and of the lag draws of the runtime study. Not specified in the paper.
EXP1_RUNTIME_SEED = 0
# Theorem 1 check: B, Monte Carlo replications, seed and the time points t. Theorem 1.
THEOREM1_B = 500
THEOREM1_REPS = 2_000
THEOREM1_SEED = 11
THEOREM1_TIMES = (50, 200, 1_000)
# Weight-scheme illustration: path length, seed and the superposition shares. Example 5.
EXP1_SHOWCASE_T = 500
EXP1_SHOWCASE_SEED = 7
EXP1_SUPERPOSITION_SHARES = (0.5, 0.5)

# ---------------------------------------------------------------------------
# Experiment 2: weighted medians under Cauchy noise (Sections 2.4, 3.4)
# ---------------------------------------------------------------------------

# Example 7(a): mu_{t+1} = mu_t + EX7A_SIGNAL_SD eta_t. Example 7(a).
EX7A_SIGNAL_SD = 0.2
# Example 7(b): mu_{t+1} = phi mu_t + sqrt(1 - phi^2) eta_t. Example 7(b).
EX7B_PHI = 0.95
# Path length of Parts A, B, C and D. Figures 1-2, Section 3.4.
EXP2_T = 1000
# Window grid ell of Parts A and B, in observations. Figures 1-2.
EXP2_ELL_GRID = tuple(range(4, 51))
# Windows drawn on the illustrative paths. Figures 1-2.
EXP2_PATH_ELLS = (1, 7, 15)
# Paths averaged in Parts A and B, and their base seeds. Not specified in the paper.
EXP2_A_PATHS = 10
EXP2_A_SEED = 100
EXP2_B_PATHS = 8
EXP2_B_SEED = 200
# Anchor shares alpha of Part B. Figure 2.
EXP2_B_ALPHAS = (0.4, 0.6, 0.8, 1.0)
# Anchor share of the illustrative Part B paths. Figure 2.
EXP2_B_PATH_ALPHA = 0.6
# Part C: lambda, Monte Carlo sizes, paths, base seed. Figure 5, Theorem 2.
EXP2_C_LAMBDA = 0.9
EXP2_C_B_GRID = (500, 5_000, 50_000)
EXP2_C_PATHS = 5
EXP2_C_SEED = 300
# Figure 4: path length, B and the lambda grid of the predictive-loss panel. Figure 4.
EXP2_FIG4_T = 5_000
EXP2_FIG4_B = 500
EXP2_FIG4_LAMBDA_GRID = tuple(round(0.80 + 0.01 * i, 2) for i in range(16))
# Figure 3: epsilon values, largest B and the B values tabulated. Figure 3.
EXP2_FIG3_EPS = (0.05, 0.02, 0.01)
EXP2_FIG3_B_MAX = 10_000
EXP2_FIG3_B_TABLE = (101, 1_001, 10_001)
# Part D, Example 9: windows (odd and even), lag draws, seed. Example 9, Remark 3.
EXP2_D_ELLS = (7, 8)
EXP2_D_B = 20_000
EXP2_D_SEED = 400

# ---------------------------------------------------------------------------
# Experiment 3: NVDA, twelve sessions (Section 4)
# ---------------------------------------------------------------------------

# Lags k of Table 1, in trades; also the ACF lags of Tables 1 and 3. Table 1, Table 3.
TABLE1_K = (1, 2, 5, 10, 25)
# Return lags s of Table 2, seconds. Table 2.
TABLE2_S = (0.1, 1.0, 5.0, 10.0, 60.0, 900.0)
# Lags of the Table 3 score autocovariance in psi*, the exponential head. Section 4.4.1.
PSI_STAR_LAGS = 30
# Tail probabilities of the Table 3 error intervals (80%, 90%, 95%, 99%). Table 3.
CI_ALPHAS = (0.20, 0.10, 0.05, 0.01)
# Random subsamples of the Hill check and their size, in trades. Section 4.4.1.
HILL_SUBSAMPLES = 12
HILL_SUBSAMPLE_SIZE = 25_000
# Width of the bins of the direction analysis, seconds, and of the price-change groups, cents. Section 4.7.
DIRECTION_BIN_S = 120.0
DIRECTION_GROUP_CENTS = 10.0
# Grid of q_t half-lives searched by predictive Bernoulli log-likelihood, seconds. Section 4.7.
Q_HALF_LIFE_GRID_S = (2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0, 30.0, 45.0, 60.0, 120.0)
# Grid of noise-scale half-lives for the check loss of b_{t,60}, seconds. Section 4.3.
SCALE_HALF_LIFE_GRID_S = (30.0, 60.0, 120.0, 240.0, 480.0, 900.0)
# b_{t,60}: the scale at the latest trade at least this long before trade t, seconds. Section 4.3.
SCALE_LOSS_LAG_S = 60.0
# Grid of price half-lives and forecast horizons k of the Huber forecast loss, trades. Section 4.4.
PRICE_HALF_LIFE_GRID = (1.5, 2.0, 2.5, 3.0, 4.0, 6.0)
FORECAST_K = (2, 5, 10, 25, 50, 100)
# Scale half-lives drawn in Figure 6, seconds. Figure 6.
FIGURE6_HALF_LIVES_S = (30.0, 240.0, 900.0)
# Starts of the two fixed 5-second windows of Figure 8, seconds after 09:30 (13:00:08, 13:28:15). Figure 8.
FIGURE8_WINDOWS_S = (12_608.0, 14_295.0)
FIGURE8_WINDOW_LENGTH_S = 5.0
# Window of the left panel of Figure 12, seconds after 09:30 (10:30 to 10:42). Figure 12.
FIGURE12_WINDOW_S = (3_600.0, 4_320.0)
# Sessions filtered in parallel. Execution setting, not from the paper.
EXP3_WORKERS = 2

# Sensitivity checks of experiment 3 (16 October).
# Seeds of the tail draws of the presample median filter in the delta_0 check. Not specified in the paper.
DELTA0_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
# Lengths of the opening window of the presample residuals, seconds. Section 4.6 gives 60.
DELTA0_MINUTE_LENGTHS_S = (59.0, 60.0, 61.0)
# The published opening threshold of 16 October, cents. Section 4.3.
DELTA0_PUBLISHED_CENTS = 2.97
# Step between trades of the systematic Hill subsamples (HILL_SUBSAMPLES offsets). Not specified in the paper.
HILL_SYSTEMATIC_STEP = 60
# Contiguous blocks of HILL_SUBSAMPLE_SIZE trades in the block Hill subsamples. Not specified in the paper.
HILL_BLOCKS = 12

# ---------------------------------------------------------------------------
# Experiment 4: robustness and the appendices
# ---------------------------------------------------------------------------

# Module A, Table A.1: repricing screen (filtered price range above 10 cents within
# +-1 second), thinning step, alternative half-life in trades. Appendix A.1.
REPRICING_RANGE_DOLLARS = 0.10
REPRICING_HALF_WINDOW_S = 1.0
THINNING_STEP = 10
ALTERNATIVE_HALF_LIFE_TRADES = 6
# Module B: the longer return lag, seconds. Appendix A.2-A.3.
LONG_LAG_S = 10.0
# Module C: rolling window and step of psi*_t, trades. Appendix A.4.
PSI_WINDOW_TRADES = 100_000
PSI_WINDOW_STEP = 1_000
# Module D: stocks and horizon of the startup comparison, seconds. Appendix A.5, Figure A.4.
STARTUP_TICKERS = ("NVDA", "AAPL", "INTC", "COST", "CROX", "IPAR")
STARTUP_HORIZON_S = 120.0
# Module D: the day the paper's Figure A.4 shows for each of STARTUP_TICKERS, in that order. Figure A.4.
FIGURE_A4_DAYS = ("2024-10-11", "2024-10-10", "2024-10-01", "2024-10-09", "2024-10-14", "2024-10-10")
# Module E: lags of the simulated signatures, seconds. Table A.2.
SIMULATION_S = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0, 900.0)
# Module E: annual volatility of the simulated price, variance-gamma rate beta
# (price in percent), Student-t degrees of freedom and noise scale in cents. Appendix A.7.
SIMULATION_SIGMA_ANNUAL = 0.40
VG_BETA = 1.0
NOISE_DF = 1.2
NOISE_SCALE_CENTS = 0.24
# Module E: simulated days per case. Appendix A.7.
SIMULATION_PATHS = 100
# Module E: base seeds of the Brownian paths, their noise, the variance-gamma
# paths, their noise and the filters. Not specified in the paper.
SEED_BM = 10_000
SEED_BM_NOISE = 20_000
SEED_VG = 30_000
SEED_VG_NOISE = 40_000
SEED_SIMULATED_FILTERS = 50_000
# Module E: simulated days filtered in parallel. Execution setting, not from the paper.
SIMULATION_WORKERS = 2
# Module F: the six stocks of Figure B.1. Figure B.1.
FIGURE_B1_TICKERS = ("NVDA", "AAPL", "INTC", "COST", "CROX", "IPAR")
# Module F: trading-rate groups of Table B.1, trades per second. Table B.1.
TABLE_B1_EDGES = (10.0, 3.0, 1.0)
# Module F: tolerances of the comparison with the published Table B.2: Hill index
# (absolute) and linear jump (relative). Not specified in the paper.
HILL_MATCH_TOLERANCE = 0.05
JUMP_MATCH_TOLERANCE = 0.30
# Module F: assets processed in parallel. Execution setting, not from the paper.
CROSS_ASSET_WORKERS = 2

# Module G, sensitivity of the Table B.2 columns.
# Assets and Mandelbrot a grid (trades) of the linear/Huber check. Not specified in the paper.
RATIO_CHECK_TICKERS = ("MRVL", "PEP", "AMAT")
RATIO_CHECK_A_GRID = (10.0, 30.0, 100.0, 300.0, 1_000.0, 3_000.0)
# Asset of the aggregation and diurnal-curve checks. Not specified in the paper.
AGGREGATION_CHECK_TICKER = "MRVL"
# Assets whose simulated Huber and linear damping factors are compared at s = 1. Appendix A.6.
DAMPING_CHECK_TICKERS = ("NVDA", "MRVL", "PEP")
# Asset whose linear jump is recomputed at NVDA_A_PUBLISHED. Appendix B.
JUMP_CHECK_TICKER = "SPY"
