
import pandas as pd
import numpy as np 
import scipy.stats as stat
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import bsm

# Phase 1 : Core Simulation Engine

def simulate_gbm ( S0 , mu , sigma , T , dt , n_simulations , seed=42):
    """
    Simulating the stock prices using the Geometric Brownian Motion (GBM)
    Paramters:
    S0           : float  — Initial stock price
    mu           : float  — Annual drift  (e.g. 0.10 for 10%)
    sigma        : float  — Annual volatility (e.g. 0.20 for 20%)
    T            : float  — Time horizon in years (e.g. 1.0 for 1 year)
    dt           : float  — Time step size (e.g. 1/252 for daily steps)
    n_simulations: int    — Number of Monte Carlo paths to simulate
    seed         : int    — Random seed for reproducibility ( seed=42 is the default value for a parameter)
    """
    np.random.seed(seed)                                      # will generate different random numbers every single run.

    # Building the time grid 
    n_steps = int(T / dt)                                     # total number of time steps
    time_grid = np.linspace(0 , T , n_steps+1)            

    # Pre- alloacting the price matrix 
    price_paths = np.zeros((n_steps + 1 , n_simulations))       # (n_steps + 1) rows × n_simulations columns. Also We are adding 1 row to include the starting price S0
    price_paths[0] = S0

    # Generating all random shocks at once (no loops)
    Z = np.random.standard_normal((n_steps, n_simulations))     # each column is one simulation's worth of randomness

    # Applying the GBM formula 
      # We are breaking the formula into two components : drift term and diffusion term 
    drift_term = (mu - 0.5 * sigma**2) * dt
    diffusion_term = sigma * np.sqrt(dt) * Z

    # Combining the two components: Computing daily return component 
    daily_return = np.exp( drift_term + diffusion_term )

    # Building price paths
    for i in range ( 1 , n_steps+1 ):
        price_paths[i] = price_paths[i-1] * daily_return[i-1]
    
    return price_paths, time_grid

def run_phase1():
    """
    Simulating GBM paths and displaying basic results 
    """
    print("=" * 60)
    print("PHASE 1 — CORE SIMULATION ENGINE")
    print("=" * 60)

    # Simulation Parameters
    S0            = 100.0    # Starting price ($100)
    mu            = 0.10     # 10% annual drift
    sigma         = 0.20     # 20% annual volatility
    T             = 1.0      # 1 year
    dt            = 1/252    # Daily steps (252 trading days)
    n_simulations = 1000     # 1,000 paths

    # Running the simulation 
    price_paths, time_grid = simulate_gbm(S0, mu, sigma, T, dt, n_simulations)

    print(f"\nSimulation Parameters:")
    print(f"  Starting Price (S0) : ${S0:.2f}")
    print(f"  Annual Drift (μ)    : {mu*100:.1f}%")
    print(f"  Annual Volatility (σ): {sigma*100:.1f}%")
    print(f"  Time Horizon (T)    : {T} year")
    print(f"  Time Steps (dt)     : 1/252 (daily)")
    print(f"  Simulations         : {n_simulations:,}")

    # Shape Confirmation - Checking the dimension of the matrix
    print(f"\n Price Matrix Shape   : {price_paths.shape}")
    print(f"  {price_paths.shape[0]} rows (time steps including t=0)")
    print(f"  {price_paths.shape[1]} columns (simulations)")

    # Sanity checks on final prices 
    final_prices = price_paths[-1]                   # last row = prices at T

    print(f"\nFinal Price Statistics (at T = {T} year):")
    print(f"  Mean Final Price    : ${final_prices.mean():.2f}")
    print(f"  Median Final Price  : ${np.median(final_prices):.2f}")
    print(f"  Std Dev             : ${final_prices.std():.2f}")
    print(f"  Min                 : ${final_prices.min():.2f}")
    print(f"  Max                 : ${final_prices.max():.2f}")

    # Theoretical check 
    # E[S(T)] = S0 * exp(mu * T)               (GBM property- Mathematically proved)
    theoretical_mean = S0 * np.exp(mu * T)
    print(f"\nTheoretical E[S(T)]  : ${theoretical_mean:.2f}")
    print(f"Simulated Mean       : ${final_prices.mean():.2f}")
    print(f" These should be close. Difference: ${abs(final_prices.mean() - theoretical_mean):.2f}")

    # Preview: first 5 simulations, first 5 and last 5 steps 
    print(f"\nFirst 3 paths — first 5 days:")
    preview_df = pd.DataFrame(
        price_paths[:5, :3],
        index=[f"t={i}" for i in range(5)],
        columns=[f"Sim {i+1}" for i in range(3)]
    )
    print(preview_df.round(4).to_string())

    print(f"\nFirst 3 paths — final price (day 252):")
    end_df = pd.DataFrame(
        price_paths[-1:, :3],
        index=["t=252"],
        columns=[f"Sim {i+1}" for i in range(3)]
    )
    print(end_df.round(4).to_string())

    return price_paths, time_grid


# Phase 2: Analysing Outcomes

def analyse_outcomes (price_paths, S0 , confidence_levels =[0.95, 0.99]):
    """
    Analysing the distribution of Simulated Final Prices
    Paramters: 
    price_paths              = np.ndarray - full price matrix from simulate_gbm()
    S0                       = float      - initial stock price
    confidence_levels        = list       — VaR/CVaR confidence levels (default 95% and 99%)
    """

    # Computing Log returns
    final_prices = price_paths[-1]                        # 1000 final prices at T
    log_returns = np.log(final_prices / S0)               # ln(S(T)/S(0)) for each sim

    # Computing Basic Statistics
    mean_price = final_prices.mean()
    median_price = np.median(final_prices)
    std_price = final_prices.std()
    mean_return = log_returns.mean()
    std_return = log_returns.std()

    # Computing Skewness and Kurtosis 
    skewness = stat.skew(log_returns)
    kurtosis = stat.kurtosis(log_returns)

    # Computing VaR and CVaR
    VaR_results  = {}
    CVaR_results = {}

    for i in confidence_levels:
        alpha = 1 - i
     
        # VaR: the percentile cutoff of log returns
        var = np.percentile( log_returns , alpha * 100)

        #CVaR: Average of all the return below the cutoff of log returns
        tail_returns = log_returns[log_returns <= var]
        cvar = tail_returns.mean()

        VaR_results [i]  = var
        CVaR_results [i] = cvar

    # Probability of profit and loss
    prob_profit = (final_prices > S0).mean() * 100             # % of sims ending above S0
    prob_loss   = (final_prices < S0).mean() * 100             # % of sims ending below S0

    # Price Percentile table
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    price_quantiles = np.percentile(final_prices, percentiles)

    results = {
        "final_prices"   : final_prices,
        "log_returns"    : log_returns,
        "mean_price"     : mean_price,
        "median_price"   : median_price,
        "std_price"      : std_price,
        "mean_return"    : mean_return,
        "std_return"     : std_return,
        "skewness"       : skewness,
        "kurtosis"       : kurtosis,
        "var"            : VaR_results,
        "cvar"           : CVaR_results,
        "prob_profit"    : prob_profit,
        "prob_loss"      : prob_loss,
        "percentiles"    : percentiles,
        "price_quantiles": price_quantiles,
    }

    return results

def run_phase2(price_paths, S0):
    """
    Analysing and displaying outcome statistics.
    """
    print("\n")
    print("=" * 60)
    print("PHASE 2 — ANALYSING OUTCOMES")
    print("=" * 60)

    results = analyse_outcomes(price_paths, S0)

    # Printing descriptive statistics 
    print(f"\n Descriptive Statistics — Final Prices:")
    print(f"   Mean Price        : ${results['mean_price']:.2f}")
    print(f"   Median Price      : ${results['median_price']:.2f}")
    print(f"   Std Deviation     : ${results['std_price']:.2f}")
    print(f"   Skewness          : {results['skewness']:.4f}")
    print(f"   Excess Kurtosis   : {results['kurtosis']:.4f}")

    print(f"\n Descriptive Statistics — Log Returns:")
    print(f"   Mean Log Return   : {results['mean_return']:.4f} ({results['mean_return']*100:.2f}%)")
    print(f"   Std Log Return    : {results['std_return']:.4f} ({results['std_return']*100:.2f}%)")

    # Printing VaR and CVaR 
    print(f"\n Risk Metrics:")
    for i in [0.95, 0.99]:
        var  = results['var'][i]
        cvar = results['cvar'][i]
        print(f"\n   {int(i*100)}% Confidence Level:")
        print(f"   VaR              : {var:.4f}  ({var*100:.2f}%)")
        print(f"   CVaR             : {cvar:.4f} ({cvar*100:.2f}%)")
        print(f"   In dollar terms  : VaR = ${S0 * (np.exp(var)-1):.2f}  |  CVaR = ${S0 * (np.exp(cvar)-1):.2f}")

    # Printing probability analysis 
    print(f"\n Probability Analysis:")
    print(f"   Probability of Profit : {results['prob_profit']:.1f}%")
    print(f"   Probability of Loss   : {results['prob_loss']:.1f}%")

    # Printing percentile table 
    print(f"\n Price Percentile Table:")
    print(f"   {'Percentile':<15} {'Price':>10}")
    print(f"   {'-'*25}")
    for p, price in zip(results['percentiles'], results['price_quantiles']):
        print(f"   {str(p)+'th':<15} ${price:>9.2f}")

    return results


# Phase3 : Real Data Integration 

import yfinance as yf

def stock_parameters (ticker , period = '2y' ):
    """
    Pulls historical stock data and calculates drift and volatility 
    Parameters:
    ticker :    str    - stock ticker symbol 
    period :    str    - historical lookback period 
    """

    print(f"\n  Fetching data for {ticker}... ")

    # Downloading historical price data and extracting closing price 
    stock = yf.Ticker(ticker)
    hist_data = stock.history(period = period)
    prices = hist_data["Close"]

    # Computing daily log returns 
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Computing daily statistics and annualising them 
    daily_mean = log_returns.mean()    
    daily_std  = log_returns.std()     

    mu = daily_mean * 252              # annualised drift
    sigma = daily_std * np.sqrt(252)   # annualised volatility  

    # Getting the latest price as S0
    S0 = prices.iloc[-1]

    return mu, sigma , S0, hist_data, log_returns

def run_phase3 ( ticker = "AAPL", period = "2y", n_simulations = 1000):
    """
    Pulling real world data, computing parameters, running simulation
    """
    print("\n")
    print("=" * 60)
    print("PHASE 3 — REAL DATA INTEGRATION")
    print("=" * 60)

    # Getting real parameters from market data 
    mu, sigma, S0, hist_data, log_returns = stock_parameters(ticker, period)

    # Displaying the data 
    print(f"\n   Ticker           : {ticker}")
    print(f"   Data Period        : {period}")
    print(f"   Data Points        : {len(log_returns)} trading days")
    print(f"   Date Range         : {hist_data.index[0].date()} - {hist_data.index[-1].date()}")
    print(f"   Latest Price (S0)  : ${S0:.2f}")

    print(f"\n   Historical Parameters:")
    print(f"   Annual Drift (μ)   : {mu*100:.2f}%")
    print(f"   Annual Vol   (σ)   : {sigma*100:.2f}%")

    # Comparing with Phase1 assumptions
    print(f"\n  Phase 1 VS Phase 3 Comparison : ")
    print(f"   {'Parameter' : <20} {'Phase 1 (Assumed)' : <22} {'Phase 3 (Real)' : <15}")
    print(f"   {'-'*57}")
    print(f"   {'Drift (μ)' : <20} {'10.00%' : <22} {mu*100:.2f}%")
    print(f"   {'Volatility (σ)':<20} {'20.00%':<22} {sigma*100:.2f}%")
    print(f"   {'Start Price (S0)':<20} {'$100.00':<22} ${S0:.2f}")

    # Running the simulation with real parameters
    print(f"\n   Running Monte Carlo with real parameters...")
    T  = 1.0
    dt = 1/252

    price_paths, time_grid = simulate_gbm(S0, mu, sigma, T, dt, n_simulations)

    # Analysing outcomes with real parameters and printing results
    results = analyse_outcomes(price_paths, S0)

    print(f"\n   Simulation Results ({n_simulations:,} paths, T = {T} year):")
    print(f"   Mean Final Price   : ${results['mean_price']:.2f}")
    print(f"   Median Final Price : ${results['median_price']:.2f}")
    print(f"   Std Deviation      : ${results['std_price']:.2f}")

    print(f"\n   Risk Metrics:")
    for i in [0.95, 0.99]:
        var  = results['var'][i]
        cvar = results['cvar'][i]
        print(f"\n   {int(i*100)}% Confidence Level:")
        print(f"   VaR              : {var*100:.2f}%  (${S0 * (np.exp(var)-1):.2f})")
        print(f"   CVaR             : {cvar*100:.2f}% (${S0 * (np.exp(cvar)-1):.2f})")

    print(f"\n   Probability Analysis:")
    print(f"   Probability of Profit : {results['prob_profit']:.1f}%")
    print(f"   Probability of Loss   : {results['prob_loss']:.1f}%")

    print(f"\n   Price Percentile Table:")
    print(f"   {'Percentile':<15} {'Price':>10}")
    print(f"   {'-'*25}")
    for p, price in zip(results['percentiles'], results['price_quantiles']):
        print(f"   {str(p)+'th':<15} ${price:>9.2f}")

    return price_paths, time_grid, results, S0, mu, sigma

# Phase4 : Visualization

def plot_price_paths (price_paths, time_grid, S0, ticker = "Stock", n_paths = 50):
    """
    Chart1 : Plotting a sample of simulated price paths 
    Parameters:
    price_paths  : np.ndarray - full price matrix (253 * n_simulations)
    time-grid    : np.ndarray - time points from 0 to T
    S0           : float      - initial stock price
    ticker       : string     - stock ticker for chart title 
    n_paths      : int        - number of paths to display (default = 50)
    """

    print("\n Building Chart1 : Simulated Price paths... ")

    fig = go.Figure()

    # We are taking 50 paths randomly out of the 1000 simulations and plotting them. 
    # Plotting all 1000 paths would be too cluttered - 50 shows the pattern clearly. 

    np.random.seed(42)
    sample_indices = np.random.choice(price_paths.shape[1], n_paths, replace= False )
    
    for i in sample_indices:
        fig.add_trace(go.Scatter(
            x = time_grid,
            y = price_paths[ :, i],
            mode = 'lines',
            line = dict(width = 0.8, color = 'royalblue'),
            opacity = 0.3,
            showlegend = False                                       
        ))
    
    # Adding median path 
    median_path = np.median(price_paths, axis=1)   # median across simulations at each time step
    fig.add_trace(go.Scatter(
        x=time_grid,
        y=median_path,
        mode="lines",
        line=dict(width=2.5, color="red"),
        name="Median Path"
    ))

    # Addinf starting price line 
    fig.add_hline(
        y=S0,
        line_dash="dash",
        line_color="green",
        annotation_text=f"S0 = ${S0:.2f}",
        annotation_position="right"
    )

    fig.update_layout(
        title=f"{ticker} — Monte Carlo Simulated Price Paths ({n_paths} paths shown)",
        xaxis_title="Time (Years)",
        yaxis_title="Stock Price ($)",
        template="plotly_dark",
        height=550
    )

    fig.show()

def plot_final_distribution (results, S0, ticker = "Stock"):
    """
    Chart2 : Creating a histogram of final prices with VaR and CVaR
    Parameters:
    results : dict  — output from analyse_outcomes()
    S0      : float — initial stock price
    ticker  : str   — stock ticker for chart title
    """

    print("\n Building Chart2 : Final Price Distribution... ")

    final_prices = results["final_prices"]
    var_95       = results['var'][0.95]
    cvar_95      = results['cvar'][0.95]

    # Converting log return thresholds back to price levels
    var_price  = S0 * np.exp(var_95)
    cvar_price = S0 * np.exp(cvar_95)
    
    fig = go.Figure()

    # Histogram of final prices
    fig.add_trace(go.Histogram(
        x=final_prices,
        nbinsx=60,
        name="Final Prices",
        marker_color="royalblue",
        opacity=0.75
    ))

    # VaR line
    fig.add_vline(
        x=var_price,
        line_dash="dash",
        line_color="orange",
        line_width=2,
        annotation_text=f"95% VaR: ${var_price:.2f}",
        annotation_position="top left",
        annotation_font_color="orange"
    )

    # CVaR line
    fig.add_vline(
        x=cvar_price,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"95% CVaR: ${cvar_price:.2f}",
        annotation_position="top left",
        annotation_font_color="red"
    )

    # Starting price line
    fig.add_vline(
        x=S0,
        line_dash="dot",
        line_color="green",
        line_width=2,
        annotation_text=f"S0: ${S0:.2f}",
        annotation_position="top right",
        annotation_font_color="green"
    )

    fig.update_layout(
        title=f"{ticker} — Distribution of Final Simulated Prices",
        xaxis_title="Final Stock Price ($)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=550,
        showlegend=True
    )

    fig.show()

def plot_confidence_bands (price_paths, time_grid, S0 , ticker = "Stock"):
    """
    Chart3 : Plotting confidence interval bands over time 
    Parameters:
    price_paths : np.ndarray — full price matrix (253 x n_simulations)
    time_grid   : np.ndarray — time points from 0 to T
    S0          : float      — initial stock price
    ticker      : str        — stock ticker for chart title
    """

    print(" Building Chart3 : Confidence Interval Charts... ")

    # Computing percentile bands at every time steps
    # axis = 1 means across all simulations for each row.
    p5  = np.percentile(price_paths, 5,  axis=1)
    p25 = np.percentile(price_paths, 25, axis=1)
    p50 = np.percentile(price_paths, 50, axis=1)
    p75 = np.percentile(price_paths, 75, axis=1)
    p95 = np.percentile(price_paths, 95, axis=1)

    fig = go.Figure()

    # Outer band : 5th to 95th percentile 
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_grid, time_grid[::-1]]),   # forward then backward for fill
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself",
        fillcolor="rgba(65, 105, 225, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),           # invisible border
        name="5th–95th Percentile"
    ))

    # Inner band : 25th to 75th percentile (shaded)
    fig.add_trace(go.Scatter(
        x=np.concatenate([time_grid, time_grid[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself",
        fillcolor="rgba(65, 105, 225, 0.30)",
        line=dict(color="rgba(255,255,255,0)"),
        name="25th–75th Percentile"
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=time_grid,
        y=p50,
        mode="lines",
        line=dict(width=2.5, color="white"),
        name="Median (50th)"
    ))

    # 5th and 95th boundary lines 
    fig.add_trace(go.Scatter(
        x=time_grid, y=p95,
        mode="lines",
        line=dict(width=1.2, color="royalblue", dash="dot"),
        name="95th Percentile"
    ))

    fig.add_trace(go.Scatter(
        x=time_grid, y=p5,
        mode="lines",
        line=dict(width=1.2, color="tomato", dash="dot"),
        name="5th Percentile"
    ))

    # Starting price line 
    fig.add_hline(
        y=S0,
        line_dash="dash",
        line_color="green",
        annotation_text=f"S0 = ${S0:.2f}",
        annotation_position="right"
    )

    fig.update_layout(
        title=f"{ticker} — Monte Carlo Confidence Interval Bands",
        xaxis_title="Time (Years)",
        yaxis_title="Stock Price ($)",
        template="plotly_dark",
        height=550
    )

    fig.show()


def run_phase4(price_paths, time_grid, results, S0, ticker="Stock"):
    """
    Run Phase 4: Generate all three visualisations.
    """
    print("\n")
    print("=" * 60)
    print("PHASE 4 — VISUALISATIONS")
    print("=" * 60)

    plot_price_paths(price_paths, time_grid, S0, ticker)
    plot_final_distribution(results, S0, ticker)
    plot_confidence_bands(price_paths, time_grid, S0, ticker)

    print(f"\n   All 3 charts generated successfully.")


# Phase5 : Options Princing via Monte Carlo 

def monte_carlo_option_pricing ( S0, X , r , T , Sigma , n_simulations , option_type = 'call' , seed = 42 ):
    """
    Here we are princing an European option using Monte Carlo Simulation.
    We have already done option pricing in out BSM project. 
    Here we are using an another method to pricing options and compare the outputs.
    Parameters:
    S0           : float  — current stock price
    K            : float  — strike price
    r            : float  — risk free rate (annual)
    sigma        : float  — volatility (annual) — from real data in Phase 3
    T            : float  — time to expiry in years
    n_simulations: int    — number of simulations
    option_type  : str    — "call" or "put"
    seed         : int    — random seed for reproducibility
    """

    # Simulating under risk neutral conditions 
    dt = 1/252
    price_paths, _ = simulate_gbm( S0 , r , Sigma , T , dt , n_simulations , seed)

    # Extracting final prices and computing payoffs
    final_prices = price_paths[-1]

    if option_type == "call":
        payoffs = np.maximum(final_prices - X, 0)   # max(S(T) - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(X - final_prices, 0)   # max(K - S(T), 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Disocunting average payoff back to today 
    discount_factor = np.exp(-r * T)
    price           = payoffs.mean() * discount_factor

    # Calculating standard deviation. (Lower standard error - more reliable estimation)
    std_error = (payoffs.std() / np.sqrt(n_simulations)) * discount_factor

    return price, std_error , payoffs , final_prices

def run_phase5 ( S0 , Sigma , ticker = "AAPL"):
    """
    Pricing options via Monte Carlo simulation and comparing with BSM 
    """
    print("\n")
    print("=" * 60)
    print("PHASE 5 — OPTIONS PRICING VIA MONTE CARLO")
    print("=" * 60)

    # Parameters:
    X   = S0           # at-the-money option (strike = current price)
    r   = 0.05         # 5% risk free rate (approximate current rate)
    T   = 1.0          # 1 year to expiry
    n_simulations = 10000   # more simulations = more accurate pricing

    print(f"\n   Option Parameters:")
    print(f"   Underlying         : {ticker}")
    print(f"   Current Price (S0) : ${S0:.2f}")
    print(f"   Strike Price  (X)  : ${X:.2f}  (at-the-money)")
    print(f"   Risk Free Rate (r) : {r*100:.1f}%")
    print(f"   Volatility    (σ)  : {Sigma*100:.2f}%  (from Phase 3 real data)")
    print(f"   Time to Expiry (T) : {T} year")
    print(f"   Simulations        : {n_simulations:,}")

    # Monte Carlo pricing 
    print("\n   Running Monte Carlo pricing...")
    
    mc_call , call_se , call_payoffs , _ = monte_carlo_option_pricing( S0 , X , r , T , Sigma , n_simulations , option_type= "call")
    mc_put , put_se , put_payoffs , _    = monte_carlo_option_pricing( S0 , X , r , T , Sigma , n_simulations , option_type="put")

    # BSM Analytical Pricing 
    print("   Running BSM analytical pricing...")
    bsm_call = bsm.Call_price(S0, X, r, T, Sigma)
    bsm_put  = bsm.Put_price(S0, X, r, T, Sigma)

    print(f"\n   {'='*54}")
    print(f"   {'':20} {'Call Option':>15} {'Put Option':>15}")
    print(f"   {'='*54}")
    print(f"   {'Monte Carlo Price':<20} ${mc_call:>13.4f} ${mc_put:>13.4f}")
    print(f"   {'BSM Price':<20} ${bsm_call:>13.4f} ${bsm_put:>13.4f}")
    print(f"   {'Difference':<20} ${abs(mc_call-bsm_call):>13.4f} ${abs(mc_put-bsm_put):>13.4f}")
    print(f"   {'Std Error':<20} ${call_se:>13.4f} ${put_se:>13.4f}")
    print(f"   {'='*54}")

    # Put-Call parity check 
    mc_parity  = mc_call  - mc_put
    bsm_parity = bsm_call - bsm_put
    theoretical_parity = S0 - X * np.exp(-r * T)

    print(f"\n   Put-Call Parity Check (Call - Put = S0 - K×e^(-rT)):")
    print(f"   Theoretical        : ${theoretical_parity:.4f}")
    print(f"   BSM                : ${bsm_parity:.4f}")
    print(f"   Monte Carlo        : ${mc_parity:.4f}")

    # Convergence Analysis - Showing how accuracy increases with more simulations
    print(f"\n   Convergence Analysis — MC Call Price vs BSM:")
    print(f"   {'Simulations':<15} {'MC Price':>12} {'BSM Price':>12} {'Error':>10}")
    print(f"   {'-'*50}")

    for i in [100, 500, 1000, 5000, 10000, 50000]:
        mc_price, se, _, _ = monte_carlo_option_pricing(
            S0, X, r, T,Sigma , i, option_type="call")
        error = abs(mc_price - bsm_call)
        print(f"   {i:<15,} ${mc_price:>11.4f} ${bsm_call:>11.4f} ${error:>9.4f}")

    return mc_call, mc_put, bsm_call, bsm_put




# Quick test
if __name__ == "__main__":
    price_paths, time_grid = run_phase1()
    results = run_phase2(price_paths, S0=100.0)
    price_paths_real, time_grid_real, results_real, S0_real, mu_real, sigma_real = run_phase3(ticker="AAPL", period="2y")
    run_phase4(price_paths_real, time_grid_real, results_real, S0_real, ticker="AAPL")
    run_phase5(S0_real, sigma_real, ticker="AAPL")










