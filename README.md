# Monte Carlo Stock Price Simulation

A Monte Carlo simulation engine for stock price modelling, risk analysis, and options pricing- validated against BSM analytical prices using real AAPL data.

## Features
- GBM-based stock price simulation with fully vectorised NumPy (no loops over simulations)
- VaR and CVaR computed at 95% and 99% confidence levels
- Real historical drift and volatility derived from live yfinance data
- European call and put pricing under the risk-neutral measure
- Convergence analysis comparing Monte Carlo prices against BSM at 100 to 50,000 simulations
- Three interactive Plotly charts — price paths, final price distribution, confidence interval bands

## Project Structure
- Monte_Carlo.py — single-file project covering all five phases: simulation engine,risk metrics, real data integration, visualisations,
  and options pricing
- bsm.py — BSM analytical pricer used for Phase 5 validation

## Libraries Required
pip install numpy pandas scipy plotly yfinance

## How to Get Data
No manual download required — yfinance pulls historical prices automatically at runtime.
Change the ticker and lookback period inside run_phase3() to simulate any stock.

## Key Results
- Simulated 1,000 daily GBM paths for AAPL using 2 years of real historical data (March 2024 to March 2026) - derived annual drift of
  20.40% and volatility of 28.21%, versus hardcoded assumptions of 10% drift and 20% volatility i.e real parameters nearly tripled the
  95% VaR from $21.94 to $65.33 on a $253 stock.
- 95% CVaR of -42.46% (-$87.73) on real AAPL parameters, meaning in the worst 5% of simulated years the average loss exceeds $87 per
  share — significantly worse than the VaR threshold alone suggests, demonstrating why Basel III mandates CVaR over VaR for regulatory
  capital.
- Monte Carlo call price of $34.10 vs BSM analytical price of $34.52 at 10,000 simulations — difference of $0.42, sitting within the
  simulation standard error of $0.53, confirming the risk-neutral MC engine is statistically consistent with the closed-form solution
- Convergence table shows MC pricing error falling from $0.88 at 100 simulations to $0.13 at 50,000 - reflecting expected random
  sampling variation — and confirming that at sufficient scale, simulation-based pricing converges to the same answer as analytical
  methods
