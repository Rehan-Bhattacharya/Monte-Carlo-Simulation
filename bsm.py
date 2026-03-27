
import numpy as np 
from scipy.stats import norm

def d1(S, X , r , T , Sigma):
    ''' Computing d1 Component of the BSM formula '''
    return (np.log(S/X) + (r + Sigma ** 2 * 0.5) * T)/(Sigma * np.sqrt(T))

def d2(S, X , r , T , Sigma):
    ''' Computing d2 component of the BSM formula '''
    return( d1(S, X , r , T , Sigma) - Sigma * np.sqrt(T))

def Call_price(S , X , r , T , Sigma):
    '''Computing BSM call option price '''
    D1 = d1(S , X , r , T , Sigma)
    D2 = d2(S , X , r , T , Sigma)
    return(S * norm.cdf(D1) - (X * np.exp(-r * T)) * norm.cdf(D2))

def Put_price(S , X , r , T , Sigma):
    '''Computing BSM put option price '''
    D1 = d1(S , X , r , T , Sigma)
    D2 = d2(S , X , r , T , Sigma)
    return((X * np.exp(-r * T) * (1 - norm.cdf(D2)) - S * (1 - norm.cdf(D1))))

# Quick Test  

if __name__ == "__main__":
    S     = 100                  # Stock price
    X     = 100                  # Strike price (at the money option)
    r     = 0.05                 # 5% risk free rate of return
    T     = 1                    # 1 year to maturity
    Sigma = 0.2                  # 20% volatility 

    print(f"Call Price: $ {Call_price(S , X , r , T , Sigma): .4f} ")
    print(f"Put Price : $ {Put_price(S , X , r , T , Sigma): .4f} ")

    # Put-Call Parity Check: C - P = S - X/e^rt
    C = Call_price(S , X , r , T , Sigma)
    P = Put_price(S , X , r , T , Sigma)
    LHS = C - P 
    RHS = S - X * np.exp( -r * T)
    print(f"\n Put-Call Parity Check: {LHS: .4f} ≈ {RHS: .4f} ✓ ")