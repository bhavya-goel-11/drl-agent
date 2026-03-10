import numpy as np
import pandas as pd
from typing import List

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate annualized Sharpe Ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(portfolio_values: List[float]) -> float:
    """Calculate Maximum Drawdown."""
    if not portfolio_values:
        return 0.0
    
    peak = portfolio_values[0]
    max_dd = 0.0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
            
    return max_dd
