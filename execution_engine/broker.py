from loguru import logger
from typing import Dict, Any

class MockBrokerAPI:
    """A simulated broker API for paper trading and backtesting."""
    def __init__(self, initial_balance=100000.0):
        self.balance = initial_balance
        self.positions: Dict[str, int] = {}
        logger.info(f"Initialized MockBrokerAPI with balance: ${self.balance}")

    def get_account_summary(self) -> Dict[str, Any]:
        return {
            "balance": self.balance,
            "positions": self.positions
        }

    def place_order(self, symbol: str, quantity: int, side: str, price: float) -> bool:
        """Mock order placement. Assuming instantaneous fill."""
        cost = quantity * price
        
        if side.upper() == "BUY":
            if self.balance >= cost:
                self.balance -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                logger.debug(f"BUY order filled: {quantity} {symbol} @ {price}")
                return True
            else:
                logger.warning(f"Insufficient funds for BUY order: {symbol}")
                return False
                
        elif side.upper() == "SELL":
            current_pos = self.positions.get(symbol, 0)
            if current_pos >= quantity:
                self.positions[symbol] -= quantity
                self.balance += cost
                logger.debug(f"SELL order filled: {quantity} {symbol} @ {price}")
                return True
            else:
                logger.warning(f"Insufficient position for SELL order: {symbol}")
                return False
                
        return False
