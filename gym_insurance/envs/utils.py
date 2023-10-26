import logging
from typing import Dict

logger = logging.getLogger(__name__)

class Wallet:
    __slots__ = ['budget', 'spent']

    def __init__(self, budget: int) -> None:
        self._validate_budget(budget)
        self._budget: int = budget
        self.spent: float = 0

    @staticmethod
    def _validate_budget(budget: int) -> None:
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError(f"Budget should be a positive integer, not a {type(budget)}")

    def withdraw(self, value: float) -> None:
        # Inline validation and direct attribute access instead of property
        if value <= 0:
            raise ValueError("Withdrawals should be positive")
        if value >= self._budget - self.spent:
            raise ValueError("Insufficient balance for withdrawal")
        
        self.spent += value
        
        # Inline logging and local variable assignment
        spent, budget = self.spent, self._budget
        pct_budget = spent / budget
        logger.info(f"Current (%) Budget: {pct_budget}")

    @property
    def balance(self) -> float:
        return self._budget - self.spent

