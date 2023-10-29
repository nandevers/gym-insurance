import logging
from typing import Dict

logger = logging.getLogger(__name__)

class Wallet:
    def __init__(self, budget: int) -> None:
        self._validate_budget(budget)
        self._budget: int = budget
        self._spent: float = 0
        self._current_budget =  {}
        self._current_case = 0 
        self._decisions = {}

    def reset(self):
        self._spent = 0

    @staticmethod
    def _validate_budget(budget: int) -> None:
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError(
                f"Budget should be a positive integer, not a {type(budget)}"
            )

    def withdraw(self, value: float) -> None:
        # Inline validation and direct attribute access instead of property
        if value <= 0:
            raise ValueError("Withdrawals should be positive")
        if value >= self._budget - self._spent:
            raise ValueError("Insufficient balance for withdrawal")

        self._spent += value
        logger.info(f"Current (%) Budget: {self.budget_spent_percentage}")

    @property
    def balance(self) -> float:
        return self._budget - self._spent
    
    @property
    def budget_spent_percentage(self) -> float:
        return self._spent / self._budget

    @property
    def current_budget(self) -> Dict:
        return self._current_budget
    
    @current_budget.setter
    def current_budget(self, value):
        self._current_budget[self.current_case] = value

    @property
    def current_case(self):
        return self._current_case

    @current_case.setter
    def current_case(self, value):
        self._current_case = value

    @property
    def decisions(self):
        return self._decisions
    
    @decisions.setter
    def decisions(self, values):
        self._decisions = dict.fromkeys(values)


    @property
    def current_case_decision(self):
        if self._current_case is None:
            raise ValueError("Current case is not set")
        if self._current_case not in self._decisions:
            raise ValueError("Current case does not exist in budget decisions")
        return self._decisions[self._current_case]

    @current_case_decision.setter
    def current_case_decision(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Decision value must be a number")
        if self._current_case is None:
            raise ValueError("Current case is not set")
        self._decisions[self._current_case] = value