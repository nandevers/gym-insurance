import os
from typing import Dict
import tensorflow as tf
from keras.callbacks import TensorBoard
import logging

# Create a custom logger
logger = logging.getLogger(__name__)

# Set level of logger
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('app.log')

# Set level of handlers
console_handler.setLevel(logging.WARNING)
file_handler.setLevel(logging.ERROR)

# Create formatters and add it to handlers
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
file_handler.setFormatter(file_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')




class Wallet:
    def __init__(self, budget: int) -> None:
        self.budget: int = self._validate_budget(budget)
        self.pct_budget: float = 0
        self.current_budget: Dict[str, float] = {}
        self.decisions: Dict[str, float] = {}
        self.spent: float = 0

    @staticmethod
    def _validate_budget(budget: int) -> bool:
        if not isinstance(budget, int):
            raise ValueError(f"Budget should be a {int} not a {type(budget)}")
        return budget

    def withdraw(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Withdrawals should be positive")
        if value >= self.balance:
            raise ValueError("Not allowed")

        self.spent += value
        if self.pct_budget <= 1 and self.spent <= self.budget:
            self.pct_budget = self.spent / self.budget
            logger.info(f"Current (%) Budget: {self.pct_budget}")
        else:
            raise ValueError(f"Balance {self.balance} does not allow withdrawal")

    @property
    def balance(self) -> float:
        return self.budget - self.spent
