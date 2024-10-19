import pandas as pd
import logging
from collections import namedtuple, OrderedDict, deque
import os
from typing import List, Optional

# Create log directory if it doesn't exist
log_dir = ".temp/logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'wallet.logs'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Define a named tuple for storing case info (budget, current budget, and decision history)
CaseInfo = namedtuple('CaseInfo', ['budget', 'current_budget', 'history'])

class Wallet:
    
    
    def __init__(self, initial_budget: float, sample_index: list) -> None:
        """Initialize the wallet with a given initial budget and cases."""
        self._validate_budget(initial_budget)
        self._initial_budget = initial_budget  # Initial budget to reset to
        self._budget = initial_budget  # Current working budget
        self._spent = 0

        # Use OrderedDict to maintain the order of insertion for decisions and budgets
        self._cases = OrderedDict()
        self.sample_index = sample_index

        # Initialize case histories and reset budget/spent
        self.reset()

        logging.info("Initialized Wallet with budget: %s and sample cases: %s", initial_budget, sample_index)

    def reset(self) -> None:
        """Reset the wallet, clearing all history and resetting the budget."""
        self._budget = self._initial_budget  # Reset to initial budget
        self._spent = 0  # Reset spent money
        self._cases.clear()  # Clear all previous decisions

        # Initialize case entries with empty history
        for case in self.sample_index:
            self._cases[case] = CaseInfo(budget=None, current_budget=None, history=[])

        logging.info("Reset Wallet. Initial budget restored: %s", self._initial_budget)

    @staticmethod
    def _validate_budget(budget: float) -> None:
        """Validate that the budget is a positive number."""
        if not isinstance(budget, (int, float)) or budget <= 0:
            logging.error("Invalid budget: %s. Budget must be a positive number.", budget)
            raise ValueError("Budget should be a positive number.")

    def decision(self, case: int, decision: int, value: float) -> None:
        """Make a decision (approve/reject) and assign a value (withdrawal amount or rejected value)."""
        if case not in self.sample_index:
            logging.error("Invalid case index: %s. Cannot make decision.", case)
            raise ValueError("Invalid case index.")

        # Get the current case info
        current_case_info = self._cases[case]

        # Auto-assign budget if not already assigned
        if current_case_info.budget is None:
            self._assign_case_budget(case, value)

        # Append decision to history
        decision_record = {'decision': decision, 'value': value}
        current_history = current_case_info.history + [decision_record]

        # If decision is 1 (approval), withdraw the value; if 0 (rejection), just log the rejection
        if decision == 1:
            self._withdraw(case, value)
        else:
            logging.info("Case %s rejected. No budget change.", case)

        # Update case info with new history and potentially reduced budget
        new_case_info = CaseInfo(budget=current_case_info.budget,
                                 current_budget=current_case_info.current_budget,
                                 history=current_history)
        self._cases[case] = new_case_info

        logging.info("Recorded decision for case %s: %s, value: %s", case, decision, value)

    def _assign_case_budget(self, case: int, amount: float) -> None:
        """Internal method to assign budget for a case if not yet set."""
        if amount > self.balance:
            logging.error("Insufficient funds to allocate %s to case %s. Current balance: %s", amount, case, self.balance)
        # Assign the budget to the case
        new_case_info = CaseInfo(budget=amount, current_budget=amount, history=[])
        self._cases[case] = new_case_info
        logging.info("Assigned budget of %s to case %s", amount, case)

    def _withdraw(self, case: int, amount: float) -> None:
        """Private method to withdraw a certain amount from a case's budget."""
        if amount <= 0:
            logging.error("Invalid withdrawal amount: %s. Must be positive.", amount)
            raise ValueError("Withdrawal amount must be positive.")
        
        if amount > self.balance:
            logging.error("Insufficient funds: Cannot withdraw %s. Current balance: %s", amount, self.balance)
            raise ValueError("Insufficient balance for withdrawal.")
        
        current_case_info = self._cases[case]
        if current_case_info.budget is None:
            logging.error("Cannot withdraw from case %s. No budget set.", case)
            raise ValueError("Cannot withdraw from case with no set budget.")
        
        # Deduct the amount from the case's budget
        new_current_budget = current_case_info.current_budget - amount
        new_case_info = CaseInfo(budget=current_case_info.budget,
                                 current_budget=new_current_budget,
                                 history=current_case_info.history)
        self._cases[case] = new_case_info

        # Update the total spent
        self._spent += amount

        logging.info("Withdrew %s from case %s. New case budget: %s, Remaining balance: %s", 
                     amount, case, new_current_budget, self.balance)

    @property
    def balance(self) -> float:
        """Return the remaining balance of the wallet."""
        return self._budget - self._spent

    @property
    def initial_budget(self) -> float:
        """Return the initial budget the wallet started with."""
        return self._initial_budget

    @property
    def budget_spent_percentage(self) -> float:
        """Return the percentage of the budget that has been spent."""
        if self._budget == 0:
            return 0.0
        return (self._spent / self._budget) * 100

    def get_case_history(self, case: int) -> list:
        """Return the decision history for a specific case."""
        if case not in self._cases:
            logging.error("Invalid case index: %s. Cannot retrieve history.", case)
            raise ValueError("Invalid case index.")
        
        return self._cases[case].history

    def get_approved_cases(self) -> List[int]:
        """Return a list of cases where the decision was approved (decision = 1)."""
        return self.get_cases_by_decision(1)

    def get_rejected_cases(self) -> List[int]:
        """Return a list of cases where the decision was rejected (decision = 0)."""
        return self.get_cases_by_decision(0)

    def get_approved_values(self) -> List[Optional[float]]:
        """Return a list of approved values (decision = 1) from the case histories."""
        return self.get_values_by_decision(1)

    def get_rejected_values(self) -> List[Optional[float]]:
        """Return a list of rejected values (decision = 0) from the case histories."""
        return self.get_values_by_decision(0)

    def get_values_by_decision(self, decision_value: int) -> List[Optional[float]]:
        """Retrieve values by decision type from case histories."""
        values = []
        for case, case_info in self._cases.items():
            values.extend([record['value'] for record in case_info.history if record['decision'] == decision_value])
        return values

    def get_cases_by_decision(self, decision_value: int) -> List[int]:
        """Retrieve case indices by decision value (1 for approved, 0 for rejected)."""
        cases = []
        for case, case_info in self._cases.items():
            if any(record['decision'] == decision_value for record in case_info.history):
                cases.append(case)
        return cases

    def get_decisions_dataframe(self) -> pd.DataFrame:
        """Generate a DataFrame from the decision history of all cases."""
        # Create a dictionary to hold decisions for each case
        decisions_data = {}

        for case, case_info in self._cases.items():
            # Flatten the decision history for each case
            for record in case_info.history:
                decisions_data[case] = {
                    "decision": record['decision'],
                    "value": record['value']
                }
        
        # Convert the dictionary to a DataFrame
        results_data = pd.DataFrame.from_dict(decisions_data, orient="index")
        return results_data