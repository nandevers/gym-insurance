
import os
import shutil
import pandas as pd
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class Wallet:
    def __init__(self, budget: int, sample_index: list) -> None:
        self._validate_budget(budget)
        self._budget: int = budget
        self._current_case: Optional[int] = None
        self.sample_index = sample_index
        self.reset()

    def reset(self) -> None:
        self._spent = 0
        self._current_case = None
        self._current_budget: Dict[int, float] = dict.fromkeys(self.sample_index)
        self._budgets: Dict[int, float] = dict.fromkeys(self.sample_index)
        self._decisions: Dict[int, Any] = dict.fromkeys(self.sample_index)
        assert len(self._decisions.keys()) == len(self.sample_index)

    @staticmethod
    def _validate_budget(budget: int) -> None:
        if not isinstance(budget, int) or budget <= 0:
            raise ValueError(
                f"Budget should be a positive integer, not a {type(budget)}"
            )

    def withdraw(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Withdrawals should be positive")
        if value > self.balance:
            raise ValueError("Insufficient balance for withdrawal")

        self._spent += value
        if self._current_case is not None:
            self._current_budget[self._current_case] = self.budget_spent_percentage
            self._budgets[self._current_case] = self.balance
        logger.info(f"Current (%) Budget: {self.budget_spent_percentage}")

    def reject_withdrawal(self) -> None:
        self._budgets[self.current_case] = self.balance

    @property
    def balance(self) -> float:
        return self._budget - self._spent

    @property
    def budget_spent_percentage(self) -> float:
        return self._spent / self._budget

    @property
    def budgets(self) -> Dict[int, float]:
        return dict(self._budgets)

    @property
    def decisions(self) -> Dict[int, Any]:
        return dict(self._decisions)

    @property
    def current_budget(self) -> Optional[float]:
        if self._current_case is None:
            return None
        return self._current_budget.get(self._current_case)

    @property
    def current_case(self) -> Optional[int]:
        return self._current_case

    @current_case.setter
    def current_case(self, value: int) -> None:
        if value is None:
            raise TypeError("This should be an index from sample_index object")
        if value < 0:
            raise ValueError("Current case cannot be negative")
        self._current_case = value

    @property
    def current_decision(self) -> Optional[Any]:
        if self._current_case is None:
            return None
        return self._decisions.get(self._current_case)

    def add_decision(self, case: int, decision: Any) -> None:
        self._decisions[case] = decision

    def get_approved_cases(self) -> list:
        """
        Retrieves the case numbers of the approved cases.

        Returns:
            list: A list of case numbers for approved cases.
        """
        return list(filter(lambda case: self._decisions[case] == 1, self._decisions))

    def get_rejected_cases(self) -> list:
        """
        Retrieves the case numbers of the rejected cases.

        Returns:
            list: A list of case numbers for rejected cases.
        """
        return list(filter(lambda case: self._decisions[case] == 0, self._decisions))

    def get_approved_values(self) -> list:
        """
        Retrieves the values associated with approved cases.

        Returns:
            list: A list of values for approved cases.
        """
        return [
            value for case, value in self._budgets.items() if self._decisions[case] == 1
        ]

    def get_rejected_values(self) -> list:
        """
        Retrieves the values associated with rejected cases.

        Returns:
            list: A list of values for rejected cases.
        """
        return [
            value for case, value in self._budgets.items() if self._decisions[case] == 0
        ]

    def rollback_decision(self, case: Optional[int] = None) -> None:
        """
        Rolls back a decision for a given case or the current case if no case is specified.

        Args:
            case (Optional[int]): The case to roll back. If not specified, the current case is used.
        """
        if case is None:
            case = self._current_case

        if case is None or case not in self._decisions:
            raise ValueError("Invalid case provided for rollback.")

        # Reset the decision and any associated budget
        logger.info(f"Rolling back decision for case {case}.")
        self._decisions[case] = None
        self._current_budget[case] = None
        self._budgets[case] = None

        if self._current_case == case:
            self._current_case = None  # Reset the current case if it's the same case





def move_parquet_files(source_dir, destination_dir, budget):
    """
    Read all Parquet files inside each immediate subdirectory of source_dir,
    concatenate them into a single DataFrame, add a column 'episodes' based on
    the subdirectory name, and write them as a single file in destination_dir.
    The output file is named results_{budget}.parquet.
    
    :param source_dir: The directory to read files from.
    :param destination_dir: The directory to write the concatenated file to.
    :param budget: The budget value to include in the output filename.
    """
    # Ensure destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    
    # List immediate subdirectories in source_dir
    subdirs = [name for name in os.listdir(source_dir)
               if os.path.isdir(os.path.join(source_dir, name))]
    
    all_dfs = []
    
    for subdir in subdirs:
        subdir_path = os.path.join(source_dir, subdir)
        # Get all Parquet files in the subdirectory
        parquet_files = [os.path.join(subdir_path, f)
                         for f in os.listdir(subdir_path)
                         if f.endswith('.parquet')]
        
        if parquet_files:
            # Read and concatenate Parquet files
            dfs = [pd.read_parquet(pf) for pf in parquet_files]
            concatenated_df = pd.concat(dfs, ignore_index=True)
            # Add 'episodes' column based on subdir name
            concatenated_df['episodes'] = int(subdir)
            # Append to list of all DataFrames
            all_dfs.append(concatenated_df)
            print(f"Processed Parquet files for episode {subdir}")
        else:
            print(f"No Parquet files found in {subdir_path}")
    
    if all_dfs:
        # Concatenate all DataFrames from all subdirectories
        final_df = pd.concat(all_dfs, ignore_index=True)
        # Write final DataFrame to destination directory
        output_filename = f"results_{budget}.parquet"
        destination_file = os.path.join(destination_dir, output_filename)
        final_df.to_parquet(destination_file)
        print(f"Wrote concatenated Parquet file to {destination_file}")
    else:
        print("No Parquet files found in any subdirectories.")
