import unittest
from gym_insurance.envs.utils import Wallet


class TestWallet(unittest.TestCase):
    def setUp(self):
        """Set up a Wallet instance with a budget of 1000 and a sample index list for testing."""
        self.wallet = Wallet(budget=1000, sample_index=[0, 1, 2, 3, 4])

    def test_initial_setup(self):
        """Test initial values of the Wallet."""
        self.assertEqual(self.wallet.balance, 1000)
        self.assertEqual(self.wallet.budget_spent_percentage, 0)
        self.assertIsNone(self.wallet.current_case)
        self.assertEqual(
            self.wallet.budgets, {0: None, 1: None, 2: None, 3: None, 4: None}
        )

    def test_withdraw(self):
        """Test the withdrawal of funds for an approved case."""
        self.wallet.current_case = 0
        self.wallet.withdraw(100)
        self.wallet.add_decision(0, 1)  # Approve case 0
        self.assertEqual(self.wallet.balance, 900)
        self.assertEqual(self.wallet.budget_spent_percentage, 0.1)

    def test_withdraw_invalid_amount(self):
        """Test withdrawal with invalid values."""
        with self.assertRaises(ValueError):
            self.wallet.withdraw(-100)

        with self.assertRaises(ValueError):
            self.wallet.withdraw(0)

        with self.assertRaises(ValueError):
            self.wallet.withdraw(1200)  # More than balance

    def test_add_decision(self):
        """Test adding decisions to cases."""
        self.wallet.add_decision(1, 1)  # Approve case 1
        self.wallet.add_decision(2, 0)  # Reject case 2

        self.assertEqual(self.wallet.decisions[1], 1)
        self.assertEqual(self.wallet.decisions[2], 0)

    def test_get_approved_cases(self):
        """Test retrieval of approved cases."""
        self.wallet.add_decision(0, 1)
        self.wallet.add_decision(1, 1)
        self.wallet.add_decision(2, 0)

        approved_cases = self.wallet.get_approved_cases()
        self.assertListEqual(approved_cases, [0, 1])

    def test_get_rejected_cases(self):
        """Test retrieval of rejected cases."""
        self.wallet.add_decision(0, 0)
        self.wallet.add_decision(1, 0)
        self.wallet.add_decision(2, 1)

        rejected_cases = self.wallet.get_rejected_cases()
        self.assertListEqual(rejected_cases, [0, 1])

    def test_get_approved_values(self):
        """Test retrieving values for approved cases."""
        # Withdraw and approve for case 0
        self.wallet.current_case = 0
        self.wallet.withdraw(100)
        self.wallet.add_decision(0, 1)  # Approve case 0

        # Withdraw but reject for case 1
        self.wallet.current_case = 1
        self.wallet.reject_withdrawal()
        self.wallet.add_decision(1, 0)  # Reject case 1

        # Withdraw and approve for case 2
        self.wallet.current_case = 2
        self.wallet.withdraw(50)
        self.wallet.add_decision(2, 1)  # Approve case 2

        # Expected approved values are:
        # - Case 0: 1000 - 100 = 900
        # - Case 2: 900 - 50 = 850
        approved_values = self.wallet.get_approved_values()
        self.assertListEqual(approved_values, [900, 850])

    def test_get_rejected_values(self):
        """Test retrieving values for rejected cases."""
        # Withdraw and reject for case 0
        self.wallet.current_case = 0
        self.wallet.reject_withdrawal()
        self.wallet.add_decision(0, 0)  # Reject case 0

        # Withdraw and approve for case 1
        self.wallet.current_case = 1
        self.wallet.withdraw(200)
        self.wallet.add_decision(1, 1)  # Approve case 1

        # Withdraw and reject for case 2
        self.wallet.current_case = 2
        self.wallet.reject_withdrawal()
        self.wallet.add_decision(2, 0)  # Reject case 2

        # Expected rejected values are:
        # - Case 0: Balance after withdrawal = 900
        # - Case 2: Balance after withdrawal = 650 (since it follows case 1 approval)
        rejected_values = self.wallet.get_rejected_values()
        self.assertListEqual(rejected_values, [1000, 800])

    def test_rollback_decision(self):
        """Test rolling back a decision."""
        # Set initial decisions
        self.wallet.current_case = 0
        self.wallet.withdraw(100)
        self.wallet.add_decision(0, 1)

        self.wallet.current_case = 1
        self.wallet.withdraw(200)
        self.wallet.add_decision(1, 0)

        # Rollback decision for case 1
        self.wallet.rollback_decision(1)

        self.assertIsNone(self.wallet.decisions[1])
        self.assertIsNone(self.wallet.budgets[1])
        self.assertIsNone(self.wallet.current_budget)

    def test_rollback_current_case(self):
        """Test rolling back the decision for the current case."""
        self.wallet.current_case = 2
        self.wallet.withdraw(150)
        self.wallet.add_decision(2, 1)

        # Rollback decision for the current case
        self.wallet.rollback_decision()

        self.assertIsNone(self.wallet.decisions[2])
        self.assertIsNone(self.wallet.budgets[2])
        self.assertIsNone(self.wallet.current_case)

    def test_invalid_rollback(self):
        """Test rollback with invalid case."""
        with self.assertRaises(ValueError):
            self.wallet.rollback_decision(10)  # Case 10 does not exist


if __name__ == "__main__":
    unittest.main()
