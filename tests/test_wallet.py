import unittest
from gym_insurance.envs.wallet import Wallet

class TestWallet(unittest.TestCase):

    def setUp(self):
        """Set up a Wallet instance before each test."""
        self.wallet = Wallet(1000.0, [0, 1, 2])

    def test_initial_budget(self):
        """Test if the initial budget is correctly set."""
        self.assertEqual(self.wallet.initial_budget, 1000.0)

    def test_add_decision_approval(self):
        """Test if adding an approval decision correctly deducts the amount."""
        self.wallet.decision(case=0, decision=1, value=500.0)  # Approve case 0 with 500
        self.assertEqual(self.wallet.balance, 500.0)  # 1000 - 500 = 500
        self.assertEqual(self.wallet.budget_spent_pct, 50.0)

    def test_add_decision_rejection(self):
        """Test if adding a rejection decision doesn't affect the balance."""
        self.wallet.decision(case=1, decision=0, value=300.0)  # Reject case 1 with 300
        self.assertEqual(self.wallet.balance, 1000.0)  # No balance reduction
        self.assertEqual(self.wallet.budget_spent_pct, 0.0)

    def test_case_history_tracking(self):
        """Test if decisions are correctly added to the case history."""
        self.wallet.decision(case=0, decision=1, value=543.23)
        history = self.wallet.get_case_history(0)
        self.assertEqual(history, [{'decision': 1, 'value': 543.23}])

        self.wallet.decision(case=0, decision=0, value=52.21)
        history = self.wallet.get_case_history(0)
        self.assertEqual(history, [{'decision': 1, 'value': 543.23}, {'decision': 0, 'value': 52.21}])

    def test_error_on_insufficient_funds(self):
        """Test if trying to approve more than available balance raises an error."""
        with self.assertRaises(ValueError):
            self.wallet.decision(case=0, decision=1, value=1100.0)  # Approve with amount exceeding balance

    def test_auto_budget_assignment(self):
        """Test if the budget is automatically assigned to a case when a decision is made."""
        self.wallet.decision(case=0, decision=1, value=400.0)  # Approve with 400
        case_info = self.wallet.get_case_history(0)
        self.assertEqual(case_info, [{'decision': 1, 'value': 400.0}])
        self.assertEqual(self.wallet.balance, 600.0)  # 1000 - 400 = 600

    def test_reset_wallet(self):
        """Test if the wallet is properly reset."""
        self.wallet.decision(case=0, decision=1, value=500.0)
        self.wallet.reset()  # Reset the wallet
        self.assertEqual(self.wallet.balance, 1000.0)
        self.assertEqual(self.wallet.budget_spent_pct, 0.0)
        self.assertEqual(self.wallet.get_case_history(0), [])  # History should be cleared


if __name__ == '__main__':
    unittest.main()
