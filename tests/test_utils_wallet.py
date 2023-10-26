import unittest
from gym_insurance.envs.utils import Wallet

class TestWallet(unittest.TestCase):

    def setUp(self):
        self.wallet = Wallet(100)  # Initialize a wallet object with a budget of 100 for each test

    def test_init(self):
        self.assertEqual(self.wallet.budget, 100)
        self.assertEqual(self.wallet.spent, 0)
        with self.assertRaises(ValueError):
            Wallet(-100)  # Budget should be a positive integer

    def test_withdraw(self):
        self.wallet.withdraw(50)
        self.assertEqual(self.wallet.spent, 50)
        self.assertEqual(self.wallet.balance, 50)
        with self.assertRaises(ValueError):
            self.wallet.withdraw(-10)  # Withdrawal amount should be positive
        with self.assertRaises(ValueError):
            self.wallet.withdraw(0)    # Withdrawal amount should be positive
        with self.assertRaises(ValueError):
            self.wallet.withdraw(60)   # Insufficient balance for withdrawal

    def test_balance(self):
        self.assertEqual(self.wallet.balance, 100)
        self.wallet.withdraw(25)
        self.assertEqual(self.wallet.balance, 75)

    def test_multiple_withdrawals(self):
        self.wallet.withdraw(20)
        self.wallet.withdraw(30)
        self.assertEqual(self.wallet.spent, 50)
        self.assertEqual(self.wallet.balance, 50)

if __name__ == "__main__":
    unittest.main()
