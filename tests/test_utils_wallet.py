import unittest
from gym_insurance.envs.utils import (
    Wallet,
)  # Adjust the import based on your project structure

Wallet(100)


class TestWallet(unittest.TestCase):
    def setUp(self):
        self.wallet = Wallet(
            1000, sample_index=[0, 1, 2, 3]
        )  # Initialize a wallet object with a budget of 1000 for each test

    def test_init(self):
        self.assertEqual(self.wallet._budget, 1000)
        self.assertEqual(self.wallet._spent, 0)
        with self.assertRaises(ValueError):
            Wallet(
                -100, sample_index=[0, 1, 2, 3]
            )  # Budget should be a positive integer

    def test_reset(self):
        self.wallet.withdraw(200)  # Withdraw 200 from the wallet
        self.wallet.reset()  # Reset the wallet
        self.assertEqual(self.wallet._spent, 0)  # Spent should be reset to 0

    def test_withdraw(self):
        self.wallet.withdraw(200)
        self.assertEqual(self.wallet._spent, 200)
        self.assertEqual(self.wallet.balance, 800)
        with self.assertRaises(ValueError):
            self.wallet.withdraw(-10)  # Withdrawal amount should be positive
        with self.assertRaises(ValueError):
            self.wallet.withdraw(0)  # Withdrawal amount should be positive
        with self.assertRaises(ValueError):
            self.wallet.withdraw(801)  # Insufficient balance for withdrawal

    def test_balance(self):
        self.assertEqual(self.wallet.balance, 1000)
        self.wallet.withdraw(200)
        self.assertEqual(self.wallet.balance, 800)

    def test_validate_budget(self):
        with self.assertRaises(ValueError):
            Wallet._validate_budget(
                -100,
            )  # Static method, Budget should be a positive integer
        with self.assertRaises(ValueError):
            Wallet._validate_budget(
                0
            )  # Static method, Budget should be a positive integer


if __name__ == "__main__":
    unittest.main()
