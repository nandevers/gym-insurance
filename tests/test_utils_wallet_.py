import unittest
import logging
from gym_insurance.envs.utils import Wallet

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestWallet(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up the Wallet with a budget of 1000")
        self.wallet = Wallet(budget=1000, sample_index=[0, 1, 2, 3])

    def test_initialization(self):
        logger.info("Testing wallet initialization")
        self.assertEqual(self.wallet.balance, 1000)
        self.assertEqual(self.wallet.budget_spent_percentage, 0)

    def test_withdraw(self):
        logger.info("Testing valid withdrawal")
        self.wallet.withdraw(100)
        self.assertEqual(self.wallet.balance, 900)
        self.assertEqual(self.wallet.budget_spent_percentage, 0.1)

    def test_invalid_withdraw(self):
        logger.info("Testing invalid withdrawal scenarios")
        with self.assertRaises(ValueError):
            self.wallet.withdraw(-100)

        with self.assertRaises(ValueError):
            self.wallet.withdraw(1100)

    def test_decisions(self):
        logger.info("Testing decisions")
        case = 1
        self.wallet.current_case = case
        self.wallet.add_decision(case=case, decision=50)
        self.assertEqual(self.wallet.decisions[case], 50)

    def test_invalid_decisions(self):
        case = None
        logger.info("Testing invalid decisions")
        with self.assertRaises(TypeError):
            self.wallet.current_case = case
            self.wallet.add_decision(case=case, decision="invalid")

        with self.assertRaises(TypeError):
            self.wallet.add_decision(case=case, decision=50)

    def test_reset(self):
        logger.info("Testing wallet reset")
        self.wallet.withdraw(100)
        self.wallet.reset()
        self.assertEqual(self.wallet.balance, 1000)


if __name__ == "__main__":
    unittest.main()
