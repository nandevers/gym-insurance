import os
import tensorflow as tf
from keras.callbacks import TensorBoard


class Wallet:
    def __init__(self, budget) -> None:
        self.budget = budget
        self.pct_budget = 1
        self.decisions = {}
        self.spent = 0

    def withdraw(self, value):
        assert value > 0
        self.spent += value
        if self.pct_budget <= 1 and self.spent<=self.budget:
            self.pct_budget = self.spent / self.budget
        else:
            raise ValueError

    @property
    def balance(self):
        return self.budget - self.spent


# Own Tensorboard class

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.writer.set_as_default()

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self.log_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self.log_dir, 'validation')
        self._val_step = self.model._test_counter


    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def update_stats(self,step, **logs):
        with self.writer.as_default():
            for log in logs.items():
                tf.summary.scalar(log[0],log[1], step=step)