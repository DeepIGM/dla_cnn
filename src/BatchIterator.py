import numpy as np


class BatchIterator:
    def __init__(self, num_samples):
        self._num_samples = num_samples
        self._samples_consumed = np.inf

    def next_batch(self, batch_size):
        # keep track of how many samples have been consumed and reshuffle after an epoch has elapsed
        if self._samples_consumed >= self._num_samples:
            self._ix_permutation = np.random.permutation(self._num_samples)
            self._samples_consumed = 0

        batch_ix = self._ix_permutation[0:batch_size]
        self._ix_permutation = np.roll(self._ix_permutation, batch_size)
        self._samples_consumed += batch_size

        return batch_ix

