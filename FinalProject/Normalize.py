import numpy as np


class Normalize:

    def norm(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
