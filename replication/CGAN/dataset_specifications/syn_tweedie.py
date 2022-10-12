import numpy as np
import tweedie

import utils
from dataset_specifications.dataset import Dataset

class SynTweedieSet(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "syn_tweedie"

        self.std_dev = np.sqrt(0.5)

    def get_support(self, x):
        return (x-2*self.std_dev, x+2*self.std_dev)

    def sample(self, n):
        xs = np.random.uniform(low=-1., high=1., size=n)
        # noise = np.random.normal(loc=0., scale=self.std_dev, size=n)
        mus = np.exp(xs) # + noise)

        ys = tweedie.tweedie(mu=mus, p=1.5, phi=1).rvs()

        return np.stack((xs, ys), axis=1)

    def get_pdf(self, x): # not true, but won't use this code (too much) anyway
        return utils.get_gaussian_pdf(x, self.std_dev)

