import numpy as np

class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10, rand_seed=0):
        self.eta = eta
        self.n_iter =n_iter
        self.rand_seed = rand_seed

    def fit(self, X, y):
        rgen = np.random.RandomState(self.rand_seed)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1]+1)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for observation, target in zip(X, y):
                update = self.eta*(target - self.predict(observation))
                self.w_ += update*np.insert(observation, 0, 1)
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)