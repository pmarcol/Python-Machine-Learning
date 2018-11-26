from adaline import AdaLiNe
import numpy as np
from numpy.random import seed

class StochasticGradientDescent(AdaLiNe):

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        AdaLiNe.__init__(self, eta=eta, n_iter=n_iter)
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
        
    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for obs, target in zip(X, y):
                cost.append(self._update_weights(obs, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for obs, target in zip(X, y):
                self._update_weights(obs, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        self.w_ = np.zeros(m+1)
        self.w_initialized = True

    def _update_weights(self, obs, target):
        output = self.net_input(obs)
        error = target - output
        self.w_[1:] += self.eta*obs.dot(error)
        self.w_[0] += self.eta*error
        cost = 0.5*error**2
        return cost