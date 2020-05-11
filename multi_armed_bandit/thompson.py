
import numpy as np
import matplotlib.pyplot as plt
import math

class Bandit:

    def __init__(self, p):
        self.P = p
        self.N = 0
        self.alpha = 1
        self.beta = 1

    def pull(self):
        return np.random.random() < self.P

    def dist(self):
        return np.random.beta(self.alpha, self.beta)

    def update(self, x):
        self.N += 1
        self.alpha += x
        self.beta += 1 - x


def thompson(bandits):

    for i in range(trials):

        bi = np.argmax([b.dist() for b in bandits])
        result = bandits[bi].pull()
        bandits[bi].update(result)

    for i in range(len(bandits)):
        print("alpha: " + str(bandits[i].alpha) + ' beta: ' + str(
            bandits[i].beta) + ' p:' + str(bandits[i].alpha / bandits[i].N))


if __name__ == "__main__":
    trials = 200
    thompson([Bandit(p) for p in [0.25, 0.5, 0.75]])
