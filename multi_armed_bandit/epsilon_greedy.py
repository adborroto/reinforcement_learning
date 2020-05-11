
import numpy as np
import matplotlib.pyplot as plt

class Bandit:

    def __init__(self, p):
        self.P = p
        self.N = 0
        self.p = 0  # Estimation

        self.stats = [] # Historical aprox of P

    def pull(self):
        return np.random.random() < self.P

    def update(self, x):
        self.N += 1
        self.p = ((self.N - 1) * self.p + x) / self.N
        self.stats.append(self.p)


def epsilon_greedy(bandits, eps):

    num_times_explored = 0
    num_times_exploited = 0

    for i in range(trials):

        # Epsilon-greedy
        if np.random.random() < eps:
            bi = np.random.randint(len(bandits))
            num_times_explored += 1
        else:
            bi = np.argmax([b.p for b in bandits])
            num_times_explored += 1

        result = bandits[bi].pull()
        bandits[bi].update(result)

    for i in range(len(bandits)):
        print("real: " + str(bandits[i].P) + ' estimation: ', bandits[i].p)
        plt.plot(bandits[i].stats)
        plt.plot(np.ones(trials) * bandits[i].P)

    plt.show()


if __name__ == "__main__":
    trials = 2000
    eps = 0.3
    epsilon_greedy([Bandit(p) for p in [0.25, 0.5, 0.75]], eps)
