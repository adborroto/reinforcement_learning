
import numpy as np
import matplotlib.pyplot as plt

N = 10  # Samples
W = 56  # Real weight

def f(x):
    return W * x

def gradient_descent(y):
    w = np.random.random()  # Initial weight
    learning_rate = 0.05
    ws = []
    iter = 20
    for x in range(iter):
        sum = 0
        for (xi, yi) in y:
            yh = (w * xi) - yi
            sum += yh * xi
        delta = sum / (N)
        w = w - learning_rate * delta
        ws.append(w)

    print("real w: " + str(W) + "final w: " + str(w))
    plt.plot(ws)
    plt.plot(np.ones(len(ws))* W)
    plt.show()


if __name__ == "__main__":
    y = np.array([(x, f(x)) for x in range(N)])
    gradient_descent(y)
