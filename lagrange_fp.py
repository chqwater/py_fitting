import numpy as np
import matplotlib.pyplot as plt

def L(x, i, X):
    p = 1
    for j in range(len(X)):
        if j != i:
            p *= (x-X[j])/(X[i]-X[j])
    return p

def lagrange(x, X, Y):
    return sum(Y[i]*L(x, i, X) for i in range(len(X)))

X = np.linspace(0, 2*np.pi, 5)
Y = np.sin(X)
xx = np.linspace(0, 2*np.pi, 200)
yy = [lagrange(x, X, Y) for x in xx]

plt.plot(xx, np.sin(xx))
plt.plot(xx, yy)
plt.scatter(X, Y)
plt.show()
