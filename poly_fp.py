import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 20)
y = np.sin(x)
X = np.vstack([x**i for i in range(5)]).T
a = np.linalg.solve(X.T@X, X.T@y)
xx = np.linspace(0, 2*np.pi, 200)
Xx = np.vstack([xx**i for i in range(5)]).T
yy = Xx@a

plt.plot(xx, np.sin(xx))
plt.plot(xx, yy)
plt.scatter(x, y)
plt.show()
