import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

x = np.linspace(0, 2*np.pi, 5)
y = np.sin(x)
p = Polynomial.fit(x, y, len(x)-1).convert()
xx = np.linspace(0, 2*np.pi, 200)
yy = p(xx)

plt.plot(xx, np.sin(xx))
plt.plot(xx, yy)
plt.scatter(x, y)
plt.show()
