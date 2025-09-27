import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

x = np.linspace(0, 2*np.pi, 20)
y = np.sin(x)
cs = CubicSpline(x, y, bc_type='natural')
xx = np.linspace(0, 2*np.pi, 200)
yy = cs(xx)

plt.plot(xx, np.sin(xx))
plt.plot(xx, yy)
plt.scatter(x, y)
plt.show()
