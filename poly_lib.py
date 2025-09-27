import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 20)
y = np.sin(x)
p = np.polyfit(x, y, 4)
xx = np.linspace(0, 2*np.pi, 200)
yy = np.polyval(p, xx)

plt.plot(xx, np.sin(xx))
plt.plot(xx, yy)
plt.scatter(x, y)
plt.show()
