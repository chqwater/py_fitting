import numpy as np
import matplotlib.pyplot as plt

def cubic_spline(x, y):
    n = len(x)
    h = np.diff(x)
    alpha = [0]+[3*(y[i+1]-y[i])/h[i]-3*(y[i]-y[i-1])/h[i-1] for i in range(1,n-1)]+[0]
    l = [1]+[0]*(n-1)
    mu = [0]*n
    z = [0]*n
    for i in range(1,n-1):
        l.append(2*(x[i+1]-x[i-1])-h[i-1]*mu[i-1])
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i]-h[i-1]*z[i-1])/l[i]
    b = [0]*(n-1)
    c = [0]*n
    d = [0]*(n-1)
    for j in range(n-2,-1,-1):
        c[j] = z[j]-mu[j]*c[j+1]
        b[j] = (y[j+1]-y[j])/h[j]-h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
    return b,c,d

def eval_spline(x, y, b, c, d, xx):
    n = len(x)
    res=[]
    for t in xx:
        i = np.searchsorted(x,t)-1
        if i<0: i=0
        if i>=n-1: i=n-2
        dx = t-x[i]
        res.append(y[i]+b[i]*dx+c[i]*dx**2+d[i]*dx**3)
    return res

x = np.linspace(0, 2*np.pi, 20)
y = np.sin(x)
b,c,d = cubic_spline(x,y)
xx = np.linspace(0, 2*np.pi, 200)
yy = eval_spline(x,y,b,c,d,xx)

plt.plot(xx, np.sin(xx))
plt.plot(xx, yy)
plt.scatter(x, y)
plt.show()
