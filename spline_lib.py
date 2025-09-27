import numpy as np
import matplotlib.pyplot as plt

def make_data(n):
    x=np.linspace(0,2*np.pi,n)
    y=np.sin(x)
    return x,y

try:
    from scipy.interpolate import CubicSpline
    has_scipy=True
except Exception:
    has_scipy=False

def run_all():
    for n in [5,20,50]:
        x,y=make_data(n)
        xx=np.linspace(0,2*np.pi,500)
        if has_scipy:
            cs=CubicSpline(x,y,bc_type='natural')
            yf=cs(xx)
        else:
            coeffs=numpy_natural_cubic_spline(x,y,xx)
            yf=coeffs
        plt.figure()
        plt.plot(xx,np.sin(xx),label='sin')
        plt.plot(xx,yf,'--',label=f'cubic spline n={n}')
        plt.scatter(x,y)
        plt.legend()
        plt.savefig(f'spline_numpy_fit_n{n}.png')
        plt.close()
        plt.figure()
        plt.plot(xx,np.abs(np.sin(xx)-yf))
        plt.title(f'abs error n={n}')
        plt.savefig(f'spline_numpy_err_n{n}.png')
        plt.close()
    x,y=make_data(20)
    rng=np.random.default_rng(4)
    for pct in [0.02,0.10,0.20]:
        y_noisy=y+(rng.random(len(y))*2-1)*pct
        xx=np.linspace(0,2*np.pi,500)
        if has_scipy:
            cs=CubicSpline(x,y_noisy,bc_type='natural')
            yf=cs(xx)
        else:
            yf=numpy_natural_cubic_spline(x,y_noisy,xx)
        plt.figure()
        plt.plot(xx,np.sin(xx),label='sin')
        plt.plot(xx,yf,'--',label=f'noisy pct={int(pct*100)}%')
        plt.scatter(x,y_noisy)
        plt.legend()
        plt.savefig(f'spline_numpy_noisy_pct{int(pct*100)}.png')
        plt.close()

def numpy_natural_cubic_spline(x,y,xx):
    n=len(x)
    h=np.diff(x)
    alpha=np.zeros(n)
    for i in range(1,n-1):
        alpha[i]=3*( (y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1] )
    l=np.ones(n)
    mu=np.zeros(n)
    z=np.zeros(n)
    for i in range(1,n-1):
        l[i]=2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i]=h[i]/l[i]
        z[i]=(alpha[i]-h[i-1]*z[i-1])/l[i]
    b=np.zeros(n-1)
    c=np.zeros(n)
    d=np.zeros(n-1)
    for j in range(n-2,-1,-1):
        c[j]=z[j]-mu[j]*c[j+1]
        b[j]=(y[j+1]-y[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        d[j]=(c[j+1]-c[j])/(3*h[j])
    yy=np.zeros_like(xx)
    for i,val in enumerate(xx):
        if val==x[-1]:
            yy[i]=y[-1]
            continue
        j=np.searchsorted(x,val)-1
        if j<0: j=0
        dx=val-x[j]
        yy[i]=y[j]+b[j]*dx + c[j]*dx**2 + d[j]*dx**3
    return yy

if __name__=='__main__':
    run_all()