import numpy as np
import matplotlib.pyplot as plt

def make_data(n):
    x=np.linspace(0,2*np.pi,n)
    y=np.sin(x)
    return x,y

def natural_cubic_spline_interpolate(x,y,xx):
    n=len(x)
    h=np.diff(x)
    A=np.zeros((n,n))
    rhs=np.zeros(n)
    A[0,0]=1
    A[-1,-1]=1
    for i in range(1,n-1):
        A[i,i-1]=h[i-1]
        A[i,i]=2*(h[i-1]+h[i])
        A[i,i+1]=h[i]
        rhs[i]=3*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
    c=np.linalg.solve(A,rhs)
    b=np.zeros(n-1)
    d=np.zeros(n-1)
    for i in range(n-1):
        b[i]=(y[i+1]-y[i])/h[i] - h[i]*(2*c[i]+c[i+1])/3
        d[i]=(c[i+1]-c[i])/(3*h[i])
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

def run_all():
    for n in [5,20,50]:
        x,y=make_data(n)
        xx=np.linspace(0,2*np.pi,500)
        yf=natural_cubic_spline_interpolate(x,y,xx)
        plt.figure()
        plt.plot(xx,np.sin(xx),label='sin')
        plt.plot(xx,yf,'--',label=f'natural spline n={n}')
        plt.scatter(x,y)
        plt.legend()
        plt.savefig(f'spline_scratch_fit_n{n}.png')
        plt.close()
        plt.figure()
        plt.plot(xx,np.abs(np.sin(xx)-yf))
        plt.title(f'abs error n={n}')
        plt.savefig(f'spline_scratch_err_n{n}.png')
        plt.close()
    x,y=make_data(20)
    rng=np.random.default_rng(5)
    for pct in [0.02,0.10,0.20]:
        y_noisy=y+(rng.random(len(y))*2-1)*pct
        xx=np.linspace(0,2*np.pi,500)
        yf=natural_cubic_spline_interpolate(x,y_noisy,xx)
        plt.figure()
        plt.plot(xx,np.sin(xx),label='sin')
        plt.plot(xx,yf,'--',label=f'noisy pct={int(pct*100)}%')
        plt.scatter(x,y_noisy)
        plt.legend()
        plt.savefig(f'spline_scratch_noisy_pct{int(pct*100)}.png')
        plt.close()

if __name__=='__main__':
    run_all()