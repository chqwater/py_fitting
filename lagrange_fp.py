import numpy as np
import matplotlib.pyplot as plt

def make_data(n):
    x=np.linspace(0,2*np.pi,n)
    y=np.sin(x)
    return x,y

def lagrange_interp(xi,yi,xx):
    xx=np.array(xx)
    yi=np.array(yi)
    xi=np.array(xi)
    out=np.zeros_like(xx, dtype=float)
    n=len(xi)
    for k in range(n):
        term=np.ones_like(xx)
        for j in range(n):
            if j==k: continue
            term*= (xx-xi[j])/(xi[k]-xi[j])
        out+=yi[k]*term
    return out

def run_all():
    for n in [5,20,50]:
        x,y=make_data(n)
        xx=np.linspace(0,2*np.pi,500)
        yf=lagrange_interp(x,y,xx)
        plt.figure()
        plt.plot(xx,np.sin(xx),label='sin')
        plt.plot(xx,yf,'--',label=f'lagrange n={n}')
        plt.scatter(x,y)
        plt.legend()
        plt.savefig(f'lagrange_scratch_fit_n{n}.png')
        plt.close()
        plt.figure()
        plt.plot(xx,np.abs(np.sin(xx)-yf))
        plt.title(f'abs error n={n}')
        plt.savefig(f'lagrange_scratch_err_n{n}.png')
        plt.close()
    x,y=make_data(20)
    rng=np.random.default_rng(3)
    for pct in [0.02,0.10,0.20]:
        y_noisy=y+(rng.random(len(y))*2-1)*pct
        xx=np.linspace(0,2*np.pi,500)
        yf=lagrange_interp(x,y_noisy,xx)
        plt.figure()
        plt.plot(xx,np.sin(xx),label='sin')
        plt.plot(xx,yf,'--',label=f'noisy pct={int(pct*100)}%')
        plt.scatter(x,y_noisy)
        plt.legend()
        plt.savefig(f'lagrange_scratch_noisy_pct{int(pct*100)}.png')
        plt.close()

if __name__=='__main__':
    run_all()