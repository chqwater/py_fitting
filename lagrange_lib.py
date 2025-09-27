import numpy as np
import matplotlib.pyplot as plt

def make_data(n):
    x=np.linspace(0,2*np.pi,n)
    y=np.sin(x)
    return x,y

def interp_poly_numpy(x,y):
    coeffs=np.polyfit(x,y,len(x)-1)
    return np.poly1d(coeffs)

def run_all():
    for n in [5,20,50]:
        x,y=make_data(n)
        p=interp_poly_numpy(x,y)
        xx=np.linspace(0,2*np.pi,500)
        plt.figure()
        plt.plot(xx,np.sin(xx),label='sin')
        plt.plot(xx,p(xx),'--',label=f'interp poly n={n}')
        plt.scatter(x,y)
        plt.legend()
        plt.savefig(f'lagrange_numpy_fit_n{n}.png')
        plt.close()
        plt.figure()
        plt.plot(xx,np.abs(np.sin(xx)-p(xx)))
        plt.title(f'abs error n={n}')
        plt.savefig(f'lagrange_numpy_err_n{n}.png')
        plt.close()
    x,y=make_data(20)
    rng=np.random.default_rng(2)
    for pct in [0.02,0.10,0.20]:
        y_noisy=y+(rng.random(len(y))*2-1)*pct
        p=interp_poly_numpy(x,y_noisy)
        xx=np.linspace(0,2*np.pi,500)
        plt.figure()
        plt.plot(xx,np.sin(xx),label='sin')
        plt.plot(xx,p(xx),'--',label=f'noisy pct={int(pct*100)}%')
        plt.scatter(x,y_noisy)
        plt.legend()
        plt.savefig(f'lagrange_numpy_noisy_pct{int(pct*100)}.png')
        plt.close()

if __name__=='__main__':
    run_all()