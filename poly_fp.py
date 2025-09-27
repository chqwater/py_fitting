import numpy as np
import matplotlib.pyplot as plt

def make_data(n):
    x=np.linspace(0,2*np.pi,n)
    y=np.sin(x)
    return x,y

def fit_poly_leastsq(x,y,deg=4):
    A=np.vander(x,deg+1)
    coeffs=np.linalg.lstsq(A,y,rcond=None)[0]
    def p(xx):
        X=np.vander(xx,deg+1)
        return X.dot(coeffs)
    return p

def run_all():
    for n in [5,20,50]:
        x,y=make_data(n)
        p=fit_poly_leastsq(x,y,4)
        xx=np.linspace(0,2*np.pi,500)
        yy=np.sin(xx)
        yf=p(xx)
        plt.figure()
        plt.plot(xx,yy,label='sin')
        plt.plot(xx,yf,'--',label=f'poly deg4 fit n={n}')
        plt.scatter(x,y)
        plt.legend()
        plt.savefig(f'poly_scratch_fit_n{n}.png')
        plt.close()
        err=np.abs(yy-yf)
        plt.figure()
        plt.plot(xx,err)
        plt.title(f'abs error n={n}')
        plt.savefig(f'poly_scratch_err_n{n}.png')
        plt.close()
    x,y=make_data(20)
    rng=np.random.default_rng(1)
    for pct in [0.02,0.10,0.20]:
        y_noisy=y+(rng.random(len(y))*2-1)*pct
        p=fit_poly_leastsq(x,y_noisy,4)
        xx=np.linspace(0,2*np.pi,500)
        plt.figure()
        plt.plot(xx,np.sin(xx),label='sin')
        plt.plot(xx,p(xx),'--',label=f'noisy pct={int(pct*100)}%')
        plt.scatter(x,y_noisy)
        plt.legend()
        plt.savefig(f'poly_scratch_noisy_pct{int(pct*100)}.png')
        plt.close()

if __name__=='__main__':
    run_all()