import numpy as np
import matplotlib.pyplot as plt

def gen_data(N, noise=0.0, seed=None):
    if seed is not None: np.random.seed(seed)
    x = np.linspace(0, 2*np.pi, N)
    y = np.sin(x)
    if noise > 0:
        y = y + (np.random.rand(N)*2 - 1) * noise
    return x, y


def fit_interp_lib(x, y):
    deg = len(x)-1
    coeff = np.polyfit(x, y, deg)
    return np.poly1d(coeff)


def lagrange_eval(x_nodes, y_nodes, x_eval):
    x_nodes = np.asarray(x_nodes)
    y_nodes = np.asarray(y_nodes)
    x_eval = np.atleast_1d(x_eval)
    n = len(x_nodes)
    y_out = np.zeros_like(x_eval, dtype=float)

    for k, xv in enumerate(x_eval):
        s = 0.0
        for j in range(n):
            num = 1.0
            den = 1.0
            xj = x_nodes[j]
            for m in range(n):
                if m == j: continue
                xm = x_nodes[m]
                num *= (xv - xm)
                den *= (xj - xm)
            s += y_nodes[j] * (num/den)
        y_out[k] = s
    return y_out if y_out.size>1 else float(y_out)


def test_and_plot(label_lib='lib', label_scratch='scratch'):
    x_fine = np.linspace(0,2*np.pi,800)
    y_true = np.sin(x_fine)

    for N in (5,20,50):
        x, y = gen_data(N)
        P = fit_interp_lib(x,y)
        y_fit_lib = P(x_fine)
        y_fit_scratch = lagrange_eval(x,y,x_fine)


        plt.figure(figsize=(7,3.5))
        plt.plot(x_fine, y_true, label='true sin')
        plt.plot(x_fine, y_fit_lib, label=f'{label_lib} interp (N={N})')
        plt.scatter(x,y,c='k')
        plt.legend()
        plt.title(f'{label_lib} interp poly, N={N}')
        plt.show()

        plt.figure(figsize=(7,2.5))
        plt.plot(x_fine, y_true - y_fit_lib)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.title(f'diff (true - {label_lib}), N={N}')
        plt.show()

        # 从头实现图
        plt.figure(figsize=(7,3.5))
        plt.plot(x_fine, y_true, label='true sin')
        plt.plot(x_fine, y_fit_scratch, label=f'{label_scratch} Lagrange (N={N})')
        plt.scatter(x,y,c='k')
        plt.legend()
        plt.title(f'{label_scratch} Lagrange, N={N}')
        plt.show()

        plt.figure(figsize=(7,2.5))
        plt.plot(x_fine, y_true - y_fit_scratch)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.title(f'diff (true - {label_scratch}), N={N}')
        plt.show()

    # N=20 噪声测试 (只演示从头实现和库实现)
    for err in (0.02, 0.10, 0.20):
        x, y = gen_data(20, noise=err, seed=2)
        P = fit_interp_lib(x,y)
        y_fit_lib = P(x_fine)
        y_fit_scratch = lagrange_eval(x,y,x_fine)

        plt.figure(figsize=(7,3.5))
        plt.plot(x_fine, y_true, label='true sin')
        plt.plot(x_fine, y_fit_lib, label=f'lib interp, noise {err*100:.0f}%')
        plt.scatter(x,y,c='k')
        plt.legend()
        plt.title(f'lib interp, N=20, noise {err*100:.0f}%')
        plt.show()

        plt.figure(figsize=(7,2.5))
        plt.plot(x_fine, y_true - y_fit_lib)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.title('diff (true - lib)')
        plt.show()

        plt.figure(figsize=(7,3.5))
        plt.plot(x_fine, y_true, label='true sin')
        plt.plot(x_fine, y_fit_scratch, label=f'scratch Lagrange, noise {err*100:.0f}%')
        plt.scatter(x,y,c='k')
        plt.legend()
        plt.title(f'scratch Lagrange, N=20, noise {err*100:.0f}%')
        plt.show()

        plt.figure(figsize=(7,2.5))
        plt.plot(x_fine, y_true - y_fit_scratch)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.title('diff (true - scratch)')
        plt.show()

if __name__ == "__main__":
    test_and_plot()
