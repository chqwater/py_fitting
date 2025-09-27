import numpy as np
import matplotlib.pyplot as plt

def gen_data(N, noise=0.0, seed=None):
    if seed is not None: np.random.seed(seed)
    x = np.linspace(0, 2*np.pi, N)
    y = np.sin(x)
    if noise > 0:
        y = y + (np.random.rand(N)*2 - 1) * noise
    return x, y


def fit_poly_lib(x, y, deg=4):
    coeff = np.polyfit(x, y, deg)
    p = np.poly1d(coeff)
    return p


def fit_poly_scratch(x, y, deg=4):
    V = np.vander(x, deg+1)
    A = V.T @ V
    b = V.T @ y
    coeff = np.linalg.solve(A, b)
    return np.poly1d(coeff)

def test_and_plot(fit_fn, label):
    x_fine = np.linspace(0, 2*np.pi, 400)
    y_true = np.sin(x_fine)

    for N in (5,20,50):
        x, y = gen_data(N)
        p = fit_fn(x, y)
        y_fit = p(x_fine)

        plt.figure(figsize=(7,3.5))
        plt.plot(x_fine, y_true, label='true sin')
        plt.plot(x_fine, y_fit, label=f'{label}, N={N}')
        plt.scatter(x, y, c='k')
        plt.legend()
        plt.title(f'{label} degree4, N={N}')
        plt.show()

        plt.figure(figsize=(7,2.5))
        plt.plot(x_fine, y_true - y_fit)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.title(f'difference (true - fit), N={N}')
        plt.show()

    # N=20 的噪声测试
    for err in (0.02, 0.10, 0.20):
        x, y = gen_data(20, noise=err, seed=1)
        p = fit_fn(x, y)
        y_fit = p(x_fine)
        plt.figure(figsize=(7,3.5))
        plt.plot(x_fine, y_true, label='true sin')
        plt.plot(x_fine, y_fit, label=f'{label}, noise={err*100:.0f}%')
        plt.scatter(x, y, c='k')
        plt.legend()
        plt.title(f'{label} degree4, N=20, noise {err*100:.0f}%')
        plt.show()

        plt.figure(figsize=(7,2.5))
        plt.plot(x_fine, y_true - y_fit)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.title(f'difference (true - fit), noise {err*100:.0f}%')
        plt.show()

if __name__ == "__main__":

    print("library version")
    test_and_plot(lambda x,y: fit_poly_lib(x,y,deg=4), 'lib')

    print("first principle version")
    test_and_plot(lambda x,y: fit_poly_scratch(x,y,deg=4), 'scratch')
