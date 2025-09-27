import numpy as np
import matplotlib.pyplot as plt

def gen_data(N, noise=0.0, seed=None):
    if seed is not None: np.random.seed(seed)
    x = np.linspace(0, 2*np.pi, N)
    y = np.sin(x)
    if noise > 0:
        y = y + (np.random.rand(N)*2 - 1) * noise
    return x, y


def fit_spline_lib(x, y):
    try:
        from scipy.interpolate import CubicSpline
    except Exception as e:
        print("scipy not found; unable to use the library", e)
        return None
    cs = CubicSpline(x, y, bc_type='natural')
    return cs


def natural_spline_M(x, y):
    n = len(x)
    if n <= 2:
        return np.zeros(n)
    h = np.diff(x)
    A = np.zeros((n-2, n-2))
    rhs = np.zeros(n-2)
    for i in range(1, n-1):
        if i-1 >= 0:
            A[i-1, i-1] = (h[i-1] + h[i]) / 3.0
        if i-2 >= 0:
            A[i-1, i-2] = h[i-1] / 6.0
        if i < n-2:
            A[i-1, i] = h[i] / 6.0
        rhs[i-1] = (y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1]
    m_inner = np.linalg.solve(A, rhs)
    M = np.zeros(n)
    M[1:-1] = m_inner
    return M

def spline_eval_nodes(x_nodes, y_nodes, M, x_eval):
    x_nodes = np.asarray(x_nodes)
    x_eval = np.atleast_1d(x_eval)
    y_out = np.zeros_like(x_eval, dtype=float)
    for k, xx in enumerate(x_eval):
        if xx <= x_nodes[0]:
            i = 0
        elif xx >= x_nodes[-1]:
            i = len(x_nodes)-2
        else:
            i = np.searchsorted(x_nodes, xx) - 1
        xi, xi1 = x_nodes[i], x_nodes[i+1]
        hi = xi1 - xi
        A = (xi1 - xx)/hi
        B = (xx - xi)/hi
        y_out[k] = A*y_nodes[i] + B*y_nodes[i+1] + ((A**3 - A)*M[i] + (B**3 - B)*M[i+1]) * (hi**2) / 6.0
    return y_out if y_out.size>1 else float(y_out)


def test_and_plot():
    x_fine = np.linspace(0,2*np.pi,800)
    y_true = np.sin(x_fine)

    for N in (5,20,50):
        x, y = gen_data(N)
        cs = fit_spline_lib(x,y)
        if cs is not None:
            y_fit_lib = cs(x_fine)
            plt.figure(figsize=(7,3.5))
            plt.plot(x_fine, y_true, label='true sin')
            plt.plot(x_fine, y_fit_lib, label=f'lib cubic spline, N={N}')
            plt.scatter(x,y,c='k')
            plt.legend()
            plt.title('lib cubic spline')
            plt.show()

            plt.figure(figsize=(7,2.5))
            plt.plot(x_fine, y_true - y_fit_lib)
            plt.axhline(0, color='k', linewidth=0.5)
            plt.title('diff (true - lib spline)')
            plt.show()

        # 从头实现
        M = natural_spline_M(x,y)
        y_fit_scratch = spline_eval_nodes(x,y,M,x_fine)
        plt.figure(figsize=(7,3.5))
        plt.plot(x_fine, y_true, label='true sin')
        plt.plot(x_fine, y_fit_scratch, label=f'scratch natural spline, N={N}')
        plt.scatter(x,y,c='k')
        plt.legend()
        plt.title('scratch natural cubic spline')
        plt.show()

        plt.figure(figsize=(7,2.5))
        plt.plot(x_fine, y_true - y_fit_scratch)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.title('diff (true - scratch spline)')
        plt.show()


    for err in (0.02, 0.10, 0.20):
        x, y = gen_data(20, noise=err, seed=10)
        cs = fit_spline_lib(x,y)
        if cs is not None:
            y_fit_lib = cs(x_fine)
            plt.figure(figsize=(7,3.5))
            plt.plot(x_fine, y_true, label='true sin')
            plt.plot(x_fine, y_fit_lib, label=f'lib spline noise {err*100:.0f}%')
            plt.scatter(x,y,c='k')
            plt.legend()
            plt.title('lib spline noise test')
            plt.show()

            plt.figure(figsize=(7,2.5))
            plt.plot(x_fine, y_true - y_fit_lib)
            plt.axhline(0, color='k', linewidth=0.5)
            plt.title('diff (true - lib spline)')
            plt.show()

        M = natural_spline_M(x,y)
        y_fit_scratch = spline_eval_nodes(x,y,M,x_fine)
        plt.figure(figsize=(7,3.5))
        plt.plot(x_fine, y_true, label='true sin')
        plt.plot(x_fine, y_fit_scratch, label=f'scratch spline noise {err*100:.0f}%')
        plt.scatter(x,y,c='k')
        plt.legend()
        plt.title('scratch spline noise test')
        plt.show()

        plt.figure(figsize=(7,2.5))
        plt.plot(x_fine, y_true - y_fit_scratch)
        plt.axhline(0, color='k', linewidth=0.5)
        plt.title('diff (true - scratch spline)')
        plt.show()

if __name__ == "__main__":
    test_and_plot()
