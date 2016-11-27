import numpy as np, pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib


def ricker_i(points, a):
    return np.imag(signal.hilbert(signal.ricker(points, a)))

def pltcwt(plt, my_cwt, widths, realize=None):
    hi = np.max(widths)
    lo = np.min(widths)
    if realize == 'abs':
        my_cwt = np.abs(my_cwt)
    elif realize == 'ang':
        my_cwt = np.angle(my_cwt)
    plt.imshow(my_cwt, extent=[-1, 1, hi, lo], cmap='seismic', aspect='auto',
               vmax=abs(my_cwt).max(), vmin=-abs(my_cwt).max())


def plt_easycwt(plt, sig, widths):
    my_cwt = signal.cwt(sig, signal.ricker, widths)
    pltcwt(plt, my_cwt, widths)



from numpy import cos, sin

matplotlib.style.use('ggplot')
plt.figure(figsize=(8, 6))

from mpl_toolkits.mplot3d import Axes3D


def scatter3d(xyz, marker='bo', *args, **kwargs):
    ax = Axes3D(matplotlib.pyplot.figure())
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker)


class Easy3dScatter(object):
    def __init__(self, plotter, data, title, s=1, figsize=None, ElevAzi=None, c=None):
        figsize = (6, 4) if figsize is None else figsize
        plotter.rcParams['figure.figsize'] = figsize
        ElevAzi = (10, 10) if ElevAzi is None else ElevAzi
        if isinstance(data, pd.DataFrame):
            xs, ys, zs = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]
        else:
            xs, ys, zs = data[:, 0], data[:, 1], data[:, 2]

        X, Y, Z = xs, ys, zs
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(*ElevAzi)
        ax.scatter(xs, ys, zs, s=s, c=c)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plotter.axis('equal')
        ax.set_title(title)

        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

        plt.show()