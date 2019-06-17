from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from pathlib import Path


def imshow_3d(Z, rstride=1, cstride=1):

    ax = plt.gca(projection='3d')

    # Make data.
    X = np.arange(Z.shape[1])
    Y = np.arange(Z.shape[0])
    X, Y = np.meshgrid(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(
        X, Y, Z,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
        rstride=1,
        cstride=1,
    )

    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # # Add a color bar which maps values to colors.
    # plt.gcf().colorbar(surf, shrink=0.5, aspect=5)


def multi_save_fig(
        basename,
        fig=None,
        verbose=0,
        suffixes=('.png', '.svg', '.pdf'),
        **kwargs):
    basename = Path(basename)
    if fig is None:
        fig = plt.gcf()
    basename.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault('bbox_inches', 'tight')
    kwargs.setdefault('pad_inches', 0.1)
    kwargs.setdefault('transparent', True)
    kwargs.setdefault('dpi', 150)
    for suffix in suffixes:
        suffixed_name = str(basename.with_suffix(suffix))
        fig.savefig(suffixed_name, **kwargs)
        if verbose > 10:
            print("saved to {}".format(suffixed_name))
