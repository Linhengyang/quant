import numpy as np
# from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt

# def use_svg_display():
#     """使用svg格式在Jupyter中显示绘图"""
#     backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小"""
    plt.rcParams['figure.figsize'] = figsize
    

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None,
         xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X只有一个轴，输出True, 即如果X是一维向量 或 scalar的列表
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None: # 如果只输入了一个轴, 那么值默认显示在Y轴, X轴用整数index
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla() # 清除当前坐标轴
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def display_save(fname):
    plt.savefig(fname)
    plt.show()




if __name__ == "__main__":
    X = np.random.uniform(0, 1, size=(3,))
    Y = np.random.uniform(1, 2, size=(3,))
    plot(X, Y)
    display_save("plt_test.png")