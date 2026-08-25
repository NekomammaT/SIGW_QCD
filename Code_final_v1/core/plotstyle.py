"""Matplotlib style: LaTeX-ish labels in Computer Modern."""
import matplotlib
try:
    get_ipython()
    _IN_NOTEBOOK = True
except NameError:
    _IN_NOTEBOOK = False
if not _IN_NOTEBOOK:
    matplotlib.use("Agg")  # headless script: no GUI backend
import matplotlib.pyplot as plt


def use_latex():
    """Try full LaTeX; fall back to matplotlib's built-in Computer Modern."""
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        })
        fig = plt.figure()
        fig.text(0.5, 0.5, r"$\eta\mathcal{H}$")
        fig.canvas.draw()
        plt.close(fig)
    except Exception:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
        })
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "legend.frameon": False, "figure.dpi": 140,
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })
