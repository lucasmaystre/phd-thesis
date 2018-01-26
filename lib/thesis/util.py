import matplotlib

from matplotlib.backends.backend_pgf import FigureCanvasPgf


MPL_RCPARAMS = {
    "figure.autolayout": True,     # Makes sure nothing the feature is neat & tight.
    "figure.figsize": (5.78, 3.0), # textwidth = 146.8mm, c.f. preamble.tex.
    "axes.linewidth": 0.5,         # Matplotlib's current default is 0.8.
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,
    "text.usetex": True,           # use LaTeX to write all text
    "text.latex.unicode": True,
    "font.family": "serif",        # use serif rather than sans-serif
    "font.serif": "lmodern",
    "font.size": 11,
    "axes.titlesize": 11,          # LaTeX default is 10pt font.
    "axes.labelsize": 9,           # LaTeX default is 10pt font.
    "legend.fontsize": 9,          # Make the legend/label fonts a little smaller
    "legend.frameon": False,       # Remove the black frame around the legend
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def setup_plotting():
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
    matplotlib.rcParams.update(MPL_RCPARAMS)
    print("Thesis settings loaded!")
