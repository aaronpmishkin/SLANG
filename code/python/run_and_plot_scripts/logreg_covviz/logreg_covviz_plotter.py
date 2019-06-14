r"""
General script to produce covariance vizualisation plots for SLANG.

Run `python logreg_covviz_plotter.py` to have the script display
the plots during execution.
Add the `--save` flag to save plots as `.pdf`.
Add the `--noshow` flag to *not* display plots during execution.

Requires the raw data to be available the same folder as the script,
which should be distributed along this script in the `logreg_covviz.zip` archive.

Check `python logreg_covviz_plotter.py --help` for possible plots and other opts.
"""

import argparse

import os

import numpy as np
import scipy as sp
import scipy.io

import matplotlib

import matplotlib.pyplot as plt
import matplotlib.ticker as tk

import pdb

import lib.utilities.plotting as plotutils

################################################################################
# GENERAL SETTINGS
################################################################################

GLOBAL_FONT_SIZE = 14
FONTWEIGHT = "bold"
COLORMAP = plt.cm.seismic

matplotlib.rcParams.update({'font.size': GLOBAL_FONT_SIZE})

################################################################################
# CONSTANTS AND MAGIC NUMBERS
################################################################################

USPS = "usps_3vs5"
A1A = "a1a"
AUSTRALIAN = "australian_scale"
BREAST = "breast_cancer_scale"
DATASETS = [A1A, USPS, AUSTRALIAN, BREAST]
SLANGS = [1, 5, 10]

METHOD_DISPLAY_TITLE = {
    "exact": "Full Gaussian",
    "mf-exact": "Mean Field",
    "slang 1": "SLANG (Rank 1)",
    "slang 5": "SLANG (Rank 5)",
    "slang 10": "SLANG (Rank 10)",
}

METHODS = ["exact", "mf-exact", "slang 1", "slang 5", "slang 10"]
METHODS_WITH_COV = ["exact", "slang 1", "slang 5", "slang 10"]
METHODS_APPROX = ["mf-exact", "slang 1", "slang 5", "slang 10"]
METHODS_APPROX_EXTR = ["mf-exact", "slang 5", "slang 10"]

COLORS = {
    "exact": [0, 0, 0],
    "mf-exact": [0, 0, 1],
    "slang 1": [.5, 0, 0],
    "slang 5": [1, 0, 0],
    "slang 10": [1, .6, .6],
}

BASE_SETTINGS = {
    "exact": {"label": "Full Gaussian", "color": COLORS["exact"], "linewidth": 6},
    "mf-exact": {"label": "MF", "color": COLORS["mf-exact"], "linewidth": 2},
    "slang 1": {"label": "SLANG-1", "color": COLORS["slang 1"], "linewidth": 3},
    "slang 5": {"label": "SLANG-5", "color": COLORS["slang 5"], "linewidth": 3},
    "slang 10": {"label": "SLANG-10", "color": COLORS["slang 10"], "linewidth": 3},
}

BASE_SETTINGS_SCATTER = {
    "exact": {"label": "Full Gaussian", "color": COLORS["exact"], "linewidth": 6, "linestyle": "none", "marker": "x"},
    "mf-exact": {"label": "MF", "color": COLORS["mf-exact"], "linewidth": 2, "linestyle": "none", "marker": "x"},
    "slang 1": {"label": "SLANG-1", "color": COLORS["slang 1"], "linewidth": 3, "linestyle": "none", "marker": "x"},
    "slang 5": {"label": "SLANG-5", "color": COLORS["slang 5"], "linewidth": 3, "linestyle": "none", "marker": "x"},
    "slang 10": {"label": "SLANG-10", "color": COLORS["slang 10"], "linewidth": 3, "linestyle": "none", "marker": "x"},
}

################################################################################
# CLI ARGUMENTS
################################################################################

arg_definitions = {
    '--save': {
        'dest': 'save', 'action': 'store_true', 'default': False,
        'help': (
            'Saves the generated plot in the current direction.' +
            'File names depend on generated plots and datasets.')
    },
    '--datafolder': {
        'dest': 'datafolder', 'action': 'store', 'default': "",
        'help': (
            'Root of the folder containing the plotting data.')
    },
    '--noshow': {
        'dest': 'noshow', 'action': 'store_true', 'default': False,
        'help': (
            'Disable showing plot during script run.' +
            'Useful if you just want to generate files.')
    },
}

parser = argparse.ArgumentParser(
    description='Run Covariance Vizualisation experiments and generate plots',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

for arg, arg_def in arg_definitions.items():
    parser.add_argument(arg, **arg_def)

################################################################################
# DATA LOADING
################################################################################


def load_dataset(dataset, datafolder=""):
    r"""
    Loads the data for the specified dataset 
    """
    METHOD_FOLDERS = {
        "exact": "exact_covariance_results",
        "mf-exact": "mf_exact_covariance_results",
        "slang": "slang_cov_results"
    }
    SLANG_M_Values = {
        "a1a": "128",
        "australian_scale": "32",
        "breast_cancer_scale": "32",
        "usps_3vs5": "64"
    }

    def load(method, L=None):
        r"""
        Loads the mean and covariance for a specific method
        """
        if method == "slang":
            filename = dataset + "_SLANG-V2_M_" + SLANG_M_Values[dataset] + "_L_" + str(L) + "_K_0_restart_1.mat"
        else:
            filename = dataset + "_" + method + "_M_1_L_0_restart_1.mat"

        path = os.path.join(os.path.join(datafolder, METHOD_FOLDERS[method]), os.path.join(dataset, filename))
        res = sp.io.loadmat(os.path.abspath(path))

        if len(res["mu"][0][0].shape) > 0:
            res = {"mu": res["mu"][0][0], "sig": res["Sigma"][0][0]}
        else:
            res = {"mu": res["mu"], "sig": res["Sigma"]}

        if len(res['sig'].shape) > 2:
            res['sig'] = res['sig'][:,:,-1]
        if res['mu'].shape[1] > 1:
            res['mu'] = res['mu'][:,-1].reshape((-1,1))

        return res

    results = {}
    results["exact"] = load("exact")
    results["mf-exact"] = load("mf-exact")
    for L in SLANGS:
        results["slang " + str(L)] = load("slang", L)

    return results

################################################################################
# PLOTTING FUNCTIONS
################################################################################


def grid_for_covariance_plot(fig):
    r"""Creates a [1 + 2x2 grid] for the covariances and colorbar"""
    gs = matplotlib.gridspec.GridSpec(2, 3, width_ratios=[0.1, 1, 1], height_ratios=[1, 1])
    axes = []
    axes.append(fig.add_subplot(gs[0, 1]))
    axes.append(fig.add_subplot(gs[0, 2]))
    axes.append(fig.add_subplot(gs[1, 1]))
    axes.append(fig.add_subplot(gs[1, 2]))
    ax_colorbar = fig.add_subplot(gs[:, 0])

    return gs, axes, ax_colorbar


def make_default_cov_plot(data, fig=None, showdiag=False, showbias=False):
    if fig is None:
        fig = plt.figure(figsize=(7, 7))

    gridspec, axes_cov, ax_colorbar = grid_for_covariance_plot(fig)
    gridspec.update(wspace=0.15, hspace=0.15)

    for methId, method in enumerate(METHODS_WITH_COV):
        cov = np.copy(data[method]["sig"])

        if not showdiag:
            np.fill_diagonal(cov, 0)
        if not showbias:
            cov = cov[1:, 1:]

        cmap = plt.cm.seismic
        img = axes_cov[methId].imshow(cov, cmap=cmap)
        axes_cov[methId].set_title(METHOD_DISPLAY_TITLE[method], fontweight=FONTWEIGHT)

    low, hig = plotutils.centerColorMap(axes_cov, COLORMAP)
    plotutils.remove(axes_cov, ["xlabel", "xticks", "yticks"])

    magnitude = int(np.log10(max(abs(low), abs(hig))))

    fmt1 = tk.FuncFormatter(lambda x, pos: "%1.1f" % (x / 10**magnitude))
    fig.colorbar(img, cax=ax_colorbar, format=fmt1)
#    if not magnitude == 0:
#        ax_colorbar.text(-0.25, 1, r'$\times$10$^{' + str(magnitude) + '}$', va='bottom', ha='left')
    ax_colorbar.yaxis.set_ticks_position('left')

    return gridspec


def make_mean_var_line_plot(data, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(8, 4))

    axes = []

    gridspec = matplotlib.gridspec.GridSpec(2, 1, width_ratios=[1], height_ratios=[1, 1])
    axes.append(fig.add_subplot(gridspec[0, 0]))
    axes.append(fig.add_subplot(gridspec[1, 0]))

    trueMeans = data["exact"]["mu"][:, -1].reshape(-1)[1:]
    trueVariances = np.diag(data["exact"]["sig"]).reshape(-1)[1:]
    sortIdxMean = np.flip(np.argsort(trueMeans))
    sortIdxVar = np.flip(np.argsort(trueVariances))

    axes[0].plot(trueMeans[sortIdxMean], **BASE_SETTINGS["exact"])
    axes[1].plot(trueVariances[sortIdxVar], **BASE_SETTINGS["exact"])
    for methId, method in enumerate(METHODS_APPROX):
        means = data[method]["mu"][:, -1].reshape(-1)[1:]
        axes[0].plot(means[sortIdxMean], **BASE_SETTINGS[method])

        variances = np.diag(data[method]["sig"]).reshape(-1)[1:]
        axes[1].plot(variances[sortIdxVar], **BASE_SETTINGS[method])

    axes[0].set_title("Mean", fontweight=FONTWEIGHT)
    axes[1].set_xlabel("Dimension", fontweight=FONTWEIGHT)
    axes[0].legend()
    axes[1].set_title("Variance", fontweight=FONTWEIGHT)
    axes[1].set_yscale("log")
    plotutils.hide_label_and_tickmark(axes[0].xaxis)

    for ax in axes:
        ax.grid()
        ax.set_xlim([0, len(sortIdxMean) - 1])

    return gridspec


def make_usps_plot(data, showdiag=False, showbias=False):

    fig = plt.figure(figsize=(14, 7))
    gs1 = make_default_cov_plot(data, fig, showdiag, showbias)
    gs1.update(left=0.05, right=0.47, wspace=0.15, hspace=0.15)

    gs2 = matplotlib.gridspec.GridSpec(2, 1, width_ratios=[2], height_ratios=[1, 1])
    gs2.update(left=0.52, right=0.98, wspace=0.15, hspace=0.15)
    axes_plt = []
    axes_plt.append(fig.add_subplot(gs2[0, 0]))
    axes_plt.append(fig.add_subplot(gs2[1, 0]))

    for methId, method in enumerate(METHODS):
        means = np.copy(data[method]["mu"])
        variances = np.copy(np.diag(data[method]["sig"]))

        if not showbias:
            means = means[1:]
            variances = variances[1:]

        axes_plt[0].plot(range(len(means)), means, **BASE_SETTINGS[method])
        axes_plt[1].plot(range(len(variances)), variances, **BASE_SETTINGS[method])

    axes_plt[0].set_ylim([-0.6, 0.4])
    axes_plt[0].set_title("Mean", fontweight=FONTWEIGHT)
    axes_plt[0].legend(mode="expand", ncol=len(METHODS), loc=8, prop={'size': 12})

    axes_plt[1].set_ylim([0.0125, 0.0425])
    axes_plt[1].set_title("Variance", fontweight=FONTWEIGHT)
    axes_plt[1].set_xlabel("Image pixel (by row)", fontweight=FONTWEIGHT)
    axes_plt[1].yaxis.set_ticks([0.02, 0.03, 0.04])

    for ax in axes_plt:
        plotutils.hide_label_and_tickmark(ax.xaxis)
        ax.set_xlim([0, len(means)])
        ax.grid()


def make_nonUSPS_plot(data, showdiag=False, showbias=False):

    fig = plt.figure(figsize=(14, 7))
    gs1 = make_default_cov_plot(data, fig, showdiag, showbias)
    gs1.update(left=0.05, right=0.45, wspace=0.15, hspace=0.15)

    gs2 = make_mean_var_line_plot(data, fig)
    gs2.update(left=0.52, right=0.98, wspace=0.15, hspace=0.15)


def plotAndSave(args, name):
    if not args.noshow:
        plt.show()
    if args.save:
        plt.savefig(name, bbox_inches='tight')
        print("Saving " + name)


def bias_term_info(data_per_dataset):

    output = "Datasets & "
    for methId, method in enumerate(METHODS):
        output += method
        if methId < len(METHODS) - 1:
            output += r" & "
        else:
            output += r"\\" + "\n"

    for dataset, data in data_per_dataset.items():
        output += dataset + " & "
        for methId, method in enumerate(METHODS):
            output += "$%.2f" % data_per_dataset[dataset][method]["mu"][0, -1]
            output += " \pm %.2f$" % data_per_dataset[dataset][method]["sig"][0, 0]

            if methId < len(METHODS) - 1:
                output += r"& "
            else:
                output += r"\\" + "\n"

    return output

################################################################################
# MAIN
################################################################################


if __name__ == "__main__":
    args = parser.parse_args()

    print("Argument passed:")
    print("    Showing output: " + str(not args.noshow))
    print("    Saving output: " + str(args.save))

    data_per_dataset = {}
    for dataset in DATASETS:
        data_per_dataset[dataset] = load_dataset(dataset, args.datafolder)

    print(bias_term_info(data_per_dataset))

    make_usps_plot(data_per_dataset[USPS])
    plotAndSave(args, "USPS_main_cov_plot.pdf")

    for dataset in DATASETS:
        make_nonUSPS_plot(data_per_dataset[dataset])
        plotAndSave(args, dataset + "_cov_plot.pdf")
