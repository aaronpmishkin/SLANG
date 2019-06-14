import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import convergence_config_bnn as BNN
import convergence_config_logreg as LOGREG
import convergence_cli as CLI

import lib.utilities.plotting as plotutils

GLOBAL_FONT_SIZE = 14
matplotlib.rcParams.update({'font.size': GLOBAL_FONT_SIZE})
FONTWEIGHT = "bold"


def plot_metric_on(ax, x, y, m, conf, display_name=None):
    if display_name is None:
        display_name = conf.DISPLAY_NAME
    ax.plot(
        x, np.mean(y, axis=0),
        color=conf.COLOR[m], linewidth=3,
        label=display_name[m]
    )
    plotutils.shaded_cov(
        ax, x, np.mean(y, axis=0), np.std(y, axis=0),
        color=conf.COLOR[m], alpha=0.2
    )


def plot_and_save_as_needed(name, args):
    if args.save:
        print("Saving " + name)
        plt.savefig(name + "_150.jpg", bbox_inches='tight', dpi=150)
        plt.savefig(name + "_300.jpg", bbox_inches='tight', dpi=300)
        plt.savefig(name + "_600.jpg", bbox_inches='tight', dpi=600)
        plt.savefig(name + ".pdf", bbox_inches='tight')
    if not args.noshow:
        plt.show()
    plt.close()


def make_main_covplot(args):
    FIGSCALE = 4
    fig = plt.figure(figsize=(4 * FIGSCALE, 2 * FIGSCALE))
    gs1 = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    gs2 = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    gs1.update(top=0.8, bottom=0.1, left=0.1, right=0.45, wspace=0.5, hspace=0.2)
    gs2.update(top=0.8, bottom=0.1, left=0.55, right=0.9, wspace=0.5, hspace=0.2)

    axes = [
        [fig.add_subplot(gs1[0, 0]), fig.add_subplot(gs1[0, 1]), fig.add_subplot(gs2[0, 0]), fig.add_subplot(gs2[0, 1])],
        [fig.add_subplot(gs1[1, 0]), fig.add_subplot(gs1[1, 1]), fig.add_subplot(gs2[1, 0]), fig.add_subplot(gs2[1, 1])],
    ]

    blacklist = ["exact", "VOG"]

    CUSTOM_DISPLAY_NAME = {
        "bbb": "BBB",
        "slang-1": "ERR",
        "slang-8": "SLANG(1)",
        "slang-16": "SLANG(2)",
        "slang-32": "SLANG(3)",
        "slang-64": "ERR",
        "mf-exact": "ERR",
        "VOG-D": "ERR",
        "VON-D": "Mean-Field",
        "SLANG-V2-1": "SLANG(1)",
        "SLANG-V2-2": "ERR",
        "SLANG-V2-5": "SLANG(2)",
        "SLANG-V2-10": "SLANG(3)",
        "VOG": "ERR",
        "VON": "Full-Gaussian",
        "exact": "ERR",
    }

    def plot_subset(axes, conf, dataset):
        METHODS = list([m for m in conf.METHODS if not any([blacklisted in m for blacklisted in blacklist])])
        for m in METHODS:
            out = conf.load(dataset, m, args.datafolder)
            mask = out[0] < conf.X_LIMS[dataset][1]
            plot_metric_on(axes[0], out[0][mask], out[1][:, mask], m, conf, CUSTOM_DISPLAY_NAME)
            axes[0].set_xlim(conf.X_LIMS[dataset])
            axes[0].set_ylim(conf.Y_LIMS[dataset]["ELBO"])
            plot_metric_on(axes[1], out[0][mask], out[2][:, mask], m, conf, CUSTOM_DISPLAY_NAME)
            axes[1].set_xlim(conf.X_LIMS[dataset])
            axes[1].set_ylim(conf.Y_LIMS[dataset]["LL"])

    plot_subset([axes[0][0], axes[0][1]], LOGREG, "usps_3vs5")
    plot_subset([axes[0][2], axes[0][3]], BNN, "usps_3vs5")
    plot_subset([axes[1][0], axes[1][1]], LOGREG, "breast_cancer_scale")
    plot_subset([axes[1][2], axes[1][3]], BNN, "breast_cancer_scale")

    for line, axlist in enumerate(axes):
        for col, ax in enumerate(axlist):
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.grid(b=True, which='major', alpha=0.8)
            ax.grid(b=True, which='minor', linestyle='--', alpha=0.2)

            from matplotlib.ticker import FormatStrFormatter

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))

            if col % 2 == 0:
                ax.set_ylabel("Neg. ELBO")
            else:
                ax.set_ylabel("Neg. Test LogLik")
            if line == 1:
                ax.set_xlabel("Epoch")

    axes[0][0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[0][1].yaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
    axes[0][2].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[0][3].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[1][0].yaxis.set_minor_formatter(FormatStrFormatter('%.2f'))
    axes[1][0].set_yticks([.15, .2, .25], minor=True)
    axes[1][1].yaxis.set_minor_formatter(FormatStrFormatter('%.2f'))
    axes[1][1].set_yticks([.14, .16, .18, .2], minor=True)
    axes[1][2].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[1][3].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    handles, labels = axes[0][0].get_legend_handles_labels()

    axes[0][0].legend(
        [handles[0], handles[2], handles[3], handles[4], handles[1]],
        [labels[0], labels[2], labels[3], labels[4], labels[1]],
        ncol=10, loc="upper center",
        bbox_to_anchor=(2.8, 1.6)
    )

    fig.text(
        .2, .86, "Bayesian Logistic Regression",
        fontweight=FONTWEIGHT,  # fontsize="xx-large"
    )
    fig.text(
        .65, .86, "Bayesian Neural Network",
        fontweight=FONTWEIGHT,  # fontsize="xx-large"
    )
    fig.text(
        .01, .65, "USPS",
        fontweight=FONTWEIGHT,  # fontsize="xx-large",
        rotation=90
    )
    fig.text(
        .01, .35, "Breast Cancer",
        fontweight=FONTWEIGHT,  # fontsize="xx-large",
        rotation=90
    )

    plot_and_save_as_needed("main_convplot", args)


def make_australian_covplot(args):
    FIGSCALE = 4
    fig = plt.figure(figsize=(4 * FIGSCALE, 1 * FIGSCALE))
    gs1 = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1])
    gs2 = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1])
    gs1.update(top=0.8, bottom=0.1, left=0.1, right=0.45, wspace=0.5, hspace=0.2)
    gs2.update(top=0.8, bottom=0.1, left=0.55, right=0.9, wspace=0.5, hspace=0.2)

    axes = [
        [fig.add_subplot(gs1[0, 0]), fig.add_subplot(gs1[0, 1]), fig.add_subplot(gs2[0, 0]), fig.add_subplot(gs2[0, 1])],
    ]

    blacklist = ["exact", "VOG"]

    CUSTOM_DISPLAY_NAME = {
        "bbb": "BBB",
        "slang-1": "ERR",
        "slang-8": "SLANG(1)",
        "slang-16": "SLANG(2)",
        "slang-32": "SLANG(3)",
        "slang-64": "ERR",
        "mf-exact": "ERR",
        "VOG-D": "ERR",
        "VON-D": "Mean-Field",
        "SLANG-V2-1": "SLANG(1)",
        "SLANG-V2-2": "ERR",
        "SLANG-V2-5": "SLANG(2)",
        "SLANG-V2-10": "SLANG(3)",
        "VOG": "ERR",
        "VON": "Full-Gaussian",
        "exact": "ERR",
    }

    def plot_subset(axes, conf, dataset):
        METHODS = list([m for m in conf.METHODS if not any([blacklisted in m for blacklisted in blacklist])])
        for m in METHODS:
            out = conf.load(dataset, m, args.datafolder)
            mask = out[0] < conf.X_LIMS[dataset][1]
            plot_metric_on(axes[0], out[0][mask], out[1][:, mask], m, conf, CUSTOM_DISPLAY_NAME)
            axes[0].set_xlim(conf.X_LIMS[dataset])
            axes[0].set_ylim(conf.Y_LIMS[dataset]["ELBO"])
            plot_metric_on(axes[1], out[0][mask], out[2][:, mask], m, conf, CUSTOM_DISPLAY_NAME)
            axes[1].set_xlim(conf.X_LIMS[dataset])
            axes[1].set_ylim(conf.Y_LIMS[dataset]["LL"])

    plot_subset([axes[0][0], axes[0][1]], LOGREG, "australian_scale")
    plot_subset([axes[0][2], axes[0][3]], BNN, "australian_scale")

    for line, axlist in enumerate(axes):
        for col, ax in enumerate(axlist):
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.grid(b=True, which='major', alpha=0.8)
            ax.grid(b=True, which='minor', linestyle='--', alpha=0.2)

            from matplotlib.ticker import FormatStrFormatter

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))

            if col % 2 == 0:
                ax.set_ylabel("Neg. ELBO")
            else:
                ax.set_ylabel("Neg. Test LogLik")
            if line == 1:
                ax.set_xlabel("Epoch")

    axes[0][0].yaxis.set_minor_formatter(FormatStrFormatter('%.2f'))
    axes[0][0].set_yticks([1], minor=False)
    axes[0][0].set_yticks([.8, .9, 1.1, 1.2], minor=True)
    axes[0][1].yaxis.set_minor_formatter(FormatStrFormatter('%.2f'))
    axes[0][2].yaxis.set_minor_formatter(FormatStrFormatter('%.2f'))
    axes[0][2].set_yticks([5], minor=True)
    axes[0][3].yaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
    axes[0][3].set_yticks([], minor=False)
    axes[0][3].set_yticks([.5, 1, 1.5, 2], minor=True)

    #axes[0][0].set_yticks([.15,.2,.25], minor=True)

    handles, labels = axes[0][0].get_legend_handles_labels()

    axes[0][0].legend(
        [handles[0], handles[2], handles[3], handles[4], handles[1]],
        [labels[0], labels[2], labels[3], labels[4], labels[1]],
        ncol=10, loc="upper center",
        bbox_to_anchor=(2.8, 1.32),  # bbox_to_anchor=(2.8,1.6)
    )

    fig.text(
        .2, .86, "Bayesian Logistic Regression",
        fontweight=FONTWEIGHT,  # fontsize="xx-large"
    )
    fig.text(
        .65, .86, "Bayesian Neural Network",
        fontweight=FONTWEIGHT,  # fontsize="xx-large"
    )

    plot_and_save_as_needed("australian_convplot", args)


def make_plot_poster(args):
    FIGSCALE = 4
    fig = plt.figure(figsize=(4 * FIGSCALE, 1.25 * FIGSCALE))
    gs1 = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1])
    gs2 = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[1, 1], height_ratios=[1])
    gs1.update(top=0.75, bottom=0.1, left=0.05, right=0.45, wspace=0.5, hspace=0.2)
    gs2.update(top=0.75, bottom=0.1, left=0.55, right=0.95, wspace=0.5, hspace=0.2)

    axes = [
        [fig.add_subplot(gs1[0, 0]), fig.add_subplot(gs1[0, 1]), fig.add_subplot(gs2[0, 0]), fig.add_subplot(gs2[0, 1])],
    ]

    blacklist = ["exact", "VOG"]

    CUSTOM_DISPLAY_NAME = {
        "bbb": "BBB",
        "slang-1": "ERR",
        "slang-8": "SLANG(1)",
        "slang-16": "SLANG(2)",
        "slang-32": "SLANG(3)",
        "slang-64": "ERR",
        "mf-exact": "ERR",
        "VOG-D": "ERR",
        "VON-D": "Mean-Field",
        "SLANG-V2-1": "SLANG(1)",
        "SLANG-V2-2": "ERR",
        "SLANG-V2-5": "SLANG(2)",
        "SLANG-V2-10": "SLANG(3)",
        "VOG": "ERR",
        "VON": "Full-Gaussian",
        "exact": "ERR",
    }

    def plot_subset(axes, conf, dataset):
        METHODS = list([m for m in conf.METHODS if not any([blacklisted in m for blacklisted in blacklist])])
        for m in METHODS:
            out = conf.load(dataset, m, args.datafolder)
            mask = out[0] < conf.X_LIMS[dataset][1]
            plot_metric_on(axes[0], out[0][mask], out[1][:, mask], m, conf, CUSTOM_DISPLAY_NAME)
            axes[0].set_xlim(conf.X_LIMS[dataset])
            axes[0].set_ylim(conf.Y_LIMS[dataset]["ELBO"])
            plot_metric_on(axes[1], out[0][mask], out[2][:, mask], m, conf, CUSTOM_DISPLAY_NAME)
            axes[1].set_xlim(conf.X_LIMS[dataset])
            axes[1].set_ylim(conf.Y_LIMS[dataset]["LL"])

    plot_subset([axes[0][0], axes[0][1]], LOGREG, "usps_3vs5")
    plot_subset([axes[0][2], axes[0][3]], BNN, "usps_3vs5")

    for line, axlist in enumerate(axes):
        for col, ax in enumerate(axlist):
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.grid(b=True, which='major', alpha=0.8)
            ax.grid(b=True, which='minor', linestyle='--', alpha=0.2)

            from matplotlib.ticker import FormatStrFormatter

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))

            if col % 2 == 0:
                ax.set_ylabel("Neg. ELBO", fontsize=20)
            else:
                ax.set_ylabel("Neg. Test LogLik", fontsize=20)
            if line == 1:
                ax.set_xlabel("Epoch")

    axes[0][0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[0][1].yaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
    axes[0][1].set_yticks([.2, .3, .4, .5], minor=True)
    axes[0][2].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axes[0][3].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    handles, labels = axes[0][0].get_legend_handles_labels()

    legHandler = axes[0][0].legend(
        [handles[0], handles[2], handles[3], handles[4], handles[1]],
        [labels[0], labels[2], labels[3], labels[4], labels[1]],
        ncol=10, loc="upper center",
        bbox_to_anchor=(2.8, 1.4),  # bbox_to_anchor=(2.8,1.6)
        fontsize=22
    )
    for line in legHandler.get_lines():
        line.set_linewidth(10.0)

    fig.text(
        .10, .785, "Bayesian Logistic Regression",
        fontweight=FONTWEIGHT, fontsize=32
    )
    fig.text(
        .55, .785, "Bayesian Neural Network",
        fontweight=FONTWEIGHT, fontsize=32
    )

    plot_and_save_as_needed("poster", args)


if __name__ == "__main__":
    args = CLI.get_parser().parse_args()
    dataset = CLI.get_dataset(args)

    print(args)

    make_main_covplot(args)
    make_australian_covplot(args)
    make_plot_poster(args)
