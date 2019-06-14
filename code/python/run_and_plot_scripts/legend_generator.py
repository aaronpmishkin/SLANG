r"""
"""

import argparse
import pdb 

import os 

import numpy as np

import matplotlib
import matplotlib.pyplot as plt 

import lib.utilities.plotting as plotutils

################################################################################
# GENERAL PLOTTING CONFIGS
################################################################################

GLOBAL_FONT_SIZE = 14
FONTWEIGHT = "bold"
COLORMAP = plt.cm.seismic

matplotlib.rcParams.update({'font.size': GLOBAL_FONT_SIZE})

################################################################################
# MAGIC NUMBERS/CONSTANTS
################################################################################

METHOD_DISPLAY_TITLE = {
    "exact":"Full Gaussian",
    "mf-exact":"Mean Field",
    "slang 1":"SLANG (Rank 1)",
    "slang 5":"SLANG (Rank 5)",
    "slang 10":"SLANG (Rank 10)",
}

METHODS = ["exact", "mf-exact", "slang 1", "slang 5", "slang 10"]
METHODS_WITH_COV = ["exact", "slang 1", "slang 5", "slang 10"]
METHODS_APPROX = ["mf-exact", "slang 1", "slang 5", "slang 10"]
METHODS_APPROX_EXTR = ["mf-exact", "slang 5", "slang 10"]

COLORS = {
    "exact":[0,0,0],
    "mf-exact":[0,0,1],
    "slang 1":[.5,0,0],
    "slang 5":[1,0,0],
    "slang 10":[1,.6,.6],
}

BASE_SETTINGS = {
    "exact":{"label":"Full Gaussian", "color":COLORS["exact"], "linewidth":6},
    "mf-exact":{"label":"MF", "color":COLORS["mf-exact"], "linewidth":2},
    "slang 1":{"label":"SLANG-1", "color":COLORS["slang 1"], "linewidth":3},
    "slang 5":{"label":"SLANG-5", "color":COLORS["slang 5"], "linewidth":3},
    "slang 10":{"label":"SLANG-10", "color":COLORS["slang 10"], "linewidth":3},
}

################################################################################
# CLI ARGUMENTS
################################################################################

arg_definitions = {
    '--save':{
        'dest':'save', 'action':'store_true', 'default':False,
        'help':(
            'Saves the generated plot in the current direction.' +
            'File names depend on generated plots and datasets.')
    },
    '--noshow':{
        'dest':'noshow', 'action':'store_true', 'default':False,
        'help':(
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
# PLOTTING FUNCTIONS
################################################################################

def make_legend():

    fig_dummy = plt.figure(figsize=(5, 5))
    ax_dummy = fig_dummy.add_subplot(111)
    
    fig_legend = plt.figure(figsize=(15, 1))
    
    lines = []
    for name, settings in BASE_SETTINGS.items():
        lines.append(ax_dummy.plot(range(2), range(2), **settings)[0])
        
    fig_legend.legend(
        lines, 
        list([METHOD_DISPLAY_TITLE[k] for k in BASE_SETTINGS.keys()]), 
        ncol=len(BASE_SETTINGS),
        loc='center',
    )
    
    plt.close(fig_dummy)

    if not args.noshow:
        plt.show()
    if args.save:
        fig_legend.savefig("legend.pdf", bbox_inches='tight')
    
################################################################################
# MAIN
################################################################################

if __name__ == "__main__":
    args = parser.parse_args()
    
    print("Argument passed:")
    print("    Showing output: " + str(not args.noshow))
    print("    Saving output: " + str(args.save))
    
    make_legend()
