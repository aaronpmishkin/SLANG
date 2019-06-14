import argparse

arg_definitions = {
    '--save': {
        'dest': 'save', 'action': 'store_true', 'default': False,
        'help': (
            'Saves the generated plot in the current direction. ' +
            'File names depend on generated plots and datasets.')
    },
    '--noshow': {
        'dest': 'noshow', 'action': 'store_true', 'default': False,
        'help': (
            'Disable showing plot during script run. ' +
            'Useful if you just want to generate files.')
    },
    '--dataset': {
        'dest': 'dataset', 'action': 'store', 'default': "australian",
        'help': (
            'Dataset to plot [australian, breast, usps].')
    },
    '--datafolder': {
        'dest': 'datafolder', 'action': 'store', 'default': "",
        'help': (
            'Root of the folder containing the plotting data.')
    },
    '--save': {
        'dest': 'save', 'action': 'store_true', 'default': False,
        'help': (
            'Saves the generated plot in the current direction. ' +
            'File names depend on generated plots and datasets.')
    },
}


def get_parser():
    parser = argparse.ArgumentParser(
        description='Run Covariance Vizualisation experiments and generate plots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    for arg, arg_def in arg_definitions.items():
        parser.add_argument(arg, **arg_def)

    return parser


DATASET_MAPPING = {
    "australian": "australian_scale",
    "breast": "breast_cancer_scale",
    "usps": "usps_3vs5",
}


def get_dataset(args):
    return DATASET_MAPPING[args.dataset]
