import argparse

arg_definitions = {
    '--run':{
        'dest':'run', 'action':'store_true', 'default':False,
        'help':(
            'Run the experiments before plotting. ' +
            'If not given, the last run of the experiments is used.')
    },
    '--noshow':{
        'dest':'noshow', 'action':'store_true', 'default':False,
        'help':(
            'Disable showing plot during script run. ' +
            'Useful if you just want to generate files.')
    },
    '--save':{
        'dest':'save', 'action':'store_true', 'default':False,
        'help':(
            'Saves the generated plot in the current direction. ' +
            'File names depend on generated plots and datasets.')
    },
    '--dataset_id':{
        'dest':'dataset_id', 'type':int, 'default':0,
        'help':'[0,1,2] 0: Default, 1:Add outlier, 2:Add gap'
    },
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Covariance Vizualisation experiments and generate plots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    for arg, arg_def in arg_definitions.items():
        parser.add_argument(arg, **arg_def)

    return parser.parse_args()
