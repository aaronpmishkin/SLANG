# @Author: amishkin
# @Date:   18-09-07
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-09-13

import argparse

import experiments.experiment_list as experiments

# Load arguments.
parser = argparse.ArgumentParser(description="Run")
parser.add_argument("--name", dest="name", default=None, type=str, help="Experiment name.")
parser.add_argument("--variant", dest="variant", default=None, type=str, help="Experiment variant.")
parser.add_argument("--method", dest="method", default=None, type=str, help="Method to run.")
parser.add_argument("--cv", dest="cv", default=False, type=int, help="Run cross validation?")

args = parser.parse_args()

print("args:", args)

# Run the Experiment
if args.cv == 0: # it's not cross-validation
    if args.method == "SLANG":
        experiments.slang_base.get_variant(args.name).get_variant(args.variant).run()
    elif args.method == "BBB":
        experiments.bbb_base.get_variant(args.name).get_variant(args.variant).run()
    elif args.method == "BBB_DECAY":
        experiments.bbb_decay_base.get_variant(args.name).get_variant(args.variant).run()
    elif args.method == "BBB_COPY_SLANG":
        experiments.bbb_copy_slang.get_variant(args.name).get_variant(args.variant).run()
    elif args.method == "SLANG_CONTINUE":
        experiments.slang_continue.get_variant(args.name).get_variant(args.variant).run()
    elif args.method == "SLANG_COMPLETE":
        experiments.slang_complete.get_variant(args.name).get_variant(args.variant).run()
    elif args.method == "UCI_SLANG_BO":
        experiments.uci_slang_bo.get_variant(args.name).get_variant(args.variant).run()
    elif args.method == "UCI_BBB_BO":
        experiments.uci_bbb_bo.get_variant(args.name).get_variant(args.variant).run()
    else:
        raise Exception('Unknown method '+args.method)
else: # it is cross-validation
    if args.method == "SLANG":
        experiments.slang_cv.get_variant(args.name).get_variant(args.variant).run()
    elif args.method == "BBB":
        experiments.bbb_cv.get_variant(args.name).get_variant(args.variant).run()
    else:
        raise Exception('Unknown name '+args.method)
