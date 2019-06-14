import argparse
import subprocess
import os
import sys

"""
Command line arguments
"""


def parse():
    parser = argparse.ArgumentParser(description='Main Runner')

    experiment_choice_options = [
        ["-exp1", "--logreg-table", ""],
        ["-exp2", "--logreg-convergence", ""],
        ["-exp3", "--logreg-vizualization", ""],
        ["-exp4", "--bnn-uci-table", ""],
        ["-exp5", "--bnn-mnist-table", ""],
        ["-exp6", "--bnn-convergence", ""],
    ]
    experiment_choice = parser.add_mutually_exclusive_group(required=True)

    for c in experiment_choice_options:
        experiment_choice.add_argument(c[0], c[1], action="store_true", help=c[2])

    parser.add_argument("-run", "--run-optimizer", action="store_true",
                        help=("Runs the optimizer for the experiment. "
                              "If not set, will use pretrained data."),
                        default=False)

    parser.add_argument("-f", "--force", action="store_true",
                        help=("Disable warnings."),
                        default=False)
    parser.add_argument("-n", "--no-exec", action="store_true",
                        help=("Does not execute code, just print the commands to run."),
                        default=False)

    return parser.parse_args()


def path_to(rel_path):
    r"""
    Makes an absolute path from a relative path from the `code` folder.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *rel_path.split("/"))


LONG = "Running this experiment can be long. Are you sure you want to continue?"
MATLAB_DATA = "Are you sure the data is correctly set in `user/data/`?"
ARTEMIS_DATA = "Are you sure the data is correctly set in `~/.artemis/experiments`?"


def confirm(question):
    while True:
        sys.stdout.write(question + " [y/n]: ")
        choice = input().lower()
        if choice in ["y", "n"]:
            if choice == "n":
                sys.exit()
            else:
                return
        else:
            sys.stdout.write("Please type with 'y' or 'n'.\n")


if __name__ == "__main__":
    args = parse()
    RUN_EXPERIMENT = args.run_optimizer

    def ask_confirm(q):
        if not args.force:
            confirm(q)

    def call(cmd):
        if args.no_exec:
            print(cmd)
        else:
            subprocess.call(cmd, shell=True, stderr=subprocess.STDOUT)

    if args.logreg_table:
        if RUN_EXPERIMENT:
            ask_confirm(LONG)
            call("matlab -nodesktop -nodisplay -nosplash -r 'run(\"" + path_to("matlab/experiments/reproduce_log_reg_table.m") + "\");exit;'")
        else:
            ask_confirm("Are you sure the data is correctly set in `" + path_to("user/data/") + "`?")
            call("matlab -nodesktop -nodisplay -nosplash -r 'run(\"" + path_to("matlab/tables/make_paper_tables_1_7.m") + "\");exit;'")

    elif args.logreg_convergence or args.bnn_convergence:
        print("--logreg-convergence and --bnn-convergence refer to the same plot.")
        print("This plot relies on data for Logistic Regression and BNN.")
        if RUN_EXPERIMENT:
            ask_confirm("Are you sure you want to run both?")
            call("python " + path_to("python/submitters/convergence/submit_bbb_convergence.py"))
            call("python " + path_to("python/submitters/convergence/submit_slang_convergence.py"))
            call("matlab -nodesktop -nodisplay -nosplash -r 'run(\"" + path_to("matlab/experiments/reproduce_log_reg_convergence.m") + "\");exit;'")
            call("python " + path_to("python/run_and_plot_scripts/logreg_convergence/convergence_main.py --noshow --save --datafolder '" + path_to("data/final-convergence-comparison") + "'"))
        else:
            ask_confirm("Are you sure the data is correctly set in `" + path_to("user/data/") + "` and in `~/.artemis/experiments`?")
            call("python " + path_to("python/run_and_plot_scripts/logreg_convergence/convergence_main.py --noshow --save --datafolder '" + path_to("paper_experiment_data/final-convergence-comparison") + "'"))

    elif args.logreg_vizualization:
        if RUN_EXPERIMENT:
            ask_confirm(LONG)
            call("matlab -nodesktop -nodisplay -nosplash -r 'run(\"" + path_to("matlab/experiments/reproduce_log_reg_table.m") + "\");exit;'")
            call("python " + path_to("python/run_and_plot_scripts/logreg_covviz/logreg_covviz_plotter.py") + " --save --noshow --datafolder '" + path_to("data/final_log_reg_table") + "'")
        else:
            ask_confirm(MATLAB_DATA)
            call("python " + path_to("python/run_and_plot_scripts/logreg_covviz/logreg_covviz_plotter.py") + " --save --noshow --datafolder '" + path_to("paper_experiment_data/final_log_reg_table") + "'")

    elif args.bnn_uci_table:
        if RUN_EXPERIMENT:
            ask_confirm(LONG)
            call("python " + path_to("python/submitters/uci/slang/submit_boston.py"))
            call("python " + path_to("python/submitters/uci/slang/submit_concrete.py"))
            call("python " + path_to("python/submitters/uci/slang/submit_energy.py"))
            call("python " + path_to("python/submitters/uci/slang/submit_kin8nm.py"))
            call("python " + path_to("python/submitters/uci/slang/submit_naval.py"))
            call("python " + path_to("python/submitters/uci/slang/submit_powerplant.py"))
            call("python " + path_to("python/submitters/uci/slang/submit_wine.py"))
            call("python " + path_to("python/submitters/uci/slang/submit_yacht.py"))
            call("python " + path_to("python/submitters/uci/bbb/submit_boston.py"))
            call("python " + path_to("python/submitters/uci/bbb/submit_concrete.py"))
            call("python " + path_to("python/submitters/uci/bbb/submit_energy.py"))
            call("python " + path_to("python/submitters/uci/bbb/submit_kin8nm.py"))
            call("python " + path_to("python/submitters/uci/bbb/submit_naval.py"))
            call("python " + path_to("python/submitters/uci/bbb/submit_powerplant.py"))
            call("python " + path_to("python/submitters/uci/bbb/submit_wine.py"))
            call("python " + path_to("python/submitters/uci/bbb/submit_yacht.py"))
            call("python " + path_to("python/run_and_plot_scripts/uci/make_table_2.py"))
        else:
            ask_confirm(ARTEMIS_DATA)
            call("python " + path_to("python/run_and_plot_scripts/uci/make_table_2.py"))

    elif args.bnn_mnist_table:
        if RUN_EXPERIMENT:
            ask_confirm(LONG)
            call("python " + path_to("python/submitters/mnist/submit_mnist_experiment.py"))
            call("python " + path_to("python/run_and_plot_scripts/mnist/make_table_3.py"))
        else:
            ask_confirm(ARTEMIS_DATA)
            call("python " + path_to("python/run_and_plot_scripts/mnist/make_table_3_from_continues.py"))
