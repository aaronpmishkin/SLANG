# @Author: amishkin
# @Date:   18-09-05
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-09-05


from artemis.experiments.ui import browse_experiments


'''
This minimal example demonstrates the use of Artemis to run experiments and
browse the records of past experiments. It will briefly showcase the programmic
and terminal interfaces that Artemis provides.
'''

# We will start by loading the bayes_by_backprop experiments.
import experiments.base.bayes_by_backprop as bbb

# There are base two experiment functions in the bbb module.
# 1) bbb_base: a "regular" experiment using Bayes by Backprop and the Adam optimizer.
# 2) bbb_cv: a cross validated experiment using Bayes by Backprop and the Adam optimizer.

# The experiment functions are annotated with @ExperimentFunction. The annotation
# transforms the functions into classes of type artemis.experiments.Experiment.
# They should not be called as functions, but used via the API provided by Artemis.

# All functions annotated with @ExperimentFunction must have default values for
# every argument. These default values are the default parameters of the experiment.
# Use the following to run a BBB experiment using the default parameters:

bbb.bbb_base.run(print_to_console=True, keep_record=True)

# Running this experiment will create an experiment record in ~/.artemis/experiments.
# This is the default location for the storage of all experiment records.

# This is an example Artemis programmic API. To run the same experiment using the
# command line interface, you can do the following:

# This will open the experiment browser. You should see a record for experiment we just ran.
bbb.bbb_base.browse()

# Use 'run 0', or '0' to run the default experiment when inside the experiment browser.
# To exit the experiment browser, use 'q'. The help command is also very useful...

# The bbb module also registers several "variants" of the two base experiments. Variants
# are a way of modifying the default inputs of an experiment class to create new
# experiments. "bbb.binclass" creates and registers a variant of bbb.bbb_base
# with no hidden layers and the logistic loss (e.g. simple logistic regression).

# Running this experiment variant is also easy:
bbb.bbb_base.get_variant('logistic regression').run(print_to_console=True,
                                                    keep_record=True)

# Variants of an experiment can be seen in the command line interface by browsing
# the base experiment.

bbb.bbb_base.browse()

# You should see bbb_base, and two variants, "logistic regression" and "bnn regression",
# listed in the command line interface. To browse all experiments that are currently loaded,
# including different base experiments, the following is used:

browse_experiments()

# You should see bbb_base and bbb_cv listed along with their variants that are registered
# the bbb module.

# IMPORTANT: Only variants that have been registered will appear in the browser
# when "bbb.bbb_base.browse()"" is used. Variants with experiment records in ~/.artmis
# that have not been registered in the current session are only shown by "browse_experiments()".


# Artemis experiment records save the console output, plots, and final return values
# of an experiment. They can be obtained using:

records = bbb.bbb_base.get_records()

# This should return a list with the record from the experiment we ran earlier.
# Our current experiment setup stores all the important information from an experiment
# (posterior distribution, metric log, final metrics, etc) in the experiment result
# object. To lookup and automatically unpickle the experiment object, use:

result = records[0].get_result()
print(result['final_metrics'])

# Another very useful feature is the ability to lookup experiment arguments using:
experiment_parameters = records[0].get_args()
print(experiment_parameters)
