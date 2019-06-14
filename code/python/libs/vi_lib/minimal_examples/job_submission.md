# Submitting Jobs

This document will briefly introduce how to submit jobs to RAIDEN or another parallel computation platform using Artemis.

## Relevant Files

#### Bash Files

`profiled_qsub_execute.sh` - a bash script that runs whatever python script it is provided **with** Python's built-in cProfiler enabled. This script is typically passed to a job scheduler to run a profiled experiment.

`qsub_execute.sh` - a bash script that runs whatever python script it is provided. This script is typically passed to a job scheduler to run a single experiment.

#### Python Files

`experiments.{{some-experiment-definition-file}}.py` - The file where the user defines the main variant used for their experiment and a list of sub-variants that will actually be executed.

`experiments.experiment_list.py` - This file lists all registered Artemis experiments. It wildcard imports all defined experiments and exports them under the `experiment_list` module. You **must** wildcard import your experiment definition file (e.g. `experiments.{{some-experiment-definition-file}}.py`) in `experiment_list` for the job submission system to work.

`lib.experiments.submitters.py` - This file contains functions for submitting experiments to different schedulers/systems, or with additional functionality like the cProfiler. It is here that the different experiment (sub)variants are iterated over and a job for each is submitted to scheduler/system.

`lib.experiments.run_experiment.py` - A Python script that runs an experiment given it's base *method*, main variant *name*, and a *sub-variant*. Note that if you define a new base method (i.e. a new optimizer or model), it must added to the switch block in this file.

`submit_jobs.py` - The Python script that is called to submit a list of jobs.


## Submitting Jobs in Parallel

#### 0. BEFORE YOU DO ANYTHING

Make sure that you have **TorchUtils** installed. Make sure that you have **vi_lib** installed. You can do this with:

```pip install -e . ```

Now make sure that you have the following libraries installed:

PyTorch - must be version `0.4.1`, Numpy, MatPlotLib, Artemis-ml, and TorchVision.

#### 1. Create an experiment definition file.
This file should extend a base experiment (e.g. `slang_experiments.slang_base`, `bayes_by_backprop.bbb_base`) and then create 1-to-many sub-variants of this base experiment. Your base experiment *must* be given a string name that is passed to Artemis when it is created --- this name will be used to lookup the experiment definition. Similarly, the sub-variants that you create must be given names. See `experiments/minimial_example.py` for an example of this file.

#### 2. Wildcard import your experiment definition file into `experiments/experiment_list.py`

Exporting all experiments from a single module allows `run_experiment.py` to easily execute experiment variants given their names and base_experiment. It also makes browsing experiments using Artemis' graphical interface more convenient.

#### 3. Figure out what submitter you need to use.

Self explanatory.

#### 4. Modify `submit_jobs.py`

Modify `submit_jobs.py` to use the submitter; base method; experiment name; and (sub)-variant names corresponding to the experiments you want to execute.

#### 5. Insure that that your default data location is correct.

You may need to change the file path in `lib.data.datasets.py` to point to the
location of your datasets.

#### 6. Insure that `qsub_execute.sh` `cd` to the correct directory

You want them to `cd` into `vi_lib`, wherever it has been installed.

#### 7. >> `python submit_jobs.py`

Call submit_jobs.py.


## Retrieving Submitted Jobs

Artemis will save the results of your submitted jobs to `~/.artemis/experiments`. A handy way to obtain the results of your experiments is the following rsyc snippet:

```>>> rsync -Pav -e "ssh -i ~/{{path_to_private_key}}" {{user}}@raiden.riken.jp:/home/{{user}}/.artemis/experiments/. ~/.artemis/experiments/'```

You can use the functions in `lib.utilities.record_utils` to lookup experiment results and summarize metrics over multiple runs.
