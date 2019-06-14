# @Author: amishkin
# @Date:   18-09-07
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-09-07

import subprocess
import os

def submit_python_jobs(name, variants, method='BBB', cv=0):
    subprocess.call("pwd", shell=True)

    basedir = os.path.dirname(os.path.abspath(__file__))

    for i, variant in enumerate(variants):
        command = "python "+os.path.join(basedir, 'run_experiment.py')+" --name=\"" + str(name) + "\" --variant=\"" + str(variant) + "\" --method=\"" + str(method) + "\" --cv=" + str(cv)
        print(command)
        exit_status = subprocess.call(command, shell=True)
        if exit_status == 1:
            print("Job {0} failed to submit".format(command))
    print("Done submitting jobs!")
