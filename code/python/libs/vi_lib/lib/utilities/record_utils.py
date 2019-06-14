# @Author: amishkin
# @Date:   18-09-07
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-09-07

import math
import numpy as np

'''
Assorted helper functions for dealing with experiment records.
'''


# Assumes that you want the most recent experiment record
def get_experiment_results(experiment_base, experiment_name, variants):
    results = []

    for variant in variants:
        exp = experiment_base.get_variant(experiment_name).get_variant(variant)
        record = exp.get_latest_record()
        if record.has_result():
            result = record.get_result()
            results.append(result)

    return results

def get_metric_histories(results):
    metric_histories = []

    for result in results:
        metric_histories.append(result['metric_history'])

    return metric_histories

def get_final_metrics(results):
    final_metrics = []

    for result in results:
        final_metrics.append(result['final_metrics'])

    return final_metrics

def summarize_metric_histories(results):
    summary = {}
    for _, key in enumerate(results[0]['metric_history']):
        values = None
        for result in results:
            if values is None:
                values = np.array([result['metric_history'][key]])
            else:
                values = np.append(values, np.array([result['metric_history'][key]]), axis=0)
        summary[key] = {'mean': np.mean(values, axis=0), 'sd': np.std(values, axis=0), 'se': np.std(values, axis=0) / math.sqrt(len(results))}

    return summary


def summarize_final_metrics(results):
    summary = {}
    for _, key in enumerate(results[0]['final_metrics']):
        values = []
        for result in results:
            values.append(result['final_metrics'][key])
        summary[key] = {'mean': np.mean(values), 'sd': np.std(values), 'se': np.std(values) / math.sqrt(len(results))}

    return summary
