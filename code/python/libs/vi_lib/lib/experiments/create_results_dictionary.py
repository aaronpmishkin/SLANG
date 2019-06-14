# @Author: amishkin
# @Date:   18-08-17
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-08-24

from lib.utilities.general_utilities import cast_to_cpu

def create_results_dictionary(save_options, final_metrics, metric_history, objective_history, model, optimizer):

    results_dict = { 'final_metrics': final_metrics }

    if save_options['metric_history']:
        results_dict['metric_history'] = metric_history

    if save_options['objective_history']:
        results_dict['objective_history'] = objective_history

    if save_options['model']:
        results_dict['model'] = model.cpu().state_dict()

    if save_options['optimizer']:
        try:
            results_dict['optimizer'] = optimizer.cpu().state_dict()
        except AttributeError:
            results_dict['optimizer'] = cast_to_cpu(optimizer.state_dict())

    return results_dict
