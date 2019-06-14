# @Author: amishkin
# @Date:   18-08-17
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-08-24

def print_progress(epoch, num_epochs, metric_history):
    message = 'Epoch [{}/{}]'.format(epoch+1, num_epochs)

    for name in metric_history.keys():
        message = (message + ', ' + name + ': {:.4f}').format(metric_history[name][-1])

    print(message)

def print_objective(epoch, num_epochs, objective_value):

    # Print average objective from last epoch
    print('Epoch [{}/{}], Objective: {:.4f}'.format(
            epoch+1,
            num_epochs,
            objective_value))

def print_cv_progress(split, n_splits, epoch, num_epochs, metric_history):
    message = 'Split [{}/{}], Epoch [{}/{}]'.format(split+1, n_splits, epoch+1, num_epochs)

    for name in metric_history.keys():
        message = (message + ', ' + name + ': {:.4f}').format(metric_history[name][-1])

    print(message)

def print_cv_objective(split, n_splits, epoch, num_epochs, objective_value):

    # Print average objective from last epoch
    print('Split [{}/{}], Epoch [{}/{}], Objective: {:.4f}'.format(
        split+1,
        n_splits,
        epoch+1,
        num_epochs,
        objective_value))
