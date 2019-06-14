# @Author: aaronmishkin
# @Date:   18-08-24
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-08-24

import numpy as np
from matplotlib import pyplot as plt

def plot_objective(record):
    print('===== CREATING PLOT OF RECORD {} NOW ===='.format(record.get_id()))
    results_dict = record.get_result()

    objective_history = results_dict['objective_history']

    # training epochs
    epochs = np.arange(len(objective_history))

    plt.plot(epochs, objective_history)
    plt.grid()
    plt.xlabel('Training Epoch')
    plt.ylabel('Objective Value')
    plt.title(record.get_id())
    plt.show()
