import experiments.base.slang_experiments as slang
experiment_base = slang.slang_base

from experiments.mnist.slang_mnist_val import experiment_name, variants

from run_and_plot_scripts.mnist.mnist_helpers import get_accuracy

import numpy as np
import pandas as pd

df = pd.read_csv("df_val.csv")

##################################
## Select best parameters per L ##
##################################

idx = df.groupby(['L'])['final_val_accuracy'].transform(min) == df['final_val_accuracy']
df_best = df[idx]

df_best = df_best.iloc[[0,3,4,5,6,7]]

#############################
## Compute test accuracies ##
#############################

df_best['final_test_accuracy'] = np.nan
    
for i, (index, row) in enumerate(df_best.iterrows()):
    df_best.loc[index, 'final_test_accuracy'] = get_accuracy(experiment_base, experiment_name, row['variant'], mc_10_multiplier = 100, data_set='mnist')
    print("Index [{}/{}] Done!".format(1+i, len(df_best)))

df_best['final_test_error'] = ['{:.2f}%'.format(err) for err in (1 - df_best['final_test_accuracy'].values) * 100]
df_best.to_csv("df_val_test.csv", index=False)



