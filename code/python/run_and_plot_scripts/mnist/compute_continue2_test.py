import experiments.mnist.slang_mnist_continue as slang_mnist_continue
experiment_base = slang_mnist_continue.slang_continue

from experiments.mnist.slang_mnist_continue2 import experiment_name, variants

from run_and_plot_scripts.mnist.mnist_helpers import get_accuracy
import pandas as pd

#############################
## Compute test accuracies ##
#############################

accuracies = []

for i, variant in enumerate(variants):
    accuracies.append(get_accuracy(experiment_base, experiment_name, variant, mc_10_multiplier = 100, data_set='mnist'))
    print("Index [{}/{}] Done!".format(1+i, len(variants)))

df_final = pd.DataFrame(dict(variant=variants, accuracy=accuracies))
df_final['error'] = ['{:.2f}%'.format(err) for err in (1 - df_final['accuracy'].values) * 100]
df_final.to_csv("df_continue2_test.csv")
