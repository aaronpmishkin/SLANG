import experiments.base.slang_experiments as slang
experiment_base = slang.slang_base

from experiments.mnist.slang_mnist_val import experiment_name, variants

from run_and_plot_scripts.mnist.mnist_helpers import get_accuracy

#############################################
## Create dataframe for inspecting results ##
#############################################

import numpy as np
import pandas as pd

df = pd.DataFrame(data=dict(variant=variants, L=None, pp=None, dec=None, runtime=None, train_accuracy=None, train_logloss=None, test_accuracy=None, test_logloss=None, elbo=None))

for i, var in enumerate(variants):
    # Fill parameters
    df.loc[i, "L"] = int(var.split("L_")[1].split("_")[0])
    df.loc[i, "pp"] = float(var.split("pp_")[1].split("_")[0])
    df.loc[i, "dec"] = float(var.split("dec_")[1])

    # Retrieve record
    exp = experiment_base.get_variant(experiment_name).get_variant(var)
    record = exp.get_latest_record()

    # Extract runtime
    info = record.info
    split_string = info.get_text().split("\nExpInfoFields.RUNTIME: ")
    if len(split_string) > 1:
        runtime = float(split_string[1].split("\n")[0])
        df.loc[i, "runtime"] = runtime/3600

    # Extract metrics
    if record.has_result():
        result = record.get_result()
        df.loc[i, "train_accuracy"] = result['final_metrics']['train_pred_accuracy'][0]
        df.loc[i, "train_logloss"] = result['final_metrics']['train_pred_logloss'][0]
        df.loc[i, "test_accuracy"] = result['final_metrics']['test_pred_accuracy'][0]
        df.loc[i, "test_logloss"] = result['final_metrics']['test_pred_logloss'][0]
        df.loc[i, "elbo"] = result['final_metrics']['elbo_neg_ave'][0]

###################################
## Compute validation accuracies ##
###################################

df['final_val_accuracy'] = None

for index, row in df.iterrows():
    row['final_val_accuracy'] = get_accuracy(experiment_base, experiment_name, row['variant'], mc_10_multiplier = 100)
    print("Index [{}/{}] Done!".format(1+index, len(df)))

df.to_csv("df_val.csv", index=False)
