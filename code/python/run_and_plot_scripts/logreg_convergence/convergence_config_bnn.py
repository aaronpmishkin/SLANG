import numpy as np

import experiments.base.slang_experiments as slang
import experiments.convergence_experiments.slang_convergence_final
import experiments.base.bbb_copy_slang as bbb
import experiments.convergence_experiments.bbb_convergence_final

from lib.utilities.record_utils import get_experiment_results

X_LIMS = {
    "australian_scale": [1, 500],
    "breast_cancer_scale": [1, 500],
    "usps_3vs5": [1, 500],
}

Y_LIMS = {
    "australian_scale": {
        "ELBO": [.8, 11],
        "LL": [0.32, 2],
    },
    "breast_cancer_scale": {
        "ELBO": [.6, 11],
        "LL": [0.07, 1],
    },
    "usps_3vs5": {
        "ELBO": [.35, 200],
        "LL": [.1, 10],
    }
}


DATASETS = ["australian_scale", "breast_cancer_scale", "usps_3vs5"]

Ns = {
    "australian_scale": 345,
    "breast_cancer_scale": 341,
    "usps_3vs5": 770,
}

DATASET_VILIB_NAME = {
    "australian_scale": "australian_presplit",
    "breast_cancer_scale": "breastcancer_presplit",
    "usps_3vs5": "usps_3vs5"
}

COLOR = {
    "bbb": [0.30, 0.75, 1.00],
    "slang-8": [0.50, 0.10, 0.20],
    "slang-16": [0.75, 0.20, 0.15],
    "slang-32": [1.00, 0.30, 0.10],
    "slang-64": [1.00, 0.60, 0.60],
}

DISPLAY_NAME = {
    "bbb": "BBB",
    #    "slang-1":"SLANG (L=1)",
    "slang-8": "SLANG (L=8)",
    "slang-16": "SLANG (L=16)",
    "slang-32": "SLANG (L=32)",
    "slang-64": "SLANG (L=64)",
}

METHODS = [
    "bbb",
    #    "slang-1",
    "slang-8",
    "slang-16",
    "slang-32",
    #    "slang-64",
]

MINIBATCH_SIZES = {
    "australian_scale": 32,
    "breast_cancer_scale": 32,
    "usps_3vs5": 64,
}


def get_methodName_and_L(method):
    if "slang-" in method:
        return "slang", int(method.replace("slang-", ""))
    else:
        return method, 0


def load(dataset, method, datafolder=None):
    print(dataset, method)

    method, L = get_methodName_and_L(method)

    if method == "slang":
        all_exps = slang.slang_base.get_variant("slang_convergence_final").variants

        def is_relevant(exp):
            return (DATASET_VILIB_NAME[dataset] in exp and "L_" + str(L) + "_" in exp)
        relevant_exps = [exp for exp in all_exps if is_relevant(exp)]

        results = get_experiment_results(slang.slang_base, "slang_convergence_final", relevant_exps)
    else:
        all_exps = bbb.bbb_copy_slang.get_variant("bbb_convergence_final").variants

        def is_relevant(exp):
            return DATASET_VILIB_NAME[dataset] in exp
        relevant_exps = [exp for exp in all_exps if is_relevant(exp)]

        results = get_experiment_results(bbb.bbb_copy_slang, "bbb_convergence_final", relevant_exps)

    nlls = []
    nlZs = []
    accs = []
    for r in results:
        nlZs.append(r['metric_history']['elbo_neg_ave'])
        nlls.append(r['metric_history']['test_pred_logloss'])
        accs.append(r['metric_history']['test_pred_accuracy'])

    x = np.array(range(len(nlZs[0]))) * MINIBATCH_SIZES[dataset] / Ns[dataset]

    return x + 1, np.vstack(nlZs), np.vstack(nlls), np.vstack(accs)
