import os
import numpy as np
import scipy as sp

X_LIMS = {
    "australian_scale": [1, 50],
    "breast_cancer_scale": [1, 20],
    "usps_3vs5": [1, 20],
}

Y_LIMS = {
    "australian_scale": {
        "ELBO": [.8, 1.2],
        "LL": [0.48, .7],
    },
    "breast_cancer_scale": {
        "ELBO": [.14, .25],
        "LL": [0.135, .2],
    },
    "usps_3vs5": {
        "ELBO": [.25, 10],
        "LL": [.15, .6],
    }
}

DATASETS = ["australian_scale", "breast_cancer_scale", "usps_3vs5"]
RESULTS_DIRECTORY = "final-convergence-comparison"

Ns = {
    "australian_scale": 345,
    "breast_cancer_scale": 341,
    "usps_3vs5": 770,
}

COLOR = {
    "exact": [0.50, 0.50, 0.50],
    "mf-exact": [0.30, 0.30, 0.30],
    #"VON":          [0.00,  0.30,   0.90],
    "VON": [0.00, 0.00, 0.00],
    "VON-D": [0.30, 0.75, 1.00],
    "VOG": [0.00, 0.80, 0.00],
    "VOG-D": [0.00, 0.50, 0.00],
    "SLANG-V2-1": [0.50, 0.10, 0.20],
    "SLANG-V2-2": [0.60, 0.15, 0.16],
    "SLANG-V2-5": [0.75, 0.20, 0.15],
    "SLANG-V2-10": [1.00, 0.30, 0.10],
}

DISPLAY_NAME = {
    "mf-exact": "MF-Exact",
    "VOG-D": "EF-Diag",
    "VON-D": "Hessian-Diag",
    "SLANG-V2-1": "SLANG (L=1)",
    "SLANG-V2-2": "SLANG (L=2)",
    "SLANG-V2-5": "SLANG (L=5)",
    "SLANG-V2-10": "SLANG (L=10)",
    "VOG": "EF",
    "VON": "Hessian",
    "exact": "Exact",
}

METHODS = [
    "VOG-D",
    "VON-D",
    "mf-exact",
    "VOG",
    "VON",
    "exact",
    "SLANG-V2-1",
    #    "SLANG-V2-2",
    "SLANG-V2-5",
    "SLANG-V2-10",
]

MINIBATCH_SIZES = {
    "australian_scale": 32,
    "breast_cancer_scale": 32,
    "usps_3vs5": 64,
}

POSSIBLE_RESTARTS = [1, 10]

RESTARTS = {
    "exact": POSSIBLE_RESTARTS[0],
    "mf-exact": POSSIBLE_RESTARTS[0],
    "VON": POSSIBLE_RESTARTS[1],
    "VOG": POSSIBLE_RESTARTS[1],
    "VON-D": POSSIBLE_RESTARTS[1],
    "VOG-D": POSSIBLE_RESTARTS[1],
    "SLANG-V2-1": POSSIBLE_RESTARTS[1],
    "SLANG-V2-2": POSSIBLE_RESTARTS[1],
    "SLANG-V2-5": POSSIBLE_RESTARTS[1],
    "SLANG-V2-10": POSSIBLE_RESTARTS[1],
}
SELECTED_LEARNING_RATES = {
    "australian_scale": {
        "VON": 0.0117,
        "VOG": 0.0117,
        "VON-D": 0.0215,
        "VOG-D": 0.0215,
        "SLANG-V2-1": 0.0117,
        "SLANG-V2-2": 0.0117,
        "SLANG-V2-5": 0.0117,
        "SLANG-V2-10": 0.0117,
    },
    "breast_cancer_scale": {
        "VON": 0.0398,
        "VOG": 0.0398,
        "VON-D": 0.0215,
        "VOG-D": 0.0215,
        "SLANG-V2-1": 0.0215,
        "SLANG-V2-2": 0.0398,
        "SLANG-V2-5": 0.0398,
        "SLANG-V2-10": 0.0398,
    },
    "usps_3vs5": {
        "VON": 0.0398,
        "VOG": 0.0398,
        "VON-D": 0.0063,
        "VOG-D": 0.0063,
        "SLANG-V2-1": 0.0117,
        "SLANG-V2-2": 0.0117,
        "SLANG-V2-5": 0.0215,
        "SLANG-V2-10": 0.0398,
    },
}


def get_methodName_and_L(methodName):
    if "SLANG-V2-" in methodName:
        return "SLANG-V2", int(methodName.replace("SLANG-V2-", ""))
    else:
        return methodName, 0


def get_minibatch(m, d):
    if "exact" in m:
        return 1
    else:
        return MINIBATCH_SIZES[d]


def get_learningrate(m, d):
    if "exact" in m:
        return 0
    else:
        return SELECTED_LEARNING_RATES[d][m]


def read_matlab_results(filename):
    tmpVal = sp.io.loadmat(filename)
    return tmpVal["nlZ"][0], tmpVal["log_loss"][0]


def get_restart(method, dataset):
    return RESTARTS[method]


def make_filename(dataset, methodname, M, L, R, lr):
    return "_".join([
        dataset, methodname,
        "M", str(M),
        "L", str(L),
        "K", str(0),
        "beta", str(lr),
        "alpha", str(lr),
        "decay", str(0),
        "restart", str(R),
    ]) + ".mat"


def interpolate_nans(x):
    nans, xx = np.isnan(x), lambda z: z.nonzero()[0]
    x[nans] = np.interp(xx(nans), xx(~nans), x[~nans])
    return x


def load(dataset, method, RESULTS_DIRECTORY=""):
    print(dataset, method)

    methodname, L = get_methodName_and_L(method)

    M = get_minibatch(method, dataset)
    lr = get_learningrate(method, dataset)
    folder = os.path.join(os.path.join(RESULTS_DIRECTORY, dataset), methodname)
    R = get_restart(method, dataset)

    nlZs = []
    nlls = []
    for r in range(R):
        fullpath = os.path.join(folder, make_filename(dataset, methodname, M, L, r + 1, lr))
        tmp_res = read_matlab_results(fullpath)
        if "exact" in method:
            nlZs.append(interpolate_nans(tmp_res[0]))
            nlls.append(interpolate_nans(tmp_res[1]))
        else:
            nlZs.append(tmp_res[0])
            nlls.append(tmp_res[1])

    x = np.array(range(len(nlZs[0])))

    if "exact" not in method:
        x = x * MINIBATCH_SIZES[dataset] / Ns[dataset]

    return x + 1, np.vstack(nlZs) * np.log2(np.exp(1)) / Ns[dataset], np.vstack(nlls), None
