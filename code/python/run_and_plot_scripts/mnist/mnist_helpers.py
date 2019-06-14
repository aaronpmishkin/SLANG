import numpy as np

import torch
from lib.data.datasets import Dataset, DEFAULT_DATA_FOLDER
from lib.metrics.metrics import softmax_predictive_accuracy
from lib.utilities.general_utilities import set_seeds
from experiments.base.slang_experiments import init_slang_experiment


def get_accuracy(experiment_base, experiment_name, variant, mc_10_multiplier, data_set=None):

    set_seeds(123)

    # Retrieve results

    exp = experiment_base.get_variant(experiment_name).get_variant(variant)
    record = exp.get_latest_record()
    if record.has_result():
        result = record.get_result()

        # Instantiate model and optimizer

        params = exp.get_args()

        if data_set is None:
            data_set = params['data_set']

        data = Dataset(data_set=data_set, data_folder=DEFAULT_DATA_FOLDER)
        train_set_size=data.get_train_size()

        model, predict_fn, kl_fn, closure_factory, optimizer = init_slang_experiment(data = data,
                                                                                     model_params = params['model_params'],
                                                                                     optimizer_params = params['optimizer_params'],
                                                                                     train_set_size = train_set_size,
                                                                                     use_cuda = torch.cuda.is_available())

        # Get state dicts

        model_state_dict = result['model']
        optim_state_dict = result['optimizer']

        # Set model and optimizer state

        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optim_state_dict)

        # Make predictions

        test_x, test_y = data.load_full_test_set()

        for i in range(mc_10_multiplier):
            print(i)
            with torch.no_grad():
                pred_y = predict_fn(test_x, 10)
                if i==0:
                    preds = pred_y
                else:
                    preds = torch.cat([preds, pred_y], 0)

        # Compute accuracy

        pred_acc = softmax_predictive_accuracy(preds, test_y).item()
    else:
        pred_acc = np.nan

    return pred_acc