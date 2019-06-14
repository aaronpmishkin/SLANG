# @Author: amishkin
# @Date:   18-08-17
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   amishkin
# @Last modified time: 18-08-23

def evaluate_model(predict_fn, train_metrics, test_metrics, metric_history, x_train, y_train, x_test, y_test, eval_mc_samples, normalize):

    # Normalize train x
    if normalize['x']:
        x_train = (x_train-normalize['x_means'])/normalize['x_stds']

    # Get train predictions
    preds = predict_fn(x=x_train, mc_samples=eval_mc_samples)

    # Unnormalize train predictions
    if normalize['y']:
        preds = [normalize['y_mean'] + normalize['y_std'] * p for p in preds]

    for name in train_metrics.keys():
        metric_history[name].append(train_metrics[name](preds, y_train).detach().cpu().item())

    # Normalize test x
    if normalize['x']:
        x_test = (x_test-normalize['x_means'])/normalize['x_stds']

    # Get test predictions
    preds = predict_fn(x=x_test, mc_samples=eval_mc_samples)

    # Unnormalize test predictions
    if normalize['y']:
        preds = [normalize['y_mean'] + normalize['y_std'] * p for p in preds]

    for name in test_metrics.keys():
        metric_history[name].append(test_metrics[name](preds, y_test).detach().cpu().item())

    return metric_history
