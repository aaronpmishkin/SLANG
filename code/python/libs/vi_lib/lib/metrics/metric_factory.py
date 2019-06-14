# @Author: amishkin
# @Date:   18-08-17
# @Email:  amishkin@cs.ubc.ca
# @Last modified by:   aaronmishkin
# @Last modified time: 18-09-13

from functools import partial

import lib.metrics.metrics as metrics

BAYES_REG = ['pred_avneg_loglik_gaussian', 'avneg_elbo_gaussian', 'predictive_rmse']
BAYES_MULTICLASS = ['pred_avneg_loglik_categorical', 'avneg_elbo_categorical', 'softmax_predictive_accuracy']
BAYES_BINCLASS = ['pred_avneg_loglik_bernoulli', 'avneg_elbo_bernoulli', 'sigmoid_predictive_accuracy']

BAYES_REG = ['pred_avneg_loglik_gaussian', 'avneg_elbo_gaussian', 'predictive_rmse']
BAYES_MULTICLASS = ['pred_avneg_loglik_categorical', 'avneg_elbo_categorical', 'softmax_predictive_accuracy']
BAYES_BINCLASS = ['pred_avneg_loglik_bernoulli', 'avneg_elbo_bernoulli', 'sigmoid_predictive_accuracy']


def make_metric_closures(metric_names, kl_fn, train_set_size, tau=None):
    train_metrics = {}
    test_metrics = {}

    for name in metric_names:
        if tau is None and (name == 'pred_avneg_loglik_gaussian' or name == 'avneg_elbo_gaussian' or name == 'predictive_rmse'):
            raise ValueError('The noise precision must be defined for regression metrics.')

        if name == 'pred_avneg_loglik_gaussian':
            train_metrics['train_pred_logloss'] = partial(metrics.predictive_avneg_loglik_gaussian, tau=tau)
            test_metrics['test_pred_logloss'] = partial(metrics.predictive_avneg_loglik_gaussian, tau=tau)

        elif name == 'avneg_loglik_gaussian':
            train_metrics['train_pred_logloss'] = partial(metrics.mc_loss, loss_fn=metrics.avneg_loglik_gaussian, tau=tau)
            test_metrics['test_pred_logloss'] = partial(metrics.mc_loss, loss_fn=metrics.avneg_loglik_gaussian, tau=tau)

        elif name == 'avneg_elbo_gaussian':
            def elbo_closure(mu_list, y):
                return metrics.avneg_elbo_gaussian(mu_list, y, tau=tau, train_set_size=train_set_size, kl=kl_fn())

            train_metrics['elbo_neg_ave'] = elbo_closure

        elif name == 'predictive_rmse':
            train_metrics['train_pred_rmse'] = metrics.predictive_rmse
            test_metrics['test_pred_rmse'] = metrics.predictive_rmse

        elif name == 'pred_avneg_loglik_categorical':
            train_metrics['train_pred_logloss'] = metrics.predictive_avneg_loglik_categorical
            test_metrics['test_pred_logloss'] = metrics.predictive_avneg_loglik_categorical

        elif name == 'avneg_loglik_categorical':
            train_metrics['train_pred_logloss'] = partial(metrics.mc_loss, loss_fn=metrics.avneg_loglik_categorical)
            test_metrics['test_pred_logloss'] = partial(metrics.mc_loss, loss_fn=metrics.avneg_loglik_categorical)

        elif name == 'avneg_elbo_categorical':
            def elbo_closure(logits_list, y):
                return metrics.avneg_elbo_categorical(logits_list, y, train_set_size=train_set_size, kl=kl_fn())
            train_metrics['elbo_neg_ave'] = elbo_closure

        elif name == 'softmax_predictive_accuracy':
            train_metrics['train_pred_accuracy'] = metrics.softmax_predictive_accuracy
            test_metrics['test_pred_accuracy'] = metrics.softmax_predictive_accuracy

        elif name == 'pred_avneg_loglik_bernoulli':
            train_metrics['train_pred_logloss'] = metrics.predictive_avneg_loglik_bernoulli
            test_metrics['test_pred_logloss'] = metrics.predictive_avneg_loglik_bernoulli

        elif name == 'avneg_loglik_bernoulli':
            train_metrics['train_pred_logloss'] = partial(metrics.mc_loss, loss_fn=metrics.avneg_loglik_bernoulli)
            test_metrics['test_pred_logloss'] = partial(metrics.mc_loss, loss_fn=metrics.avneg_loglik_bernoulli)

        elif name == 'avneg_elbo_bernoulli':
            def elbo_closure(logits_list, y):
                return metrics.avneg_elbo_bernoulli(logits_list, y, train_set_size=train_set_size, kl=kl_fn())
            train_metrics['elbo_neg_ave'] = elbo_closure

        elif name == 'sigmoid_predictive_accuracy':
            train_metrics['train_pred_accuracy'] = metrics.sigmoid_predictive_accuracy
            test_metrics['test_pred_accuracy'] = metrics.sigmoid_predictive_accuracy

    return train_metrics, test_metrics

def make_objective_closure(objective_name, kl_fn, train_set_size=None, tau=None):
    train_metrics, _ = make_metric_closures([objective_name], kl_fn, train_set_size=train_set_size, tau=tau)
    objective = list(train_metrics.values())[0]
    return objective
