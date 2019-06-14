# Libraries
import torch
import torch.nn.functional as F
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Our Code
from lib.optimizers.slang import SLANG
import lib.optimizers.closure_factories as closure_factories
import lib.metrics.metric_factory as metric_factory
from lib.experiments.printers import print_progress, print_objective
from lib.experiments.evaluate_model import evaluate_model
from lib.utilities.general_utilities import construct_prior_vector, set_seeds

from lib.data.classification.synthetic_classification import generate_data
from torchutils.models import MLP as MultiSampleMLP

if False:

    #####################################
    ### Define the Prior Distribution ###
    #####################################

    # The prior for weights is N(0, I/prior_prec)
    prior_prec = 0.01
    # The prior for biases is chosen to be vague.
    bias_prior_prec = 1e-6

    ##################################
    ### Generate Synthetic Dataset ###
    ##################################

    X, y = generate_data()
    y = y.to(torch.long)

    N,D = X.size()

    #################################
    ### SLANG Core Initialization ###
    #################################

    output_size = 1
    act_func = F.relu

    # Initialize the a logistic regression model with parallelized sampling.
    model = MultiSampleMLP(input_size=D,
                           hidden_sizes=[10],
                           output_size=output_size,
                           act_func=act_func)

    if torch.cuda.is_available():
        model = model.cuda()

    # Create a PyTorch-style closure that returns separate minibatch gradients.
    closure_factory = closure_factories.individual_gradients_closure_factory

    # Construct a
    prior_vector = construct_prior_vector(weight_prior=prior_prec,
                                          bias_prior=bias_prior_prec,
                                          named_parameters=model.named_parameters())


    # don't decay the parameters:
    decay_params = { 'learning_rate': 0, 'beta': 0 }

    # optimizer parameters
    num_epochs = 1000
    num_mc_samples = 20
    alpha = 0.01
    beta = 0.05
    momentum = 0.9
    initial_diagonal = 1
    # Number of low-rank vectors
    L = 1

    optimizer = SLANG(model.parameters(),
                      lr=alpha,
                      betas=(momentum, 1-beta),
                      prior_prec=prior_vector,
                      s_init=initial_diagonal,
                      decay_params=decay_params,
                      num_samples=num_mc_samples,
                      L=L,
                      train_set_size=N)

    def predict_fn(x, mc_samples):
        noise = optimizer.distribution().rsample(mc_samples).t()
        preds = model(x, noise, False)
        if output_size == 1:
            preds = preds.reshape(mc_samples, -1)
        return preds

    def kl_fn():
        return optimizer.kl_divergence()


    ##################
    ### Experiment ###
    ##################

    set_seeds(123)

    # Set objective and metrics:
    objective = 'avneg_loglik_bernoulli'
    metrics = ['avneg_elbo_bernoulli', 'pred_avneg_loglik_bernoulli', 'pred_avneg_loglik_bernoulli', 'sigmoid_predictive_accuracy']
    normalize = {'x': False, 'y': False}

    objective = metric_factory.make_objective_closure("avneg_elbo_bernoulli", kl_fn, N)
    metrics = metric_factory.BAYES_BINCLASS
    train_metrics, test_metrics = metric_factory.make_metric_closures(metrics, kl_fn, N)

    #####################################
    ########### Training Loop ###########
    #####################################

    metric_history = {}
    for name in train_metrics.keys():
        metric_history[name] = []
    for name in test_metrics.keys():
        metric_history[name] = []


    for epoch in range(num_epochs):
        # Set model in training mode
        model.train(True)

        closure = closure_factory(X, y, objective, model, predict_fn, optimizer, num_mc_samples)
        loss = optimizer.step(closure)

        model.train(False)
        # Evaluate model
        with torch.no_grad():
            metric_history = evaluate_model(predict_fn, train_metrics, test_metrics, metric_history, X, y, X, y, num_mc_samples, {'x':False, 'y':False})
        # Print progress
        print_progress(epoch, num_epochs, metric_history)

    # Set model in test mode
    model.train(False)
