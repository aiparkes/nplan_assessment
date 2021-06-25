"""
nPlan Machine Learning Research Scientist Assessment Entry Point
----------------------------------------------------------------

:Author: Amy Parkes
:Date: 25/06/2021

Parameters are used in both models.

All bar the loss and performance metrics
are arbitrary. This is because 'good' prediction accuracy is not to be expected
on the given dataset. Given a dataset with predictive abilities I would tune
these using evolutionary computation if it was computationally feasible.

Epochs are set very low for computational ease, since no predictive abilities
are expected from the data. Ordinarily I would try to set the number of epochs
such that early stopping terminates training.

I am not sure these are necessarily the most appropriate error measures. The
targets look like it is a multi-class, multi-label problem, but it is not clear
from the set-up if the class probabilities are independent. Given more
time/problem information I would look into others including categorical cross
entropy and binary cross entropy.

For the feed-forward network, if argument use_pre_trained=True a pre-trained model
(feedforward_weights.h5) is loaded and tested. If use_pre_trained=False a new
model with the same parameters is created, trained and tested.

For the graph neural network a new model is created, trained and tested every time.
This is because I ran out of time to attach a trained one as well.
"""

import tensorflow_addons as tfa

from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.regularizers import l1_l2

from models.naive_approach import ff_neural_net
from models.GNN import graph_net
from utils.load_data import load_data

params = {
    "epochs": 1,
    "patience": 10, # Patience for early stopping
    "GNN_batch_size": 1, # Batch for GNN
    "NN_batch_size": 50, # Batch for NN -larger because ignoring graph architecture
    "initialiser": TruncatedNormal(mean=0.0, stddev=1.0),
    "optimiser": "Adamax",
    "activation": "relu",
    "layers": 4,
    "neurons": 500,
    "regulariser": l1_l2(0.001, 0.001),
    "perf_metric": tfa.metrics.HammingLoss,
    "loss": "mean_absolute_error",
}

# load data provided
data = load_data()

# load pre-trained model and test feedforward model
ff_neural_net(data, params, use_pre_trained=True)

# make, train and test graph neural network model
graph_net(data, params)
