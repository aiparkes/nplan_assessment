"""
Feed-forward Neural Network Implementation
------------------------------------------

:Author: Amy Parkes
:Date: 25/06/2021

Class PlainNetwork makes feed-forward networks using the keras framework with
the same number of neurons in each layer.

convert_text is provided for embedding text features into numeric featues.

flatten converts configs from keras layers to dict[str, str] for saving specifications

ff_neural_net acts as a main, creating, training and testing a network.
"""

import collections
from collections import OrderedDict
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple, Union
import logging
logging.basicConfig(handlers=[logging.FileHandler('outputs.log', 'w', 'utf-8')], level=logging.DEBUG)

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, concatenate, Dropout, experimental
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_addons as tfa

from utils.text_to_feat import text_to_feats


class PlainNetwork(object):
    """
    Creates feed-forward neural networks using the keras functional API framework.
    Each network has the same number of neurons in every hidden layer.

    Produces specification dict with all network architecture params.
    """

    def __init__(self, params: Dict[str, str]):
        self.specs = OrderedDict()
        self.activation = params["activation"]
        self.initialiser = params["initialiser"]
        self.regulariser = params["regulariser"]
        self.neurons = params["neurons"]
        self.num_layers = params["layers"]
        self.optimiser = params["optimiser"]
        self.loss = params["loss"]
        self.patience = params["patience"]
        super().__init__()

    def dense_layer(self, name: str, neurons: int, activation: str, trainable=True) -> tensorflow.keras.layers:
        """
        Creates a dense feed-forward neural network layer, and updates specs with layer specifications.

        :return: keras dense layer with the given parameters
        """
        layer = Dense(
            neurons,
            activation=activation,
            use_bias=True,
            kernel_initializer=self.initialiser,
            bias_initializer=self.initialiser,
            kernel_regularizer=self.regulariser,
            bias_regularizer=self.regulariser,
            activity_regularizer=self.regulariser,
            kernel_constraint=max_norm(100),
            bias_constraint=max_norm(100),
            name=name,
            trainable=trainable,
        )
        self.specs.update(flatten(layer.get_config()))
        return layer

    def add_layers(self, input: tensorflow.keras.layers) -> tensorflow.keras.layers:
        """
        Stacks hidden dense layers sequentially onto input layer.

        :param input: The input layer of the neural network.
        :return: The last layer in the chain of layers created.
        """
        x = self.dense_layer("layer1", self.neurons, self.activation)(input)
        for layer in np.arange(1, self.num_layers):
            x = self.dense_layer(
                "layer" + str(layer + 1), self.neurons, self.activation
            )(x)
        return x

    def create_neural_net(self, input_shape: Tuple[int], output_shape: Tuple[int]) -> tensorflow.keras.Model:
        """
        Creates feed-forward neural network using keras functional API, with
        given input and output data shapes. Uses linear activation in the final
        (output) layer. Complies and updates specs with network details.

        :return: compiled Keras neural network model.
        """
        input = Input(shape=(input_shape,), name="input")
        x = self.add_layers(input)
        output = self.dense_layer("output", output_shape, "linear", trainable=False)(x)

        model = Model(inputs=[input], outputs=[output])
        model.compile(optimizer=self.optimiser, loss=self.loss)
        self.specs.update({"optimizer": self.optimiser, "loss": self.loss})
        return model

    def early_stop(self) -> tensorflow.keras.callbacks:
        """
        Creates early stopping procedure for neural network and updates specs.

        :return: keras callback early stopping.
        """
        self.specs.update({"patience": self.patience, "restore_best_weights": True})
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            verbose=1,  # 1 gives progress bar, 0 prints nothing to console, # 2 tells you which epoch you are on
            mode="auto",
            restore_best_weights=True,
        )
        return early_stopping


def flatten(d: tensorflow.keras.layers, parent_key="", sep="_") -> Dict[str,str]:
    """
    Converts config of keras object into dictionary items for printing.

    :param d: config of keras layer.
    :return: Dictionary of parameters.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def convert_text(data: Dict[str, Union[np.array, Dict]]) -> Dict[str, Union[np.array, Dict]]:
    """
    Converts dictionary of strings (data["***_graphs"]["nodes"]) to np.arrays.
    Appends to data["***_feats"] to increase number of features per node.

    :param data: dictionary holding train/test/valid data.
    :return: dictionary holding train/test/valid data.
    """
    for set in ["train", "valid", "test"]:
        text = pd.DataFrame(data[set + "_graphs"]["nodes"])
        text = text.rename({1: "text"})
        sum, mean, var = text_to_feats(text)
        data[set + "_feats"] = np.concatenate(
            (data[set + "_feats"], sum, mean, var), axis=1
        )
    return data


def testing(model: tensorflow.keras.Model, data: Dict[str, Union[np.array, Dict]], metric: tfa.metrics) -> np.float:
    """
    Tests trained model on testing set. Tests nodes irrespective of graph.

    :param model: Trained model.
    :param data: Dictionary of train/test/valid data.
    :param metric: Performance measuse to assess predictions with.
    :return: Error in prediction.
    """
    prediction = model.predict(data["test_feats"], batch_size=1)
    metric.update_state(data["test_labels"], prediction)
    return metric.result().numpy()


def ff_neural_net(data: Dict[str, Union[np.array, Dict]], params: Dict[str, str], use_pre_trained=True):
    """
    Creates, trains and tests a feed-forward network on provided data.

    :param data: Dictionary of train/test/valid data.
    :param params: Dictionary of parameters for network and testing.
    :params use_pre_trained: Bool denoting if a pre-trained network should be loader
    or a new model built and trained.
    """
    logging.info("Using feedforward network approach.")

    #create network
    plain_nets = PlainNetwork(params)

    # add character summary as numeric representaiton of text for input feature
    data = convert_text(data)
    input_shape = len(data["train_feats"][0])
    output_shape = len(data["train_labels"][0])

    # create feedforward neural network model
    model = plain_nets.create_neural_net(input_shape, output_shape)

    if use_pre_trained:
        logging.info("Using pre-trained model provided.")
        weight_file = os.path.join("utils", "feedforward_weights.h5")
        model.load_weights(weight_file)
    else:
        logging.info("Training model from scratch.")
        early_stopping = plain_nets.early_stop()

        # train nn on all nodes features, irrespective of which graph they belong to
        # this approach is clearly missing all the graph-related information
        training_history = model.fit(
            x=data["train_feats"],
            y=data["train_labels"],
            validation_data=(data["valid_feats"], data["valid_labels"]),
            epochs=params["epochs"],
            batch_size=params["NN_batch_size"],
            shuffle=True,  # Shuffle data at each epoch
            callbacks=[early_stopping],
            verbose=1,
        )

    # define performance metric for testing
    metric = params["perf_metric"](mode="multilabel", threshold=0.8)

    # test trained model
    test_results = testing(model, data, metric)

    logging.info("Done.\n" "Test Hamming Distance for All Nodes: {}".format(test_results))
