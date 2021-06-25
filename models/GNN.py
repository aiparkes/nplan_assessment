"""
Convolutional Graph Neural Network Implementation
-------------------------------------------------

:Author: Amy Parkes
:Date: 25/06/2021

Class Dataset holds graph information, formatted for training GCN.

testing and evaluate perform network testing on each graph in the testing set in turn.

graph_net acts as a main, creating, training and testing a network.
"""

import tensorflow as tf
import numpy as np
from typing import List, Dict, Tuple, Union
import pandas as pd
import logging
logging.basicConfig(handlers=[logging.FileHandler('outputs.log', 'w', 'utf-8')], level=logging.DEBUG)

import spektral
from spektral.data import Dataset, Graph, DisjointLoader
from spektral.models.gcn import GCN

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_addons as tfa

from utils.text_to_feat import text_to_feats


class Dataset(Dataset):
    """
    Spektral class for storing graph representaitons.
    Needed to make a data Loader to feed graphs into a GNN
    """

    def __init__(self, data: Dict[str, Union[np.array, Dict]], set: str):
        self.data = data
        self.set = set
        self.feats = data[set + "_feats"]
        self.labels = data[set + "_labels"]
        self.graph_id = data[set + "_graph_id"]
        self.graph = data[set + "_graphs"]
        self.links = pd.DataFrame(self.graph["links"])
        self.text = pd.DataFrame(self.graph["nodes"])
        self.text = self.text.rename({1: "text"})
        super().__init__()

    def make_adj_matrix(self, edges: pd.DataFrame) -> np.array:
        """
        Converts df of edges to adjacency matrix.

        :param edges: df of edges with 'source' and 'taget' columns.
        :return: adjacency matrix where mat[i,j]==1 indicates directed edge from i to j.

        I am unsure about taking the identity away from the adj matrix, but the
        json provided specifies that the graphs are NOT multigraphs, so there
        should not be edges from a node to itself.
        """
        edges = edges - min(edges.source)
        mat = np.zeros((self.num_nodes, self.num_nodes))
        mat[edges.source.values, edges.target.values] = 1
        mat = mat - np.eye(self.num_nodes)
        return mat

    def make_feature_matrix(self, inds: np.array) -> np.array:
        """
        Combines features derived from text in self.text with numeric features self.feats.

        :param inds: Indices of nodes relevant to the current graph.
        :return: Updated feature matrix with text 'embeddings'.
        """
        sum, mean, var = text_to_feats(self.text.iloc[inds])
        x = np.concatenate((self.feats[inds], sum, mean, var), axis=1)
        return x

    def extract_graph_attrib(self, graph_num: int) -> Tuple[np.array]:
        """
        Converts np and json data formats to graph attributes for a single graph.

        :param graph_num: The graph ID.
        :return: features, adj matrix and targets of graph
        """
        inds = np.where(self.graph_id == graph_num)
        self.num_nodes = len(inds[0])
        x = self.make_feature_matrix(inds)
        mat = self.make_adj_matrix(self.links[self.links.source.isin(inds[0])])
        labels = self.labels[inds]
        return x, mat, labels

    def read(self) -> List[spektral.data.Graph]:
        """
        Required for compatability with Spektral class.

        :return: A list of Graph objects.
        """
        def make_graph(graph_num: int) -> Tuple[np.array]:
            """
            Converts np and json data formats to graph attributes for a single graph.

            :param graph_num: The graph ID.
            :return: features, adj matrix and targets of graph
            """
            x, mat, labels = self.extract_graph_attrib(graph_num)
            df = pd.DataFrame(labels, columns=None)
            y = df.sum().values
            return Graph(x=x, a=mat, y=y)

        return [make_graph(graph_num) for graph_num in set(self.graph_id)]


def evaluate(model: spektral.models.gcn, graph_num: int, metric: tfa.metrics, test_data: spektral.data.Dataset) -> Tuple[np.float, int]:
    """
    Predict for single graph and calculate performance metric value.

    :param model: Trained conv graph network for testing.
    :param graph_num: The graph ID.
    :param metric: Performance measure to asses performance with.
    :param test_data: Data for testing.
    :return: metric value and number of nodes in graph
    """
    test_x, test_mat, test_y = test_data.extract_graph_attrib(graph_num)
    prediction = model((test_x, test_mat), training=False)
    metric.update_state(test_y, prediction)
    return metric.result().numpy(), len(test_y)


def testing(model: spektral.models.gcn, data: Dict[str, Union[np.array, Dict]], perf_metric: tfa.metrics) -> np.float:
    """
    Test on all graphs in test set, weight results based on number of nodes in
    each graph.

    :param model: Trained conv graph network for testing.
    :param data: Data provided.
    :param metric: Performance measure to asses performance with.
    :return: Average performance metric value across all testing graphs.

    This weighting system may not be appropriate, but more problem information
    is required to investigate this further.
    """
    metric = perf_metric(mode="multilabel", threshold=0.8)
    results = []
    test_data = Dataset(data, "test")
    for graph_num in set(data["test_graph_id"]):
        results.append(evaluate(model, graph_num, metric, test_data))
    results = np.array(results)
    return np.average(results[:, :-1], 0, weights=results[:, -1])


def graph_net(data: Dict[str, Union[np.array, Dict]], params: str):
    """
    Creates, trains and tests a convolutional graph network on provided data.

    :param data: Dictionary of train/test/valid data.
    :param params: Dictionary of parameters for network and testing.

    I am not sure if graphSAGE layers might be better for this task, but more
    problem information is probably needed to answer this.
    """
    logging.info("Using graph network approach.")

    # create spektral datasets
    train_data = Dataset(data, "train")
    valid_data = Dataset(data, "valid")

    # creates spektral data loaders to feed graphs into gnn
    loader_train = DisjointLoader(train_data, batch_size=params["GNN_batch_size"])
    loader_valid = DisjointLoader(valid_data, batch_size=params["GNN_batch_size"])

    # create basic graph conv nn
    model = GCN(
        n_labels=train_data.n_labels, n_input_channels=train_data.n_node_features
    )
    model.compile(optimizer=params["optimiser"], loss=params["loss"])

    # train model on each graph in training set in turn
    model.fit(
        loader_train.load(),
        steps_per_epoch=loader_train.steps_per_epoch,
        validation_data=loader_valid.load(),
        validation_steps=loader_valid.steps_per_epoch,
        epochs=params["epochs"],
        callbacks=[
            EarlyStopping(patience=params["patience"], restore_best_weights=True)
        ],
    )

    # test trained model
    test_results = testing(model, data, params["perf_metric"])

    logging.info("Done.\n" "Test Average Hamming Distance per Graph: {}".format(*test_results))
