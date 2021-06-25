# nPlan Machine Learning Research Scientist Assessment
## Amy Parkes

I have created two models to train on the PPI dataset.

A 'naive' feed-forward neural network (models/naive_approach.py) which uses node features to predict node targets, ignoring all graph-related information. This uses the Keras (https://keras.io/) functional API.

A graph convolutional neural (models/GNN.py) which uses a Spektral (https://graphneural.network/) model that implements the architecture from the paper "Semi-Supervised Classification with Graph Convolutional Networks" by Thomas N. Kipf and Max Welling (https://arxiv.org/abs/1609.02907).

## Usage

A requirements file is provided, this was built using python 3.8.5. Install required packages to a virtual environment with the follow commands for linux.

```bash
virtualenv -p <path/to/python3.8.5> <myenv>
source <myenv>/bin/activate
python -m pip install -r requirements.txt
```

To run and test the pre-trained feed-forward network and then 'dummy' train (i.e. only one epoch of training) and test the graph network use run_model.py

```bash
python run_models.py
```

Model testing results are logged to outputs.log

## Observations

I have written thoughts/justifications throughout the code commenting.

Given more time, I would have further investigated the following:

-- I have noticed that for all numeric node features within a graph, there are only two distinct values. I would assume this means the graphs could be condensed in some way to improve training.

-- I have only implemented a very basic character summary for including the text as a feature for training. I would imagine there is a more elegant way to incorporate sentence embeddings into features.

-- In most cases I have used arbitrary network parameters. I have not tuned parameters in any way as good predictive accuracy is not expected, given a dataset where I could expect meaningful predictions I would consider using a genetic algorithm for this if it was computationally feasible.

-- As the meaning of the targets is not clear, it is hard to choose a loss function and performance metrics.
