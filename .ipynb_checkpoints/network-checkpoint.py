#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score
import matplotlib.pyplot as plt

# In[2]:


class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                'window_size':[i for i in range(1,50)]
                'nb_neurons': [i for i in range(3, 41, 1)],
                'nb_layers': [i for i in range(1,11)],
                'batch_size':[i for i in range(1,21)],
                'epoch':[i for i in range(10,501)],
                'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                                  'adadelta', 'adamax', 'nadam','ftrl'],
        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters
        self.model = None
        self.performance_indicator = None
        self.y_true = None
        self.y_predict = None
        
    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self):
        """Train the network and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.accuracy == 0.:
            self.accuracy = train_and_score(self)

    def set_model(self, trained_model):
        self.model = trained_model
    
    def plot_learning_graph (self):
    
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), dpi=80, facecolor="w", edgecolor="k")

        data = self.performance_indicator
        for i in range(0,2):
            ax = axes[i]

            train = data.iloc[:,i]
            test = data.iloc[:,i+2]

            ax.plot(train, label="Train")
            ax.plot(test, label="Test")
            ax.legend()
            ax.set_xlabel("epoch")

            if i==0:
                ax.set_title("Loss")
                ax.set_ylabel("Loss")
            else:
                ax.set_title("Accuracy")
                ax.set_ylabel("Accuracy")

            ax.ticklabel_format(style='plain', useOffset=False, axis='both')

        plt.tight_layout()
        
        
    def plot_prediction_value(self):
    
        fig, axes = plt.subplots(figsize=(15, 5), dpi=80, facecolor="w", edgecolor="k")

        axes.plot(self.y_true, label="Actual")
        axes.plot(self.y_predict, label="Prediction")
        axes.legend()
        axes.set_xlabel("Sample index(Weekly)")
        axes.set_ylabel("Changjiang Copper Spot Price $CNY/ton")
        axes.set_title("Actual data and predicted data comparison")
        axes.ticklabel_format(style='plain', useOffset=False, axis='both')

        plt.tight_layout()
    
    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))

