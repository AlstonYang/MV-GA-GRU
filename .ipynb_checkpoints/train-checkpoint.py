#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, Flatten
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Metric
import matplotlib.pyplot as plt


# In[2]:

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(sess)


# In[3]:


np.set_printoptions(suppress=True)


# In[4]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))


# In[5]:


def read(path):
    return pd.read_csv(path)


# In[6]:


def buildTrain(train, pastWeek, futureWeek=1, defaultWeek=1):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureWeek-pastWeek):
        
        X = np.array(train.iloc[i:i+pastWeek,:])
        
        X_train.append(X.reshape(X.size))
        Y_train.append(np.array(train.iloc[i+pastWeek:i+pastWeek+futureWeek]["CCSP"]))
    return (np.array(X_train), np.array(Y_train))


# In[7]:


def get_data(timeLag):
    
    ## Read weekly copper price data
    path = "WeeklyFinalData.csv"
    data = read(path)
    
    date = data["Date"]
    data.drop("Date", axis=1, inplace=True)
    
    ## Add time lag (pastWeek=4, futureWeek=1)
    x_data, y_data = buildTrain(data, timeLag)
    
    ## Data split
    x_train = x_data[0:int(x_data.shape[0]*0.8)]
    x_test = x_data[int(x_data.shape[0]*0.8):]
    
    y_train = y_data[0:int(y_data.shape[0]*0.8)]
    y_test = y_data[int(y_data.shape[0]*0.8):]
    
    ## Other information
    nb_output = 1
    
    return (nb_output, x_train, x_test, y_train, y_test)


# In[18]:


# path = "WeeklyFinalData.csv"
# data = read(path)

# date = data["Date"]
# data.drop("Date", axis=1, inplace=True)

# ## Add time lag (pastWeek=4, futureWeek=1)
# x_data, y_data = buildTrain(data, 30)
# print(x_data.shape)
# # print(x_data.reshape(-1,3,15)[0])
# # print(y_data[0])
# # print(x_data.shape, y_data.shape)


# In[9]:


def compile_model(nb_neurons, nb_layers, optimizer, nb_output, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    keras.backend.clear_session()
    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(GRU(units = nb_neurons, batch_input_shape=input_shape, return_sequences=True))
        if i==(nb_layers-1):
            model.add(GRU(units = nb_neurons, batch_input_shape=input_shape))
        else:
            model.add(GRU(units = nb_neurons, return_sequences=True))

#         model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
#     model.add(Flatten())
    model.add(Dense(units = nb_output))

#     print(model.summary())
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    
    return model


# In[10]:


def train_and_score(Network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """

    ## The moving window mechanism for incremental learning (# of batch)
    batch_size = Network.network['batch_size']
    
    ## The time lag as forecasting variable (# of sequence)
    window_size = Network.network['window_size']
    
    ## The hyperparameter for GRU
    nb_neurons = Network.network['nb_neurons']
    nb_layers = Network.network['nb_layers']
    optimizer = Network.network['optimizer']

    ## Get the training data from help method: get_data(window_size)
    nb_output, x_train, x_test, y_train, y_test = get_data(window_size)
    
    ## The number of forecasting variable (# of variable)
    nb_input_factor = 15
    
    ## Data transformation
    x_train_scaled = sc.fit_transform(x_train).reshape(-1,window_size,nb_input_factor)
    x_test_scaled = sc.transform(x_test).reshape(-1,window_size,nb_input_factor)
    y_train_scaled = sc.fit_transform(y_train)
    
    ## Define the GRU input_shape and compile the model
    input_shape = (None, window_size, nb_input_factor)
    model = compile_model(nb_neurons, nb_layers, optimizer, nb_output, input_shape)
    
    ## The volume of training data
    nb_data = x_train_scaled.shape[0]
    
    ## The performance_indicator list to store loss and accuracy
    performance_indicator = pd.DataFrame(columns=["Loss_train","Accuracy_train","Loss_test","Accuracy_test"])
    
    ## The times of training
    epoch = 300
    
    ## The training step of GRU
    for e in range(epoch):
        
        minimum_loss = np.inf
        current_times = 0

        if (current_times > 5):

            model.load_weights("model_weight.h5")
            Network.set_model(model)
           # Network.set_weights(model.get_weights())
            break

        else:

            for i in range(0, nb_data-batch_size+1):

                end = i + batch_size

                if end < nb_data:
                    x = x_train_scaled[i:end]
                    y = y_train_scaled[i:end]
#                     print(x.shape,y.shape)
                    model.train_on_batch(x, y)

                else:
                    x = x_train_scaled[i:nb_data]
                    y = y_train_scaled[i:nb_data]
#                     print(i)
#                     print(x.shape, y.shape)
                    model.train_on_batch(x, y)

            ## To calculate the forecasting performance
            y_pred_train = sc.inverse_transform(model.predict(x_train_scaled))
            loss_train = tf.reduce_mean(tf.square(y_train - y_pred_train)).numpy()
            accuracy_train = tf.reduce_sum(tf.cast(tf.less_equal(tf.abs(y_train - y_pred_train), 1000), dtype = tf.float32)).numpy()/y_train.shape[0]

            y_pred_test = sc.inverse_transform(model.predict(x_test_scaled))
            loss_test = tf.reduce_mean(tf.square(y_test - y_pred_test)).numpy()
            accuracy_test = tf.reduce_sum(tf.cast(tf.less_equal(tf.abs(y_test - y_pred_test), 1000), dtype = tf.float32)).numpy()/y_test.shape[0]

            epoch_performance = pd.DataFrame({
                "Loss_train":loss_train,
                "Accuracy_train":accuracy_train,
                "Loss_test":loss_test,
                "Accuracy_test":accuracy_test
            },index=[0])

            performance_indicator = performance_indicator.append(epoch_performance,ignore_index=True)
            print("Epoch: %d, Loss: %.2f, Accuracy_train: %.2f%%, Accuracy_test: %.2f%%"%(e,loss_train,accuracy_train*100,accuracy_test*100))
            print("-"*50)


            if(loss_train <= minimum_loss):
                minimum_loss = loss_train
                current_times=0
                model.save_weights("model_weight.h5")
                Network.set_model(model)

            else:
                current_times += 1
                if (e>=epoch-1):
                    Network.set_model(model)
    
    Network.performance_indicator = performance_indicator
    
    
    y_pred_test = sc.inverse_transform(model.predict(x_test_scaled))
    loss_test = tf.reduce_mean(tf.square(y_test - y_pred_test)).numpy()
    accuracy_test = tf.reduce_sum(tf.cast(tf.less_equal(tf.abs(y_test - y_pred_test), 1000), dtype = tf.float32)).numpy()/y_test.shape[0]

    Network.y_true = y_test
    Network.y_predict = y_pred_test
    
    return accuracy_test  


# In[11]:


# """Class that represents the network to be evolved."""
# import random
# import logging


# In[12]:


# class Network():
#     """Represent a network and let us operate on it.

#     Currently only works for an MLP.
#     """

#     def __init__(self, nn_param_choices=None):
#         """Initialize our network.

#         Args:
#             nn_param_choices (dict): Parameters for the network, includes:
#                 'window_size':[i for i in range(1,50)]
#                 'nb_neurons': [i for i in range(3, 41, 1)],
#                 'nb_layers': [i for i in range(1,11)],
#                 'batch_size':[i for i in range(1,21)],
#                 'epoch':[i for i in range(10,501)],
#                 'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
#                                   'adadelta', 'adamax', 'nadam','ftrl'],
#         """
#         self.accuracy = 0.
#         self.nn_param_choices = nn_param_choices
#         self.network = {}  # (dic): represents MLP network parameters
#         self.model = None
#         self.performance_indicator = None
#         self.y_true = None
#         self.y_predict = None
        
#     def create_random(self):
#         """Create a random network."""
#         for key in self.nn_param_choices:
#             self.network[key] = random.choice(self.nn_param_choices[key])

#     def create_set(self, network):
#         """Set network properties.

#         Args:
#             network (dict): The network parameters

#         """
#         self.network = network

#     def train(self):
#         """Train the network and record the accuracy.

#         Args:
#             dataset (str): Name of dataset to use.

#         """
#         if self.accuracy == 0.:
#             self.accuracy = train_and_score(self)

#     def set_model(self, trained_model):
#         self.model = trained_model
    
#     def plot_learning_graph (self):
    
#         fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), dpi=80, facecolor="w", edgecolor="k")

#         data = self.performance_indicator
#         for i in range(0,2):
#             ax = axes[i]

#             train = data.iloc[:,i]
#             test = data.iloc[:,i+2]

#             ax.plot(train, label="Train")
#             ax.plot(test, label="Test")
#             ax.legend()
#             ax.set_xlabel("epoch")

#             if i==0:
#                 ax.set_title("Loss")
#                 ax.set_ylabel("Loss")
#             else:
#                 ax.set_title("Accuracy")
#                 ax.set_ylabel("Accuracy")

#             ax.ticklabel_format(style='plain', useOffset=False, axis='both')

#         plt.tight_layout()
        
        
#     def plot_prediction_value(self):
    
#         fig, axes = plt.subplots(figsize=(15, 5), dpi=80, facecolor="w", edgecolor="k")

#         axes.plot(self.y_true, label="Actual")
#         axes.plot(self.y_predict, label="Prediction")
#         axes.legend()
#         axes.set_xlabel("Sample index(Weekly)")
#         axes.set_ylabel("Changjiang Copper Spot Price $CNY/ton")
#         axes.set_title("Actual data and predicted data comparison")
#         axes.ticklabel_format(style='plain', useOffset=False, axis='both')

#         plt.tight_layout()
    
#     def print_network(self):
#         """Print out a network."""
#         logging.info(self.network)
#         logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))


# In[13]:


# nn_param_choices = {
    
#     'window_size':[20],
#     'batch_size':[i for i in range(4,9)],
#     'nb_neurons': [i for i in range(3, 41, 1)],
#     'nb_layers': [i for i in range(1,11)],
#     'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
#                   'adadelta', 'adamax', 'nadam','ftrl']
# }

# network = Network(nn_param_choices)
# network.create_random()


# In[14]:


# network.network


# In[15]:


# network.train()


# In[16]:


# network.plot_learning_graph()


# In[17]:


# network.plot_prediction_value()

