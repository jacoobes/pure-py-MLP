import math
from abc import ABC, abstractmethod
from typing import List , Tuple
import numpy as np

def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    pass

class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, x: np.ndarray):
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return (1 / (1 + np.exp(-x)))

    def derivative(self, x):
        gx = self.forward(x)
        return gx * (1 - gx)
        

class Tanh(ActivationFunction):
    def forward(self, x):
        return math.tanh(x)

    def derivative(self, x):
        gx = self.forward(x)
        return 1 - np.square(gx) 


class Relu(ActivationFunction):
    def forward(self, x ):
        return np.maximum(0, x)

    def derivative(self, x:  np.ndarray):
        if(x < 0):
            return 0
        else:
            return 1

class Softmax(ActivationFunction):
    def forward(self, x):
        exp_logits = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability improvement
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def derivative(self, x):
        pass


class Linear(ActivationFunction):
    def forward(self, x):
        return x

    def derivative(self, x):
        return 1


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true, y_pred) -> np.floating:
        ...
       
    @abstractmethod
    def derivative(self, y_true, y_pred):
        ...


class SquaredError(LossFunction):
    def loss(self, y_true, y_pred):
        diffs = [((y - yhat) ** 2) for y, yhat in zip(y_true, y_pred)]
        return np.mean(diffs)
        
    def derivative(self, y_true, y_pred):
        pass

class CrossEntropy(LossFunction):
    def loss(self, y_true, y_pred) -> np.floating:
        return -1 * np.sum([y * math.log(yhat) for y, yhat  in zip(y_true, y_pred)])
        

    def derivative(self, y_true, y_pred):
        pass



class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function

        # this will store the activations (forward prop)
        self.activations = None
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = None

        # Initialize weights and biaes
        self.W = None  # weights
        self.b = None  # biases

    def forward(self, h: np.ndarray):
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """
        self.activations = None

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        dL_dW = None
        dL_db = None
        self.delta = None
        return dL_dW, dL_db


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """

        return None

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all = []
        dl_db_all = []


        return None, None

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """
        training_losses = None
        validation_losses = None

        return training_losses, validation_losses
