import math
from abc import ABC, abstractmethod
from typing import Generator, Tuple, List
import numpy as np
from functools import reduce

def batch_generator(train_x: np.ndarray, train_y: np.ndarray, batch_size: int) :
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    n_samples = train_x.shape[0]
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield train_x[batch_indices], train_y[batch_indices]
    

class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, x: np.ndarray):
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return (1 / (1 + np.exp(-x)))

    def derivative(self, x) -> np.ndarray:
        gx = self.forward(x)
        return gx * (1 - gx)
        

class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray):
        return math.tanh(x)

    def derivative(self, x):
        gx = self.forward(x)
        return 1 - np.square(gx) 


class Relu(ActivationFunction):
    def forward(self, x: np.ndarray):
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
    def loss(self, y_true, y_pred) -> np.ndarray:
        ...
       
    @abstractmethod
    def derivative(self, y_true, y_pred) -> np.ndarray:
        ...


class SquaredError(LossFunction):
    def loss(self, y_true, y_pred) -> np.ndarray:
        diffs = [((y - yhat) ** 2) for y, yhat in zip(y_true, y_pred)]
        return np.array(diffs)
        
    def derivative(self, y_true, y_pred) -> np.ndarray:
        return np.array([2. * (y - yhat) for y, yhat in zip(y_true, y_pred)])

class CrossEntropy(LossFunction):
    def loss(self, y_true, y_pred) -> np.ndarray:
        return np.array([y * math.log(yhat) for y, yhat  in zip(y_true, y_pred)])
        

    def derivative(self, y_true, y_pred) -> np.ndarray:
        raise NotImplemented()



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
        # self.activations = self.O 
        self.activations = None


        self.Z: np.ndarray
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta: np.ndarray|None = None

        # Initialize weights and biaes

        # we need a weights matrix where each row is a connection between this layer and next
        # that way we can multiply and get m x N * M x n
        scale = max(1., 6/(fan_in+fan_out))
        limit = math.sqrt(scale)
        self.W = np.random.uniform(-limit, limit, size=(fan_in, fan_out))
        print("weights shape", self.W.shape)
        self.b = np.random.rand(fan_out) # biases
        print("bias shape", self.b.shape)

    def forward(self, h: np.ndarray) -> np.ndarray:
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """
        self.Z = h @ self.W + self.b
        self.activations = self.activation_function.forward(self.Z)
        return self.activations

    def backward(self, 
        prev: np.ndarray,
        delta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param os: input to this layer :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """

        if self.activations is None or self.Z is None:
            raise Exception("Layer has not been activated with forward()")

        dO_dZ  = self.activation_function.derivative(self.activations)

#        print("dO_dZ", dO_dZ)
#        print("do_dL", dodl)
        # on first layer, delta is derivative of loss
        hadmard =  np.multiply(delta, dO_dZ)

        dL_dW = np.dot(np.transpose(prev), hadmard)

        # derivative of Z wrt b is just 1!
        dL_db = np.sum(hadmard, axis=0)

        # saving the computation of do_dL
        self.delta = np.dot(hadmard, self.W)
        
        return dL_dW, dL_db


class MultilayerPerceptron:
    def __init__(self, layers: List[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        assert len(layers) >= 3

        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output, Y hat
        """
        # as i grow older i realize life is reducible
        def layer_reducer(acc: np.ndarray, lyr: Layer): 
            return lyr.forward(acc)

        return reduce(layer_reducer, self.layers[1:], self.layers[0].forward(x))


    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_gradient: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        # loss grad gives us direction of steepest ascent. Terefore, to minimize, 
        # we take steps in opposite direciton
        # input_data at first one is yhat
        dl_dw_all = []
        dl_db_all = []
        
        # calculate first layer backprop and delta
        rev_layers = reversed(self.layers)
        cur_delta = None
        cur_z = input_data
        for cur_lyr in rev_layers:
            print("Backpropping...")
            if cur_delta is None:
                dl_dw, dl_db = cur_lyr.backward(prev=cur_z, delta=loss_grad)
            else:
                dl_dw, dl_db = cur_lyr.backward(prev=cur_z, delta=cur_delta)
            cur_delta = cur_lyr.delta
            cur_z = cur_lyr.Z
            dl_dw_all.append(dl_dw)
            dl_db_all.append(dl_db)

        return dl_dw_all,dl_db_all 

    def train(self, 
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        loss_func: LossFunction,
        learning_rate: float=1E-3, 
        batch_size: int=16,
        epochs: int=32
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        batches = batch_generator(train_x, train_y, batch_size) 
        training_losses = []
        validation_losses = [] 
        for epoch in range(epochs):
            total_loss = 0
            for input, target in batches:
                feed_forward_output = self.forward(input);
                # loss gradient is derivative of components
                loss_gradient = loss_func.derivative(target, feed_forward_output)
                
                dl_dw_all, dl_db_all = self.backward(loss_gradient, feed_forward_output)
                for wgrad, bgrad, layer in zip(reversed(dl_dw_all), reversed(dl_db_all), self.layers):
                    layer.W  -= learning_rate * wgrad
                    layer.b  -= learning_rate * bgrad

                val_loss = loss_func.loss(feed_forward_output, target)
                train_loss = total_loss / len(train_x)
            # average loss
            print("Epoch ::", epoch, " ")
            training_losses.append(0)


        return np.array(training_losses), np.array(validation_losses)
