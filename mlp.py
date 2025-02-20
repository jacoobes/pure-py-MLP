import math
from abc import ABC, abstractmethod
from typing import Generator, Tuple, List
import itertools
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
    np.random.shuffle(indices)  # Randomize order of samples
    # Generate batches by slicing the indices
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield train_x[batch_idx], train_y[batch_idx]
    

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

    def derivative(self, x: np.ndarray) -> np.ndarray:
        gx = self.forward(x)
        return gx * (1 - gx)
        

class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray):
        return np.tanh(x)

    def derivative(self, x):
        gx = self.forward(x)
        return 1 - np.square(gx) 


class Relu(ActivationFunction):
    def forward(self, x: np.ndarray):
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray):
        return np.heaviside(x, 1)

class Softmax(ActivationFunction):
    def forward(self, x):
        exp_logits = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability improvement
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def derivative(self, x):
        bsize, num_classes = x.shape
        jacobian = np.zeros((bsize, num_classes, num_classes))

        for i in range(bsize):
            s_i = x[i].reshape(-1, 1) # col vec
            jacobian[i] = np.diagflat(s_i) - (s_i @ s_i.T)
        return jacobian


class Linear(ActivationFunction):
    def forward(self, x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true, y_pred) -> np.ndarray:
        ...
       
    @abstractmethod
    def derivative(self, y_true, y_pred) -> np.ndarray:
        ...


class SquaredError(LossFunction):
    def loss(self, y_true, y_pred) -> np.ndarray:
        return np.square(y_true - y_pred)
        
    def derivative(self, y_true, y_pred) -> np.ndarray:
        return 2 * (y_true - y_pred)

class CrossEntropy(LossFunction):
    def loss(self, y_true, y_pred) -> np.ndarray:
        return y_true * np.log(y_pred)
        

    def derivative(self, y_true, y_pred) -> np.ndarray:
        return -y_true / y_pred 



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
        limit = math.sqrt(6/(fan_in+fan_out))
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
        self.Z = (h @ self.W) + self.b
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

        if isinstance(self.activation_function, Softmax):
            dL_dZ = np.einsum('bij, bj -> bi', dO_dZ, delta) 
        else:
            dL_dZ = np.multiply(delta, dO_dZ)

        #print("prev shape", prev.T.shape, "hadmard shape", dL_dZ.shape)
        dL_dW = prev.T @ dL_dZ
        assert dL_dW.shape == self.W.shape

        # derivative of Z wrt b is just 1!
        dL_db = np.sum(dL_dZ, axis=0, keepdims=True)
        assert dL_db.shape == self.b.shape

        # saving the computation of do_dL
        self.delta = np.dot(dL_dZ, self.W.T)

        #print(dL_dW, dL_db)
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
 
        return reduce(layer_reducer, self.layers, x)


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
        cur_delta = loss_grad
        for cur_lyr, prev_lyr in itertools.pairwise(reversed(self.layers)):
            dl_dW, dl_db = cur_lyr.backward(prev_lyr.activations, cur_delta)
            cur_delta = cur_lyr.delta
            assert cur_delta is not None and prev_lyr.activations is not None
            dl_dw_all.append(dl_dW)
            dl_db_all.append(dl_db)

        # backprop final layer, with input_data being X
        dl_dw, dl_db = self.layers[0].backward(input_data, cur_delta)
        dl_dw_all.append(dl_dw)
        dl_db_all.append(dl_db)

        return dl_dw_all,dl_db_all


    def train(self, 
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        loss_func: LossFunction,
        learning_rate: float=1e-4 , 
        batch_size: int=32,
        epochs: int=42
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
        training_losses = []
        validation_losses = [] 

        for epoch in range(epochs):
            total_loss = 0.
            for input, target in batch_generator(train_x, train_y, batch_size):
                feed_forward_output = self.forward(input);
                loss = loss_func.loss(target, feed_forward_output)
                # loss gradient is derivative of components
                loss_gradient = loss_func.derivative(target, feed_forward_output)
                dl_dw_all, dl_db_all = self.backward(loss_gradient, input)
                for wgrad, bgrad, layer in zip(dl_dw_all, dl_db_all, reversed(self.layers)):
                    layer.W  -=  learning_rate * wgrad
                    layer.b  -=  learning_rate * bgrad
                total_loss += loss

            val_loss = np.mean(loss_func.loss(val_y, self.forward(val_x))) 
            train_loss = np.mean(total_loss / math.ceil(len(train_x) / batch_size))

            training_losses.append(train_loss)
            validation_losses.append(val_loss)
            # average loss
            print("Epoch ::", epoch+1, "::", "Train Loss=", train_loss, "::", "Val Loss", val_loss)


        return np.array(training_losses), np.array(validation_losses)
