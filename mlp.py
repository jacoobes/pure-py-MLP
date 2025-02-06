import math
from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, x):
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return (1 / (1 + math.e ** (-x) ))

    def derivative(self, x):
        gx = self.forward(x)
        return gx * (1 - gx)
        

class Tanh(ActivationFunction):
    def forward(self, x):
        return math.tanh(x)

    def derivative(self, x):
        gx = self.forward(x)
        return 1 - gx ** 2


class Relu(ActivationFunction):
    def forward(self, x ):
        return np.maximum(0, x)

    def derivative(self, x):
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
    def loss(self, y_true, y_pred):
       pass 
       
    def derivative(self, y_true, y_pred):
        pass


class SquaredError(LossFunction):
    def loss(self, y_true, y_pred):
        pass

    def derivative(self, y_true, y_pred):
        pass

class CrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        pass

    def derivative(self, y_true, y_pred):
        pass



class Layer:

    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        """
        Initializes a Layer instance.

        Args:
            fan_in (int): The number of neurons in the layer prior to this one.
            fan_out (int): The number of neurons in this layer.
            activation_function (ActivationFunction): An instance of an activation function.
        """
        pass
    def forward(self, h):
        pass
    def backward(self, h,delta): 
        pass


class MultilayerPerceptron:
    
    def forward(self, x):
        pass 
    def backward(self, loss_grad, input_data): 
        """
            network, 
            computing (𝜕𝑾ℒ, 𝜕+,⃗ ℒ+ for each layer and storing them in lists. The method returns
            tuple of (𝜕𝑾(𝟏) ℒ, 𝜕𝑾(𝟐) ℒ, ⋯ , 𝜕𝑾(𝑵) ℒ), (𝜕+,⃗ (𝟏) ℒ, 𝜕+,⃗ (𝟐) ℒ, ⋯ , 𝜕+,⃗ (𝑵) ℒ) for 𝑁 layers"""

    def train(self, train_x, train_y, val_x, val_y, loss_func, learning_rate, batch_size, epochs):
        pass
