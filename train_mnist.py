#!/usr/bin/env python3
from mlp import ActivationFunction, CrossEntropy, Layer, LeakyRelu, Linear, MultilayerPerceptron, Relu, Sigmoid, Softmax, SquaredError, Tanh, batch_generator
import struct
import numpy as np
from array import array

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (np.array(x_train), np.array(y_train)),(np.array(x_test), np.array(y_test))        

 

def instantiate_model(layers: list[Layer], is_training=False):
    """
    Placeholder function to instantiate your MLP model.
    """
    print("Instantiating the MLP model...")
    
    return MultilayerPerceptron(layers, is_training=is_training)


def main():
    # Instantiate and train the model.
    model = instantiate_model([
        Layer(fan_in=28*28,  fan_out=64,       activation_function=  Relu()),
        Layer(fan_in=64,    fan_out=87,       activation_function= Relu()), 
        Layer(fan_in=87,    fan_out=72,       activation_function= LeakyRelu()), 
        Layer(fan_in=72,     fan_out=10,       activation_function= Softmax()), 
    ])
    loss = CrossEntropy()
    mnist_dataloader = MnistDataloader(
        training_images_filepath="data/train-images.idx3-ubyte",
        training_labels_filepath="./data/train-labels.idx1-ubyte",
        test_images_filepath="./data/t10k-images.idx3-ubyte",
        test_labels_filepath="./data/t10k-labels.idx1-ubyte",
    )
    (train_x, train_y), (test_x, test_y) = mnist_dataloader.load_data()
    train_x = train_x.reshape(train_x.shape[0], 784) 
    test_x =  test_x.reshape(test_x.shape[0], 784) 
    x_mean = np.mean(train_x)
    x_std = np.std(test_x)

    train_x = (train_x - x_mean) / x_std 
    test_x = (test_x - x_mean)  /  x_std

    
    train_y = np.eye(10)[train_y]
    prev_test_y = np.copy(test_y)
    test_y = np.eye(10)[test_y]

    np.testing.assert_array_equal(np.argmax(test_y, axis=1), prev_test_y)

    train_data = np.array(list(zip(train_x, train_y)), dtype=object)  # Convert to NumPy array

    # Shuffle the data to ensure randomness
    np.random.shuffle(train_data)

    # Calculate the split index
    split_idx = int(0.8 * len(train_data))  # 80% for training, 20% for validation

    # Split the data
    train_split = train_data[:split_idx]  # First 80% for training
    val_split = train_data[split_idx:]    # Remaining 20% for validation

    # Separate features (images) and labels
    train_x_split = np.array([item[0] for item in train_split])
    train_y_split = np.array([item[1] for item in train_split])
    val_x_split = np.array([item[0] for item in val_split])
    val_y_split = np.array([item[1] for item in val_split])
    model.train( 
        train_x=train_x_split,
        train_y=train_y_split,
        val_x=val_x_split,
        val_y=val_y_split,
        loss_func=loss,
        learning_rate=1E-3,
        batch_size=32,
        epochs=18
    )

    predicted_labels  = np.argmax(model.forward(test_x), axis=1)
    true_labels  = np.argmax(test_y, axis=1)
    test_accuracy = np.mean(predicted_labels == true_labels)

    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    decoded = list(enumerate(np.argmax(test_y, axis=1)))

    indices = []
    found = set()

    for i, el in decoded:
        if len(indices) == 10:
            break
        if el not in found:
            indices.append([i])
            found.add(el)
            
    # Extract one example of each digit
    selected_images = test_x[indices]
    selected_labels = test_y[indices]
            
    for img, lbl in zip(selected_images, selected_labels):
        print("Predicted:", np.argmax(model.forward(img)), "true:", np.argmax(lbl))
    

if __name__ == "__main__":
    main()
