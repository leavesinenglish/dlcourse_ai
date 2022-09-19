import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layers = [FullyConnectedLayer(n_input, hidden_layer_size), ReLULayer(),
                       FullyConnectedLayer(hidden_layer_size, n_output)]

    def forward(self, X):
        x = X.copy()
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        params = self.params()
        for key, value in params.items():
            value.grad = 0

        data = X
        for layer in self.layers:
            data = layer.forward(data)
        loss, d_out = softmax_with_cross_entropy(data, y)
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        for key, value in params.items():
            loss_l2, grad_l2 = l2_regularization(value.value, self.reg)
            value.grad += grad_l2
            loss += loss_l2
        return loss


    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        pred = np.zeros(X.shape[0], np.int)
        pred = self.forward(X)
        return np.argmax(pred, axis=1)

    def params(self):
        result = {
            'layer_1_W': self.layers[0].params()['W'],
            'layer_1_B': self.layers[0].params()['B'],
            'layer_2_W': self.layers[2].params()['W'],
            'layer_2_B': self.layers[2].params()['B']
        }
        return result
