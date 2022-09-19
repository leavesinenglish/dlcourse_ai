import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    pred = np.copy(predictions)
    probs = {}
    if (pred.ndim == 1):
        pred -= np.max(predictions)
        probs = np.exp(pred) / np.sum(np.exp(pred))
    else:
        pred = [pred[i] - np.max(pred[i]) for i in range(pred.shape[0])]
        exp_pred = np.exp(pred)
        exp_sum = np.sum(exp_pred, axis=1)
        probs = np.asarray([exp_pred[i] / exp_sum[i] for i in range(exp_pred.shape[0])])
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if probs.ndim == 1:
        return -np.log(probs[target_index])
    else:
        log_s = -np.log(probs[range(probs.shape[0]), target_index])
        return np.sum(log_s) / probs.shape[0]



def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    soft_max = softmax(predictions)
    loss = cross_entropy_loss(soft_max, target_index)
    dprediction = soft_max
    if soft_max.ndim == 1:
        dprediction[target_index] -= 1
    else:
        dprediction[range(predictions.shape[0]), target_index] -= 1
        dprediction /= predictions.shape[0]

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return np.where(X<0,0,X)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        return np.where(self.X<0,0,1)*d_out

    def params(self):
        return {}



class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.dot(np.ones((1,d_out.shape[0])), d_out)
        return np.dot(d_out, self.W.value.T)

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1

        self.X = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant',
                        constant_values=0)
        W = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)

        XW = np.zeros((batch_size, out_height, out_width, self.out_channels))

        for y in range(out_height):
            for x in range(out_width):
                x_board = x + self.filter_size
                y_board = y + self.filter_size
                fragment = self.X[:, y:y_board, x:x_board, :].reshape(batch_size, (self.filter_size**2)*channels)
                XW[:, y, x, :] = np.dot(fragment, W)

        return XW + self.B.value
                
                


    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        dX = np.zeros((batch_size, height, width, channels))
        W = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)
        for y in range(out_height):
            for x in range(out_width):
                x_board = x + self.filter_size
                y_board = y + self.filter_size
                fragment = self.X[:, y:y_board, x:x_board,:]
                fragment_arr = fragment.reshape(batch_size, (self.filter_size **2)* self.in_channels)
                d_local = d_out[:, y:y + 1, x:x + 1, :]
                dX_arr = np.dot(d_local.reshape(batch_size, -1), W.T)
                dX[:, y:y_board, x:x_board, :] += dX_arr.reshape(fragment.shape)
                dW = np.dot(fragment_arr.T, d_local.reshape(batch_size, -1))
                dB = np.dot(np.ones((1, d_local.shape[0])), d_local.reshape(batch_size, -1))
                self.W.grad += dW.reshape(self.W.value.shape)
                self.B.grad += dB.reshape(self.B.value.shape)
        return dX[:, self.padding: (height - self.padding), self.padding: (width - self.padding), :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape
        out_height = int((height - self.pool_size) / self.stride + 1)
        out_width = int((width - self.pool_size) / self.stride + 1)

        output = np.zeros((batch_size, out_height, out_width, channels))

        for y in range(out_height):
            for x in range(out_width):
                output[:, y, x, :] += np.amax(X[:, y * self.stride : y * self.stride + self.pool_size, x * self.stride : x * self.stride + self.pool_size, :], axis=(1, 2))
        return output

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, _ = d_out.shape
        
        d_input = np.zeros_like(self.X)
        batch_idxs = np.repeat(np.arange(batch_size), channels)
        channel_idxs = np.tile(np.arange(channels), batch_size)
        
        for y in range(out_height):
            for x in range(out_width):
                slice_X = self.X[:, y * self.stride : y * self.stride + self.pool_size, x * self.stride : x * self.stride + self.pool_size, :].reshape(batch_size, -1, channels)
                max_idxs = np.argmax(slice_X, axis=1) 
                slice_d_input = d_input[:, y * self.stride : y * self.stride + self.pool_size, x * self.stride : x * self.stride + self.pool_size, :].reshape(batch_size, -1, channels)
                slice_d_input[batch_idxs, max_idxs.flatten(), channel_idxs] = d_out[batch_idxs, y, x, channel_idxs]
                d_input[:, y * self.stride : y * self.stride + self.pool_size, x * self.stride : x * self.stride + self.pool_size, :] = slice_d_input.reshape(batch_size, self.pool_size, self.pool_size, channels)
        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        return X.reshape(batch_size, height*width*channels)
        

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)
        

    def params(self):
        return {}
