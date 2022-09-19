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
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
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
    """
    soft_max = softmax(preds)
    loss = cross_entropy_loss(soft_max, target_index)
    dprediction = soft_max
    if soft_max.ndim == 1:
        dprediction[target_index] -= 1
    else:
        dprediction[range(preds.shape[0]), target_index] -= 1
        dprediction /= preds.shape[0]
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X
        X = X.reshape(-1)
        res = np.array([0 if X[i] < 0 else X[i] for i in range(X.size)])
        res = res.reshape(self.X.shape[0], self.X.shape[1])
        return res

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
        X = self.X.reshape(-1)
        d_result = np.array([0 if X[i] < 0 else 1 for i in range(X.size)])
        d_result = d_result.reshape(self.X.shape[0], self.X.shape[1])
        return d_out * d_result

    def params(self):
        # ReLU Doesn't have any parameters
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
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.dot(np.ones((1,d_out.shape[0])), d_out)
        return np.dot(d_out, self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}
