"""
Softmax Classifier
"""
import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_class = W.shape[1]
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        ps = np.exp(scores) / np.sum(np.exp(scores))
        loss += -np.log(ps[y[i]])
        for j in range(num_class):
            if j == y[i]:
                dW[:, j] += (ps[j] - 1) * X[i]
            else:
                dW[:, j] += (ps[j] - 0) * X[i]
    loss = loss/num_train + 0.5 * reg * np.sum(W ** 2)
    dW = dW/num_train + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_class = W.shape[1]
    scores = X.dot(W)
    scores -= np.max(scores, axis=1).reshape(-1, 1)
    ps = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1, 1)
    loss += -np.mean(np.log(ps[np.arange(num_train), y]))
    loss += 0.5 * reg * np.sum(W ** 2)

    delta = ps
    ps[np.arange(num_train), y] -= 1.0
    dW = (X.T).dot(delta)/num_train + reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
