import numpy as np
from random import shuffle

def structured_loss_simple(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if j == y[i]:
        continue

      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW   += reg * W


  return loss, dW


def structured_loss_fast(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as structured_loss_simple.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  scores = X.dot(W)
  correct_class_score = scores[range(y.shape[0]), y]
  margin = scores - np.reshape(correct_class_score, (X.shape[0], -1)) + 1
  
  loss = margin
  loss[loss < 0] = 0
  loss[range(y.shape[0]), y] = 0
  loss = np.sum(loss)
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= X.shape[0]

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  # Calcualte gradients based on margins where loss was accounted for
  grad = (margin > 0) * 1
  grad[range(y.shape[0]), y] = 0
  grad[range(y.shape[0]), y] = -np.sum(grad, axis=1)
    
  dW = X.T.dot(grad)/X.shape[0]


  return loss, dW
