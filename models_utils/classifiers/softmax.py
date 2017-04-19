import numpy as np
from random import shuffle

def softmax_loss_simple(W, X, y, reg):
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
  # If you get stuck, don't forget about these resources:                     #
  # http://cs231n.github.io/neural-networks-case-study/#linear                #
  # http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/#
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
    
  for i in range(num_train):
    scores = X[i].dot(W)
    exp_scores = np.exp(scores)
    norm_prob = exp_scores/np.sum(exp_scores)
    
    correct_log_error = -np.log(norm_prob[y[i]])
    loss += correct_log_error
    
    grad = norm_prob
    grad[y[i]] -= 1
    for j in range(num_class):
        dW[:,j] += grad[j] * X[i]
    
  loss /= num_train
  dW   /= num_train
    
  loss += 0.5*reg* np.sum(W*W)
  dW   += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_fast(W, X, y, reg):
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
  num_dim   = X.shape[1]
  num_class = W.shape[1]
    
  scores = X.dot(W)
  exp_scores = np.exp(scores)
  norm_prob = exp_scores/np.sum(exp_scores, axis=1).reshape(num_train,-1)
  corr_log_error = -np.log(norm_prob[range(num_train),y])
  
  grad = norm_prob
  grad[range(num_train),y] -= 1
             
  dW = X.T.dot(grad)/num_train
  dW += reg*W
  loss += np.sum(corr_log_error)/num_train
  loss += 0.5*reg*np.sum(W*W) 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

