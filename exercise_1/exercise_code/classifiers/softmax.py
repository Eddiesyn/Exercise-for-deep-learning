"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    lgc = - np.dot(X, W).max(axis = 1, keepdims = True)
    Rw = 0
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Rw += (W[i,j]**2)
            
    for i in range(X.shape[0]):
        f_i = np.dot(np.expand_dims(X[i,:], axis=0), W)
        f_i += lgc[i]
        part1 = -f_i[:,y[i]]
        temp = np.sum(np.exp( f_i ))
        part2 = np.log(temp)
        loss += (part1 + part2)
        D_i = np.exp(f_i) / temp
        D_i[:,y[i]] -= 1
        dW += np.dot(np.expand_dims(X[i,:], axis=0).T, D_i)

    loss /= X.shape[0]
    loss += (0.5 * reg * Rw)
    dW /= X.shape[0]
    dW += reg*W
    
    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    N = X.shape[0]
    C = W.shape[1]
    
    f = np.dot(X, W)
    f -= np.max(f, axis=1, keepdims=True)
    part1 = -f[np.arange(N), y]
    temp = np.sum(np.exp(f),axis=1)
    part2 = np.log( temp )
    loss = np.sum(part1 + part2) / N + 0.5*reg*np.sum(W**2)
    
    D = np.exp(f) / temp.reshape((N, 1))
    D[np.arange(N), y] -= 1
    dW = np.dot(X.T, D) / N
    dW += reg*W
    
    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [3e-6, 4e-6, 6e-6, 8e-6, 1e-5]
    regularization_strengths = [2e3, 4e3, 6e3, 8e3]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    for lr in learning_rates:
        for reg in regularization_strengths:
            softmax_ = SoftmaxClassifier()
            all_classifiers.append(softmax_)
            loss_hist = softmax_.train(X_train, y_train, learning_rate=lr, reg=reg,
                          num_iters=150, verbose=False)
            pred_train = softmax_.predict(X_train)
            pred_val = softmax_.predict(X_val)
            acc_train = np.mean(y_train == pred_train)
            acc_val = np.mean(y_val == pred_val)
            results[lr,reg] = acc_train, acc_val
            if acc_val > best_val:
                best_val = acc_val
                best_softmax = softmax_
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
