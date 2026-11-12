import numpy as np

class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) Raw vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        ### BEGIN CODE HERE
        N = y_true.shape[0]
        return np.sum((y_pred-y_true)**2) / N ### END CODE HERE

    @staticmethod
    def gradient(y_true, y_pred):
        ### BEGIN CODE HERE
        N = y_true.shape[0]
        return 2*(y_pred-y_true) / N ### END CODE HERE

class CrossEntropy:
    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true: (array) One hot encoded truth vector.
        :param y_pred: (array) Prediction vector
        :return: (flt)
        """
        
        ### BEGIN CODE HERE
        logits = y_pred - np.max(y_pred, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return -np.mean(np.sum(y_true * np.log(softmax + 1e-10), axis=1)) ### END CODE HERE

    @staticmethod
    def gradient(y_true, y_pred):
        ### BEGIN CODE HERE
        exp_pred = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        softmax_pred = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        return (softmax_pred - y_true) / y_true.shape[0] ### END CODE HERE