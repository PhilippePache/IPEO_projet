import numpy as np # added

class MLP:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: list of dimensions of the neural net. (input, hidden layer, ... ,hidden layer, output)
        :param activations: list of activation functions. Must contain N-1 activation function, where N = len(dimensions).

        Example of one hidden layer with
        - 2 inputs
        - 10 hidden nodes
        - 5 outputs
        layers -->    [0,        1,          2]
        ----------------------------------------
        dimensions =  (2,     10,          5)
        activations = (      Sigmoid,      Sigmoid)
        """

        ### BEGIN CODE HERE
        self.n_layers = len(dimensions)
        self.loss = None
        self.learning_rate = None
        
        # Weights and biases are indexed by layer l.
        self.w = {}
        self.b = {}
        # Activations are also indexed by layer l. 
        self.activations = {}

        for l in range(1, self.n_layers):
            self.w[l] = np.random.randn(dimensions[l], dimensions[l-1]) / np.sqrt(dimensions[l - 1])
            self.b[l] = np.zeros(dimensions[l])
            self.activations[l] = activations[l-1]
        ### END CODE HERE

    def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        ### BEGIN CODE HERE
        # a = Wx + b
        a = {}

        # z = sigma(a)
        z = {0: x}  # First layer := input layer
        
        for l in range(1, self.n_layers):
            # current layer = l
            a[l] = z[l-1] @ self.w[l].T + self.b[l]
            z[l] = self.activations[l].forward(a[l])

        return a, z ### END CODE HERE
    

    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """

        ### BEGIN CODE HERE
        _, z = self.feed_forward(x)
        return z[self.n_layers - 1] ### END CODE HERE


    def back_prop(self, a, z, y_true):
        """
        The input dicts keys represent the layers of the net.
        a = { 0: x,
              1: f(w1(x) + b1)
              2: f(w2(a2) + b2)
              }
        :param a: (dict) w^T@x + b
        :param z: (dict) f(a)
        :param y_true: (array) One hot encoded truth vector.
        :return:
        """

        ### BEGIN CODE HERE
        # Determine partial derivative and delta for the output layer.
        y_pred = z[self.n_layers - 1]
        delta = self.loss_function.gradient(y_true, y_pred) * self.activations[self.n_layers - 1].gradient(a[self.n_layers - 1])
        
        dw = np.dot(delta.T, z[self.n_layers - 2])

        update_params = {
            self.n_layers - 1: (dw, delta)
        }

        # Determine partial derivative and delta for the rest of the layers.
        # Each iteration requires the delta from the previous layer, propagating backwards.
        for l in reversed(range(1, self.n_layers - 1)):
            delta = np.dot(delta, self.w[l+1]) * self.activations[l].gradient(a[l])
            dw = np.dot(delta.T, z[l - 1])
            update_params[l] = (dw, delta)
        
        # finally update weights and biases
        for k, v in update_params.items():
            self.update_w_b(k, v[0], v[1])
        ### END CODE HERE


    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """

        ### BEGIN CODE HERE
        self.w[index] -= self.learning_rate * dw
        self.b[index] -= self.learning_rate * np.sum(delta, 0)
        ### END CODE HERE

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        """

        ### BEGIN CODE HERE
        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")
        # Initiate the loss object with the final activation function
        self.loss_function = loss
        self.learning_rate = learning_rate

        for i in range(epochs):
            # Shuffle the data
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x_ = x[indices]
            y_ = y_true[indices]

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                a, z = self.feed_forward(x_[k:l])
                self.back_prop(a, z, y_[k:l])

            if (i + 1) % 10 == 0:
                _, z = self.feed_forward(x)
                #print("Loss at epoch {}: {}".format(i + 1, self.loss_function.loss(y_true, z[self.n_layers - 1])))
        ### END CODE HERE
