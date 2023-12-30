# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: [Bhanuprakash , Dilip Nikhil Francies, Prinston Rebello] -- [bhnaraya dfranci prebello]
#
# Based on skeleton code by CSCI-B 551 Fall 2023 Course Staff

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden = 16, hidden_activation = 'relu', n_iterations = 2000, learning_rate = 0.01,reg_strength=0.005):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}
        

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.reg_strength = reg_strength     # regularization parameter
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax   # output activation for our output layer to predict prob
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None

    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        fan_in = X.shape[1]
        self._X = X
        self._y = one_hot_encoding(y)  # one hot encoding
        self._h_weights = np.random.randn(fan_in, self.n_hidden) / np.sqrt(fan_in)  # He intialization
        self._h_bias = np.zeros((1, self.n_hidden))
        self._o_weights = np.random.randn(self.n_hidden, self._y.shape[1]) / np.sqrt(self.n_hidden)
        self._o_bias = np.zeros((1, self._y.shape[1]))
        
        
    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y and stores the cross-entropy loss every 20
        iterations.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        self._initialize(X, y)
        for i in range(self.n_iterations):
            batch_size = 30       # batch size
            indices = np.random.choice(X.shape[0], size=batch_size, replace=True) # pick random observations based on the batch size
            X_batch, y_batch = self._X[indices], self._y[indices] # our input data
            h_input = np.dot(X_batch, self._h_weights) + self._h_bias # dot product of weights and input data and add biases
            h_output = self.hidden_activation(h_input) # send the input from the NN to activation function
            o_input = np.dot(h_output, self._o_weights) + self._o_bias
            o_output = self._output_activation(o_input)
        
           # backpropogation starts here
            o_error = y_batch - o_output #calculate the error
            o_delta = o_error * self._output_activation(o_output, True) #our gradients
            h_error = o_delta.dot(self._o_weights.T) #error in the hidden layer calculated to update the gradients
            h_delta = h_error * self.hidden_activation(h_output, True) #gradient of the loss

            self._o_weights += h_output.T.dot(o_delta) * self.learning_rate #update the weights connecting the hidden layer to the o/p layer
            self._o_bias += np.sum(o_delta, axis=0, keepdims=True) * self.learning_rate #update the bias
            self._h_weights += X_batch.T.dot(h_delta) * self.learning_rate #update the weights based on the gradient of the loss
            self._h_bias += np.sum(h_delta, axis=0, keepdims=True) * self.learning_rate
            #self._h_weights -= self.reg_strength * self._h_weights   # regularize
            #self._o_weights -= self.reg_strength * self._o_weights

            if i % 20 == 0:
                loss = self._loss_function(y_batch, o_output)
                self._loss_history.append(loss)
    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        h_input = np.dot(X, self._h_weights) + self._h_bias # input to the NN
        h_output = self.hidden_activation(h_input) #o/p
        o_input = np.dot(h_output, self._o_weights) + self._o_bias
        o_output = self._output_activation(o_input) #activation function output of the last layer
        return np.argmax(o_output, axis=1) # get the predicted class
