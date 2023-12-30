# a4

# K-Nearest Neighbors Classifier Implementation

![image](https://media.github.iu.edu/user/24375/files/57f4abce-86ff-443a-b26c-cb559c3b5687)


## Problem Description

We're tasked with building a k-nearest neighbors (KNN) classifier from scratch. We've got some starter code in two files, `utils.py` and `k_nearest_neighbors.py`, that we need to fill out. The `utils.py` file has utility functions for calculating Euclidean and Manhattan distances, and the `k_nearest_neighbors.py` file has a `KNearestNeighbors` class, which is our KNN classifier.

## Strategy and code explaination:

### Utility Functions (`utils.py`)

#### `euclidean_distance(x1, x2)`
This function calculates the Euclidean distance between two vectors. The Euclidean distance is the straight-line distance between two points in a space. We're using the NumPy library here to make array operations faster and easier.

![CodeCogsEqn-2](https://media.github.iu.edu/user/24660/files/eb7f2c21-81fc-4e4b-bc78-44b145f4c02c)


#### `manhattan_distance(x1, x2)`
This function calculates the Manhattan distance between two vectors. The Manhattan distance is the sum of the absolute differences in each dimension. Just like the Euclidean distance function, it uses NumPy for array operations.

![CodeCogsEqn-3](https://media.github.iu.edu/user/24660/files/a206dc13-2a07-4858-88aa-423fc267988f)

### K-Nearest Neighbors Class (`k_nearest_neighbors.py`)

#### `__init__(self, n_neighbors=5, weights='uniform', metric='l2')`
This is the constructor for the KNearestNeighbors class. It initializes the KNN classifier with the specified number of neighbors, weights, and distance metric.

#### `fit(self, X, y)`
This method fits the model to the provided data matrix `X` and targets `y`. It just assigns the input data and target values to the class attributes `_X` and `_y`.

#### `predict(self, X)`
This method predicts class target values for the given test data matrix `X` using the fitted classifier model. For each test sample, it calculates distances to all training samples, finds the nearest neighbors, and predicts the class based on either uniform or distance-weighted voting.

## Testing

We tested the implementation using a driver program, which compares the accuracy scores of our custom KNN implementation with scikit-learn's KNN implementation on two datasets and different parameter settings. The accuracy scores are saved in `knn_iris_results.html` and `knn_digits_results.html`.

## Observations

### Consistent Accuracy: 

In most cases, the accuracy scores of our custom KNN model are pretty close to those of the scikit-learn implementation, which means our custom implementation is working as expected.

### Stability: 

Across different configurations (number of neighbors, weight function, distance metric), both models consistently achieve high accuracy scores, suggesting the KNN algorithm is pretty stable and reliable on both datasets.

### Uniform vs. Distance Weighting: 

Whether we choose uniform or distance-based weighting doesn't really change the model's performance much in this context.

## Conclusion

The results show that our custom KNN implementation performs pretty well compared to the scikit-learn KNN on the datasets. The KNN algorithm is robust and provides reliable predictions across various parameter settings. The small difference in accuracy scores can be chalked up to implementation differences and the inherent variability in machine learning algorithms.


#  Part 2:

### Utility Functions (`utils.py`)

#### `identity(x, derivative = False)`
This function returns the input `x` as it is. If `derivative` is `True`, it returns an array of ones with the same shape as `x`.

#### `sigmoid(x, derivative = False)`
This function applies the sigmoid function to `x`. The sigmoid function maps any value to a range between 0 and 1. If `derivative` is `True`, it returns the derivative of the sigmoid function.

#### `tanh(x, derivative = False)`
This function applies the hyperbolic tangent function to `x`, which maps any value to a range between -1 and 1. If `derivative` is `True`, it returns the derivative of the tanh function.

#### `relu(x, derivative = False)`
This function applies the Rectified Linear Unit (ReLU) function to `x`, which sets all negative values in `x` to 0. If `derivative` is `True`, it returns 1 for all positive values of `x` and 0 for all negative values.

#### `softmax(x, derivative = False)`
This function applies the softmax function to `x`, which is useful in multiclass classification problems. The softmax function outputs a vector that represents the probabilities of each class. If `derivative` is `True`, it returns the derivative of the softmax function.

#### `cross_entropy(y, p)`
This function calculates the cross-entropy loss between the true labels `y` and the predicted probabilities `p`. Cross-entropy loss is commonly used in classification problems.

#### `one_hot_encoding(y)`
This function converts the class labels `y` into a one-hot encoded matrix. In the resulting matrix, each row corresponds to a sample and each column corresponds to a class. The entry at a specific row and column is 1 if the sample belongs to the class and 0 otherwise.

## Multilayer Perceptron: 

![image](https://media.github.iu.edu/user/24375/files/9fe6695f-b9f0-48dc-9538-f9685ea4b196)

## The goal in this part is to implement a feedforward fully-connected multilayer perceptron classifier with one hidden layer from scratch.

### Strategy and code explaination:

#### 1. Initializing the parameters of the MLP

- We define a Multilayer Perceptron Class where we initialize the primary parameters of the network such as input shape, output shape, iterations, loss function, etc. We have used cross-entropy as our loss function and softmax activation function for our final layer output activation. We also have defined a new parameter called "regularization strength" to avoid overfitting of the model.

####  A- "_initialize(self, X, y)" method.

- The function "initialize" performs three main things.
1. We initially initialize the weights randomly, but later due to poor performance, we changed into He-Initialization.
2. We one-hot encode our target variable.
3. We initialize the biases for the network.

We have used He-Initialization primarily to avoid the exploding and vanishing gradient problem in the network so that the network can learn effectively.

#### B- fit(self, X, y) method.

-The fit method is where we define our neural network.
-Instead of training the network on the entire dataset and updating the gradients, we have used batched learning with a batch size of 30.
-First, We perform a dot product of our training data with the weights and the biases.
-We pass this output to the activation function. This is where non-linearity is introduced.
-Once we get the output from the neural network, we find the difference between the actual and predicted output from which we calculate the error.
-Based on the error at every iteration, and the learning parameter, the gradients are updated so that the model converges to a solution.


#### C-  "predict(self, X)" method.

- Once we have trained our model and we have our weights and biases, we calculate the output for our testing data.

Once we have our predictions from the model, and the actual output from our testing data, we calculate the accuracy of the model

## 2. Challenges We faced

-We initially played around with adding a regularization parameter to avoid overfitting of the model. We used a range of values from 0.001- 0.01. Though we did get good accuracy for some of the models built, overall the accuracy was not as good as scikit-learn accuracies.

-Instead of training the model with the entire data, and updating the weights after every iteration, we used batch learning. The best accuracies we obtained were for the batch size of 30.


##  3. Conclusion

There are some instances where our model has performed better than scikit-learn models, and very few instances where it is average. But the overall accuracies are definitely closer to scikit-learn's accuracies.
