import numpy as np


def sigmoid(x):
    ```
    This function returns the sigmoid function evaluated at x
    ```

    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x):
    ```
    This function returns the gradient of the sigmoid function evaluated at x
    ```

    s = sigmoid(x)
    return s * (1 - s)


def initialize_parameters(layer_dims):
    ```
    This function initializes the parameters of the network. It has input
    layer_dims, which is a list containing the number of nodes in each layer
    of the network and returns a dictionary of parameters
    ```

    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters[f'W{l}]'] = np.random.randn(
            layer_dims[l], layer_dims[l - 1]) * np.sqrt(2/layer_dims[l - 1]
                                                        )
        parameters[f'b{l}]'] = np.zeros((layer_dims[l], 1))

    return parameters


def forward_propagation(X, parameters):
    ```
    This function computes the forward propagation through the network.
    It has input X, which is the input data, and parameters, which is
    a dictionary of parameters.
    ```

    cache = {'A0': X}  # Dictionary to store values of intermediate layers
    L = len(parameters) // 2

    # Get previously activated value
    for l in range(1, L + 1):
        A_prev = cache[f'A{l - 1}']

        # Calculate linear combination
        Z = np.dot(parameters[f'W{l}'], A_prev) + parameters[f'b{l}']
        cache[f'Z{l}'] = Z

        # Apply sigmoid activation
        chache[f'A{l}'] = sigmoid(Z)

    return cache


def compute_cost(AL, Y):
    ```
    This function computes the cost function
    ```

    m = Y.shape[1]
    cost = (1/(2*m)) * np.sum(np.square(AL - Y))

    return cost


def backward_propagation(parameters, chache, X, Y):
    ```
    This function computes the backward propagation
    ```

    grads = {}
    L = len(parameters) // 2
    m = Y.shape[1]

    return 0
