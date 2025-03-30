# XOR prblem neural network

Simple neural netowk for XOR problem. But realistically it is netowrk for anything which has non-linear decision boundary.

## What it is

This implementation of neural network has overall three layers.
One layer of hidden neurons with 4 neurons,
two neurons in input (because XOR problem has two inputs) and one
neuron for output layer.
And sigmoid as activation function.

Also, we must have Hidden layer because XOR problem is not linearly separable.

## Components and Parameters

### Parameters

- ``W1`` - Weights connecting input → hidden layer.
- ``b1`` - Bias for the hidden layer.
- ``W2`` - Weights connecting hidden → output layer.
- ``b2`` - Bias for the output layer.
- ``lr`` - Learning rate (step size for updates).
- ``sigmoid`` - Activation function for non-linearity.
- ``sigmoid_derivative`` - Used for backpropagation updates.
- ``epochs`` - Number of training iterations.

### Workflow

- **Initialization**: Randomly initialize weights and biases.
- **Forward Pass**:
    - Compute the output of the hidden layer using the sigmoid activation function.
    - Compute the final output using the weights and biases of the output layer.
- **Backpropagation**:
    - Calculate the error between predicted and actual output.
    - Compute gradients for weights and biases using the sigmoid derivative.
    - Update weights and biases using the gradients and learning rate.
- **Training Loop**:
    - Repeat the forward pass and backpropagation for a specified number of epochs.
- **Prediction**:
    - After training, use the learned weights and biases to make predictions on new data.

## Results

![XOR Neural Network](../results/XOR%20problem%20boundary.png)
Display of decision boundary learned by the neural network.