"""
Backpropagation and Gradient Descent Tutorial
This file demonstrates the implementation of backpropagation and gradient descent
in a simple neural network with one hidden layer.
"""

import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    """
    A simple neural network with one hidden layer implementing backpropagation.
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights with random values
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        
        # Initialize biases
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def forward(self, X):
        """Forward propagation"""
        # Hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        # Output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_input)
        
        return self.predicted_output
    
    def backward(self, X, y, output):
        """Backward propagation"""
        # Calculate output layer error
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        # Calculate hidden layer error
        self.hidden_error = np.dot(self.output_delta, self.weights_hidden_output.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.learning_rate * np.dot(self.hidden_output.T, self.output_delta)
        self.weights_input_hidden += self.learning_rate * np.dot(X.T, self.hidden_delta)
        
        self.bias_output += self.learning_rate * np.sum(self.output_delta, axis=0, keepdims=True)
        self.bias_hidden += self.learning_rate * np.sum(self.hidden_delta, axis=0, keepdims=True)
    
    def train(self, X, y, epochs):
        """Train the neural network"""
        errors = []
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate error
            error = np.mean(np.square(y - output))
            errors.append(error)
            
            # Backward pass
            self.backward(X, y, output)
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Error: {error}")
        
        return errors

# Example usage
if __name__ == "__main__":
    # Create a simple XOR dataset
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    # Create and train the neural network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
    errors = nn.train(X, y, epochs=10000)
    
    # Test the network
    predictions = nn.forward(X)
    print("\nNeural Network Predictions:")
    for i, (x, pred) in enumerate(zip(X, predictions)):
        print(f"Input: {x}, Predicted: {pred[0]:.4f}, Actual: {y[i][0]}")
    
    # Plot the training error
    plt.figure(figsize=(10, 6))
    plt.plot(errors)
    plt.title('Training Error Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()

"""
Key Concepts Explained:

1. Backpropagation:
   - Algorithm for training neural networks
   - Calculates gradients of the loss function with respect to weights
   - Propagates error backwards through the network

2. Gradient Descent:
   - Optimization algorithm that minimizes the loss function
   - Updates weights in the direction of steepest descent
   - Learning rate controls the size of weight updates

3. Components:
   - Forward Pass: Computes predictions
   - Backward Pass: Computes gradients
   - Weight Updates: Adjusts network parameters

4. Activation Functions:
   - Sigmoid: Maps values to range (0,1)
   - Derivative: Used in backpropagation
   - Other options: ReLU, tanh, etc.

Next Steps:
- Implementing deep neural networks
- Adding more layers
- Using different activation functions
- Implementing regularization
""" 