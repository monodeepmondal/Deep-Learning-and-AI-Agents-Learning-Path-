"""
Neural Networks Basics Tutorial
This file introduces the fundamental concepts of neural networks through a simple perceptron implementation.
"""

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """
    A simple perceptron implementation for binary classification.
    The perceptron is the building block of neural networks.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def activation(self, x):
        """Step activation function"""
        return 1 if x >= 0 else 0
    
    def fit(self, X, y):
        """
        Train the perceptron
        
        Parameters:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Training loop
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # Calculate prediction
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_output)
                
                # Update weights and bias
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update
    
    def predict(self, X):
        """Make predictions for new data"""
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(x) for x in linear_output])

# Example usage
if __name__ == "__main__":
    # Create a simple dataset for AND gate
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])  # AND gate output
    
    # Create and train the perceptron
    perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
    perceptron.fit(X, y)
    
    # Test the perceptron
    predictions = perceptron.predict(X)
    print("\nPerceptron Predictions:")
    for i, (x, pred) in enumerate(zip(X, predictions)):
        print(f"Input: {x}, Predicted: {pred}, Actual: {y[i]}")
    
    # Visualize the decision boundary
    plt.figure(figsize=(8, 6))
    
    # Plot the data points
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
    
    # Plot the decision boundary
    x1 = np.linspace(-0.5, 1.5, 100)
    x2 = -(perceptron.weights[0] * x1 + perceptron.bias) / perceptron.weights[1]
    plt.plot(x1, x2, 'g-', label='Decision Boundary')
    
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Perceptron Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()

"""
Key Concepts Explained:

1. Perceptron:
   - The simplest form of a neural network
   - Takes multiple inputs, applies weights, and produces a binary output
   - Can learn to classify linearly separable data

2. Components:
   - Weights: Parameters that determine the importance of each input
   - Bias: A constant term that helps shift the decision boundary
   - Activation Function: Converts the weighted sum into a binary output

3. Learning Process:
   - The perceptron learns by adjusting weights and bias
   - Updates are based on the error between predicted and actual output
   - Learning rate controls how quickly the model adapts

4. Limitations:
   - Can only learn linearly separable patterns
   - Cannot solve XOR problem
   - Limited to binary classification

Next Steps:
- Understanding backpropagation
- Implementing multi-layer neural networks
- Exploring different activation functions
""" 