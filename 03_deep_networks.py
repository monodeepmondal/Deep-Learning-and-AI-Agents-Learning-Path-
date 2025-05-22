"""
Deep Neural Networks Tutorial
This file demonstrates the implementation of deep neural networks using TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DeepNeuralNetwork:
    """
    A deep neural network implementation using TensorFlow/Keras.
    """
    def __init__(self, input_shape, hidden_layers, output_shape):
        """
        Initialize the deep neural network.
        
        Parameters:
        input_shape: Shape of input data
        hidden_layers: List of integers representing number of neurons in each hidden layer
        output_shape: Number of output classes
        """
        self.model = self._build_model(input_shape, hidden_layers, output_shape)
    
    def _build_model(self, input_shape, hidden_layers, output_shape):
        """Build the neural network architecture"""
        model = keras.Sequential()
        
        # Input layer
        model.add(keras.layers.Input(shape=input_shape))
        
        # Hidden layers
        for neurons in hidden_layers:
            model.add(keras.layers.Dense(neurons, activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(0.3))
        
        # Output layer
        model.add(keras.layers.Dense(output_shape, activation='softmax'))
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

def plot_training_history(history):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load and preprocess the MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape the data
    X_train = X_train.reshape(-1, 28 * 28)
    X_test = X_test.reshape(-1, 28 * 28)
    
    # Split training data into training and validation sets
    val_size = 5000
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    # Create and train the model
    model = DeepNeuralNetwork(
        input_shape=(784,),
        hidden_layers=[512, 256, 128],
        output_shape=10
    )
    
    # Train the model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=10,
        batch_size=128
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Make predictions on some test samples
    predictions = model.predict(X_test[:5])
    print("\nPredictions for first 5 test samples:")
    for i, pred in enumerate(predictions):
        predicted_class = np.argmax(pred)
        print(f"Sample {i+1}: Predicted class = {predicted_class}, True class = {y_test[i]}")

"""
Key Concepts Explained:

1. Deep Neural Networks:
   - Multiple layers of neurons
   - Each layer learns different features
   - Deeper networks can learn more complex patterns

2. Components:
   - Dense Layers: Fully connected layers
   - Batch Normalization: Normalizes layer inputs
   - Dropout: Prevents overfitting
   - Activation Functions: ReLU, Softmax

3. Training Process:
   - Forward Pass: Computes predictions
   - Backward Pass: Updates weights
   - Optimization: Adam optimizer
   - Loss Function: Cross-entropy

4. Best Practices:
   - Data Normalization
   - Batch Normalization
   - Dropout for regularization
   - Learning rate scheduling
   - Early stopping

Next Steps:
- Implementing Convolutional Neural Networks
- Transfer Learning
- Hyperparameter Tuning
- Model Architecture Optimization
""" 