"""
Recurrent Neural Networks (RNN) and LSTM Tutorial
This file demonstrates the implementation of RNNs and LSTM networks using TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class RecurrentNeuralNetwork:
    """
    A Recurrent Neural Network implementation using TensorFlow/Keras.
    """
    def __init__(self, input_shape, num_classes, model_type='lstm'):
        """
        Initialize the RNN.
        
        Parameters:
        input_shape: Shape of input sequences (time_steps, features)
        num_classes: Number of output classes
        model_type: Type of RNN ('simple', 'lstm', or 'gru')
        """
        self.model = self._build_model(input_shape, num_classes, model_type)
    
    def _build_model(self, input_shape, num_classes, model_type):
        """Build the RNN architecture"""
        model = keras.Sequential()
        
        # Input layer
        model.add(keras.layers.Input(shape=input_shape))
        
        # RNN layers
        if model_type == 'simple':
            model.add(keras.layers.SimpleRNN(128, return_sequences=True))
            model.add(keras.layers.SimpleRNN(64))
        elif model_type == 'lstm':
            model.add(keras.layers.LSTM(128, return_sequences=True))
            model.add(keras.layers.LSTM(64))
        elif model_type == 'gru':
            model.add(keras.layers.GRU(128, return_sequences=True))
            model.add(keras.layers.GRU(64))
        
        # Dense layers
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        
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

def create_sequences(data, time_steps):
    """Create sequences for time series data"""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

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

def plot_predictions(y_true, y_pred, title='Predictions vs Actual'):
    """Plot predictions against actual values"""
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate synthetic time series data
    time = np.linspace(0, 100, 1000)
    data = np.sin(0.1 * time) + np.random.normal(0, 0.1, 1000)
    
    # Create sequences
    time_steps = 20
    X, y = create_sequences(data, time_steps)
    
    # Split data into train, validation, and test sets
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Reshape data for RNN input
    X_train = X_train.reshape(-1, time_steps, 1)
    X_val = X_val.reshape(-1, time_steps, 1)
    X_test = X_test.reshape(-1, time_steps, 1)
    
    # Create and train the model
    model = RecurrentNeuralNetwork(
        input_shape=(time_steps, 1),
        num_classes=1,
        model_type='lstm'
    )
    
    # Train the model
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,
        batch_size=32
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Plot predictions
    plot_predictions(y_test, predictions.flatten())

"""
Key Concepts Explained:

1. Recurrent Neural Networks:
   - Designed for sequential data
   - Maintain memory of previous inputs
   - Process data one step at a time

2. Types of RNNs:
   - Simple RNN: Basic recurrent architecture
   - LSTM: Long Short-Term Memory
   - GRU: Gated Recurrent Unit

3. LSTM Components:
   - Forget Gate: Decides what to forget
   - Input Gate: Decides what to store
   - Output Gate: Decides what to output
   - Cell State: Maintains long-term memory

4. Best Practices:
   - Proper sequence length
   - Appropriate number of units
   - Dropout for regularization
   - Batch normalization
   - Gradient clipping

Next Steps:
- Implementing attention mechanisms
- Sequence-to-sequence models
- Transformer architectures
- Advanced sequence modeling
""" 