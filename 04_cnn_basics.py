"""
Convolutional Neural Networks (CNN) Tutorial
This file demonstrates the implementation of CNNs using TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ConvolutionalNeuralNetwork:
    """
    A Convolutional Neural Network implementation using TensorFlow/Keras.
    """
    def __init__(self, input_shape, num_classes):
        """
        Initialize the CNN.
        
        Parameters:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
        """
        self.model = self._build_model(input_shape, num_classes)
    
    def _build_model(self, input_shape, num_classes):
        """Build the CNN architecture"""
        model = keras.Sequential([
            # First Convolutional Block
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Second Convolutional Block
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Third Convolutional Block
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        """Train the model"""
        # Data augmentation
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            steps_per_epoch=len(X_train) // batch_size,
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

def visualize_predictions(model, X_test, y_test, num_samples=5):
    """Visualize model predictions"""
    predictions = model.predict(X_test[:num_samples])
    
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'Pred: {np.argmax(predictions[i])}\nTrue: {y_test[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load and preprocess the MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize and reshape the data
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Add channel dimension
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Split training data into training and validation sets
    val_size = 5000
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    # Create and train the model
    model = ConvolutionalNeuralNetwork(
        input_shape=(28, 28, 1),
        num_classes=10
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
    
    # Visualize predictions
    visualize_predictions(model, X_test, y_test)

"""
Key Concepts Explained:

1. Convolutional Neural Networks:
   - Specialized for processing grid-like data (images)
   - Use convolutional layers to extract features
   - Maintain spatial relationships in data

2. Key Components:
   - Convolutional Layers: Extract features using filters
   - Pooling Layers: Reduce spatial dimensions
   - Batch Normalization: Normalize layer inputs
   - Dropout: Prevent overfitting

3. CNN Architecture:
   - Multiple convolutional blocks
   - Each block contains:
     * Convolutional layers
     * Batch normalization
     * Activation functions
     * Pooling
     * Dropout

4. Best Practices:
   - Data augmentation
   - Batch normalization
   - Dropout for regularization
   - Proper padding
   - Appropriate filter sizes

Next Steps:
- Implementing more complex CNN architectures
- Transfer learning with pre-trained models
- Object detection and segmentation
- Advanced data augmentation techniques
""" 