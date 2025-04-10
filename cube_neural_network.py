#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Network for Rubik's Cube Solver

This script defines the neural network architecture for solving the Rubik's cube.
Based on the DeepCube approach but adapted for 1LLL (Last Layer in one Look).

The architecture uses a value and policy network similar to AlphaZero:
- Value head: Predicts how close the current state is to being solved
- Policy head: Predicts the probability distribution over possible actions
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

class CubeNeuralNetwork:
    def __init__(self, input_size=54, action_size=15, learning_rate=0.001):
        """
        Initialize the neural network for Rubik's cube solving

        Args:
            input_size: Size of the state representation (54 for a flattened cube)
            action_size: Number of possible actions (15 for U, D, F, L, R and their variants)
            learning_rate: Learning rate for the optimizer
        """
        self.input_size = input_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """
        Build the neural network model with value and policy heads

        Returns:
            A compiled Keras model
        """
        # Input layer
        input_layer = keras.Input(shape=(self.input_size,))

        # Convert to one-hot encoding if not already
        # For cube state where each position has 6 possible colors
        reshaped_input = tf.reshape(input_layer, [-1, 54, 1])
        one_hot_input = tf.one_hot(tf.cast(reshaped_input, tf.int32), depth=6)
        flattened_input = tf.reshape(one_hot_input, [-1, 54 * 6])

        # Shared network layers
        x = keras.layers.Dense(1024, activation='elu')(flattened_input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(512, activation='elu')(x)
        x = keras.layers.BatchNormalization()(x)

        # Value head - predicts how close to solved
        value_head = keras.layers.Dense(256, activation='elu')(x)
        value_head = keras.layers.BatchNormalization()(value_head)
        value_output = keras.layers.Dense(1, name='value_output')(value_head)

        # Policy head - predicts action probabilities
        policy_head = keras.layers.Dense(256, activation='elu')(x)
        policy_head = keras.layers.BatchNormalization()(policy_head)
        policy_output = keras.layers.Dense(self.action_size, activation='softmax', name='policy_output')(policy_head)

        # Create model with two outputs
        model = keras.Model(inputs=input_layer, outputs=[value_output, policy_output])

        # Compile model with appropriate losses
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss={
                'value_output': 'mean_squared_error',
                'policy_output': 'categorical_crossentropy'
            }
        )

        # Print model summary
        model.summary()

        return model

    def predict(self, state):
        """
        Predict value and policy for a given state

        Args:
            state: Current cube state representation

        Returns:
            value: Estimated value of the state
            policy: Action probability distribution
        """
        value, policy = self.model.predict(np.array([state]), verbose=0)
        return value[0][0], policy[0]

    def train(self, states, values, policies, epochs=10, batch_size=64):
        """
        Train the model on collected data

        Args:
            states: Array of cube states
            values: Target values for each state
            policies: Target policies for each state
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Training history
        """
        history = self.model.fit(
            np.array(states),
            {'value_output': np.array(values), 'policy_output': np.array(policies)},
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history

    def save_model(self, filepath):
        """Save the model to a file"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load the model from a file"""
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            return True
        return False


def create_1lll_network():
    """
    Create a neural network specifically designed for 1LLL cube solving

    Returns:
        A CubeNeuralNetwork instance
    """
    # Default parameters suitable for 1LLL
    return CubeNeuralNetwork(input_size=54, action_size=15, learning_rate=0.001)


if __name__ == "__main__":
    # Simple test to ensure the model compiles correctly
    network = create_1lll_network()

    # Generate a random cube state for testing
    test_state = np.random.randint(0, 6, size=54)
    value, policy = network.predict(test_state)

    print(f"Test prediction - Value: {value}, Top actions: {np.argsort(policy)[-3:][::-1]}")