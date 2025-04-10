#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Network for Rubik's Cube 1LLL Solver

This module implements multiple neural network architectures inspired by DeepCube:
- ResNet architecture with residual connections
- LSTM architecture for sequence modeling
- Combined value and policy network for reinforcement learning
- One-hot encoded state representation
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


class ResidualBlock(keras.layers.Layer):
    """
    Residual block as used in DeepCube architecture
    """

    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.dense1 = keras.layers.Dense(filters, activation=None)
        self.batch_norm1 = keras.layers.BatchNormalization()
        self.dense2 = keras.layers.Dense(filters, activation=None)
        self.batch_norm2 = keras.layers.BatchNormalization()
        self.activation = keras.layers.Activation('relu')

    def call(self, inputs, training=False, *args, **kwargs):
        x = self.dense1(inputs)
        x = self.batch_norm1(x, training=training)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.batch_norm2(x, training=training)
        x = x + inputs  # Skip connection
        return self.activation(x)


class CubeNeuralNetwork:
    """
    Neural network for the Rubik's Cube 1LLL solver
    Supports multiple architectures including ResNet and LSTM
    """

    def __init__(self,
                 input_shape=(54, 6),
                 action_size=15,
                 learning_rate=0.001,
                 architecture='resnet'):
        """
        Initialize the neural network

        Args:
            input_shape: Shape of input state (54 stickers, 6 colors one-hot encoded)
            action_size: Number of possible actions (15 for 5 faces x 3 rotations)
            learning_rate: Learning rate for optimizer
            architecture: 'resnet' or 'lstm'
        """
        self.input_shape = input_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.model = self._build_model()

    def _build_model(self):
        """
        Build the neural network model

        Returns:
            Compiled Keras model
        """
        if self.architecture == 'resnet':
            return self._build_resnet_model()
        elif self.architecture == 'lstm':
            return self._build_lstm_model()
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def _build_resnet_model(self):
        """
        Build a ResNet model with residual connections

        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = keras.layers.Input(shape=self.input_shape)

        # Flatten the input
        x = keras.layers.Flatten()(inputs)

        # Initial dense layer
        x = keras.layers.Dense(1024, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)

        # Residual blocks (8 blocks as in DeepCube, but with fewer filters for 1LLL)
        num_residual_blocks = 8
        for _ in range(num_residual_blocks):
            x = ResidualBlock(1024)(x)

        # Value head - estimates how "good" the current state is
        value_head = keras.layers.Dense(512, activation='relu')(x)
        value_head = keras.layers.BatchNormalization()(value_head)
        value_head = keras.layers.Dense(256, activation='relu')(value_head)
        value_output = keras.layers.Dense(1, name='value_output')(value_head)

        # Policy head - outputs probability distribution over actions
        policy_head = keras.layers.Dense(512, activation='relu')(x)
        policy_head = keras.layers.BatchNormalization()(policy_head)
        policy_head = keras.layers.Dense(256, activation='relu')(policy_head)
        policy_output = keras.layers.Dense(self.action_size,
                                           activation='softmax',
                                           name='policy_output')(policy_head)

        # Create model with two outputs
        model = keras.Model(inputs=inputs,
                            outputs=[value_output, policy_output])

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss={
                          'value_output': 'mean_squared_error',
                          'policy_output': 'categorical_crossentropy'
                      })

        # Print model summary
        model.summary()

        return model

    def _build_lstm_model(self):
        """
        Build an LSTM model for sequence modeling

        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = keras.layers.Input(shape=self.input_shape)

        # Reshape for LSTM (treating each sticker as a sequence element)
        reshaped = keras.layers.Reshape((54, 6))(inputs)

        # LSTM layers
        x = keras.layers.LSTM(512, return_sequences=True)(reshaped)
        x = keras.layers.LSTM(512)(x)

        # Dense layers
        x = keras.layers.Dense(1024, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.3)(x)

        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)

        # Value head
        value_head = keras.layers.Dense(256, activation='relu')(x)
        value_head = keras.layers.BatchNormalization()(value_head)
        value_output = keras.layers.Dense(1, name='value_output')(value_head)

        # Policy head
        policy_head = keras.layers.Dense(256, activation='relu')(x)
        policy_head = keras.layers.BatchNormalization()(policy_head)
        policy_output = keras.layers.Dense(self.action_size,
                                           activation='softmax',
                                           name='policy_output')(policy_head)

        # Create model with two outputs
        model = keras.Model(inputs=inputs,
                            outputs=[value_output, policy_output])

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer,
                      loss={
                          'value_output': 'mean_squared_error',
                          'policy_output': 'categorical_crossentropy'
                      })

        # Print model summary
        model.summary()

        return model

    def predict(self, state):
        """
        Make a prediction for a single state

        Args:
            state: One-hot encoded state representation

        Returns:
            value, policy: Predicted state value and action probabilities
        """
        # Ensure state is properly shaped
        if len(state.shape) == 2:  # Single state
            state = np.expand_dims(state, axis=0)

        # Make prediction
        value, policy = self.model.predict(state, verbose=0)

        # Return value scalar and policy vector
        return value[0][0], policy[0]

    def train(self, states, values, policies, batch_size=64, epochs=1):
        """
        Train the model on a batch of data

        Args:
            states: Batch of states (one-hot encoded)
            values: Target values for states
            policies: Target policies (action probabilities)
            batch_size: Batch size for training
            epochs: Number of training epochs

        Returns:
            History object from training
        """
        history = self.model.fit(states, {
            'value_output': values,
            'policy_output': policies
        },
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1)
        return history

    def save_model(self, filepath):
        """Save model weights to file"""
        self.model.save_weights(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model weights from file"""
        if os.path.exists(filepath):
            try:
                self.model.load_weights(filepath)
                print(f"Model loaded from {filepath}")
                return True
            except:
                print(f"Failed to load model from {filepath}")
                return False
        else:
            print(f"Model file {filepath} not found")
            return False
