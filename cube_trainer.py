#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rubik's Cube Solver Trainer

This script implements the training process for the Rubik's cube solver using
Autodidactic Iteration (ADI) similar to DeepCube's approach. It focuses on
solving the 1LLL (Last Layer in one Look) case of the Rubik's cube.

The script supports:
1. Training with custom scrambles
2. Training with random scrambles from OLL+PLL algorithms
3. Testing the trained solver on new scrambles
"""

import numpy as np
import time
import random
import argparse
import os
from collections import deque
import tensorflow as tf

# Import our custom modules
from cube_neural_network import CubeNeuralNetwork, create_1lll_network
# Import the cube environment
from cube_env import CubeEnv, create_scrambled_cube

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

class CubeSolverTrainer:
    def __init__(self, model_path='cube_solver_model.h5'):
        """
        Initialize the Rubik's cube solver trainer

        Args:
            model_path: Path to save/load the model
        """
        self.model_path = model_path
        self.network = create_1lll_network()

        # Try to load existing model
        if os.path.exists(model_path):
            self.network.load_model(model_path)

        # Initialize experience replay buffer
        self.replay_buffer = deque(maxlen=10000)

        # Maximum number of steps to solve
        self.max_steps = 30

    def generate_training_data(self, num_samples=1000, use_oll_pll=True, custom_scramble=None):
        """
        Generate training data using Autodidactic Iteration (ADI)

        Args:
            num_samples: Number of scrambles to generate
            use_oll_pll: Whether to use OLL+PLL algorithms for scrambling
            custom_scramble: Custom scramble formula (overrides random generation)

        Returns:
            states, values, policies: Training data
        """
        states, values, policies = [], [], []

        for i in range(num_samples):
            if i % 100 == 0:
                print(f"Generating data sample {i}/{num_samples}")

            # Initialize environment
            env = CubeEnv()

            # Apply scramble
            if custom_scramble:
                env.custom_scramble(custom_scramble)
                scramble_type = "custom"
            elif use_oll_pll:
                env.reset(use_oll_pll=True)
                scramble_type = "OLL+PLL"
            else:
                # Random scramble with n random moves
                env.reset()
                n_moves = random.randint(5, 15)
                for _ in range(n_moves):
                    action = random.choice(env.action_space)
                    env.step(action)
                scramble_type = f"random-{n_moves}"

            # Get the initial state
            state = env.get_state()

            # For each possible action, evaluate the resulting state
            action_values = np.zeros(len(env.action_space))

            for action_idx, action in enumerate(env.action_space):
                # Create a copy of the environment
                test_env = CubeEnv(initial_state=env.cube)
                next_state, reward, done, _ = test_env.step(action)

                if done and reward > 0:  # Solved in one step
                    action_values[action_idx] = 1.0
                else:
                    # Use the network to estimate the value of the resulting state
                    value, _ = self.network.predict(next_state)
                    action_values[action_idx] = value

            # The best action has the highest value
            best_action_idx = np.argmax(action_values)

            # Create a policy distribution focused on the best action
            policy = np.zeros(len(env.action_space))
            policy[best_action_idx] = 1.0  # One-hot encoding for the best action

            # The value of the current state is the value of the best action
            value = action_values[best_action_idx]

            # Add to training data
            states.append(state)
            values.append([value])  # Value is a scalar
            policies.append(policy)

            # Add to replay buffer
            self.replay_buffer.append((state, value, policy))

            # Occasionally print an example
            if i % 500 == 0 and i > 0:
                print(f"\nScramble type: {scramble_type}")
                print(f"Best action: {env.action_space[best_action_idx]} (idx {best_action_idx})")
                print(f"State value: {value}")
                print(f"Action values: {action_values}")

        return np.array(states), np.array(values), np.array(policies)

    def train_network(self, epochs=10, batch_size=64, samples_per_epoch=1000, use_oll_pll=True, custom_scramble=None):
        """
        Train the network using ADI

        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            samples_per_epoch: Number of new samples to generate per epoch
            use_oll_pll: Whether to use OLL+PLL algorithms for scrambling
            custom_scramble: Custom scramble formula

        Returns:
            Training history
        """
        print(f"Starting training for {epochs} epochs")
        print(f"Generating {samples_per_epoch} samples per epoch")

        history = []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            start_time = time.time()

            # Generate new training data
            print("Generating training data...")
            states, values, policies = self.generate_training_data(
                num_samples=samples_per_epoch,
                use_oll_pll=use_oll_pll,
                custom_scramble=custom_scramble
            )

            # Mix with replay buffer for more stable training
            if len(self.replay_buffer) > 0:
                print(f"Adding {min(samples_per_epoch, len(self.replay_buffer))} samples from replay buffer")
                replay_samples = random.sample(self.replay_buffer, min(samples_per_epoch, len(self.replay_buffer)))
                replay_states, replay_values, replay_policies = zip(*replay_samples)

                states = np.vstack([states, np.array(replay_states)])
                values = np.vstack([values, np.array(replay_values).reshape(-1, 1)])
                policies = np.vstack([policies, np.array(replay_policies)])

            # Train the network
            print("Training network...")
            epoch_history = self.network.train(states, values, policies, epochs=1, batch_size=batch_size)
            history.append(epoch_history)

            # Save the model
            self.network.save_model(self.model_path)

            # Print epoch summary
            end_time = time.time()
            print(f"Epoch {epoch+1} completed in {end_time - start_time:.2f} seconds")

            # Evaluate on a few test cases
            self.evaluate_model(num_tests=5, use_oll_pll=use_oll_pll)

        return history

    def evaluate_model(self, num_tests=10, use_oll_pll=True, custom_scramble=None):
        """
        Evaluate the model on test scrambles

        Args:
            num_tests: Number of test cases
            use_oll_pll: Whether to use OLL+PLL algorithms for scrambling
            custom_scramble: Custom scramble formula

        Returns:
            success_rate: Percentage of successful solves
        """
        print(f"\nEvaluating model on {num_tests} test cases")

        successes = 0
        total_steps = 0

        for i in range(num_tests):
            env = CubeEnv()

            # Apply scramble
            if custom_scramble:
                env.custom_scramble(custom_scramble)
                print(f"Test {i+1}: Using custom scramble: {custom_scramble}")
            elif use_oll_pll:
                scramble = env.reset(use_oll_pll=True)
                print(f"Test {i+1}: Using OLL+PLL scramble")
            else:
                env.reset()
                n_moves = random.randint(5, 15)
                moves = []
                for _ in range(n_moves):
                    action = random.choice(env.action_space)
                    env.step(action)
                    moves.append(action)
                print(f"Test {i+1}: Using random {n_moves}-move scramble: {' '.join(moves)}")

            # Try to solve
            solution, is_solved, steps = self.solve_cube(env)

            if is_solved:
                successes += 1
                total_steps += steps
                print(f"  ✓ Solved in {steps} steps. Solution: {solution}")
            else:
                print(f"  ✗ Failed to solve. Attempted {steps} steps.")

        success_rate = successes / num_tests * 100
        avg_steps = total_steps / successes if successes > 0 else 0

        print(f"\nEvaluation results:")
        print(f"Success rate: {success_rate:.1f}% ({successes}/{num_tests})")
        if successes > 0:
            print(f"Average steps for successful solves: {avg_steps:.1f}")

        return success_rate

    def solve_cube(self, env, max_steps=None):
        """
        Attempt to solve a cube using the trained network

        Args:
            env: Cube environment
            max_steps: Maximum number of steps (defaults to self.max_steps)

        Returns:
            solution: List of moves applied
            is_solved: Whether the cube was solved
            steps: Number of steps taken
        """
        if max_steps is None:
            max_steps = self.max_steps

        solution = []
        is_solved = False

        # Original state
        original_env = CubeEnv(initial_state=env.cube)

        for step in range(max_steps):
            # Current state
            state = env.get_state()

            # Check if already solved
            if env.is_solved():
                is_solved = True
                break

            # Get action probabilities from the network
            _, policy = self.network.predict(state)

            # Choose the action with the highest probability
            action_idx = np.argmax(policy)
            action = env.action_space[action_idx]

            # Apply the action
            env.step(action)
            solution.append(action)

        return ' '.join(solution), env.is_solved(), len(solution)

    def solve_custom_scramble(self, scramble):
        """
        Solve a cube with a custom scramble

        Args:
            scramble: Scramble formula (e.g., "R U R' U'")

        Returns:
            Original cube state, solution sequence, and whether it was solved
        """
        env = CubeEnv()
        env.custom_scramble(scramble)

        print(f"Attempting to solve custom scramble: {scramble}")
        print("Initial state:")
        env.render()

        solution, is_solved, steps = self.solve_cube(env)

        print("\nFinal state:")
        env.render()

        if is_solved:
            print(f"Solved in {steps} steps!")
            print(f"Solution: {solution}")
        else:
            print(f"Failed to solve after {steps} steps.")
            print(f"Partial solution attempted: {solution}")

        return env, solution, is_solved

    def solve_random_scramble(self, use_oll_pll=True, n_random_moves=10):
        """
        Solve a cube with a random scramble

        Args:
            use_oll_pll: Whether to use OLL+PLL algorithms for scrambling
            n_random_moves: Number of random moves if not using OLL+PLL

        Returns:
            Original cube state, solution sequence, and whether it was solved
        """
        env = CubeEnv()

        if use_oll_pll:
            env.reset(use_oll_pll=True)
            scramble_type = "OLL+PLL"
        else:
            env.reset()
            for _ in range(n_random_moves):
                action = random.choice(env.action_space)
                env.step(action)
            scramble_type = f"{n_random_moves} random moves"

        print(f"Attempting to solve {scramble_type} scramble")
        print("Initial state:")
        env.render()

        solution, is_solved, steps = self.solve_cube(env)

        print("\nFinal state:")
        env.render()

        if is_solved:
            print(f"Solved in {steps} steps!")
            print(f"Solution: {solution}")
        else:
            print(f"Failed to solve after {steps} steps.")
            print(f"Partial solution attempted: {solution}")

        return env, solution, is_solved


def main():
    """Main function to parse arguments and run training/evaluation"""
    parser = argparse.ArgumentParser(description='Train and evaluate Rubik\'s cube solver')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'solve_custom', 'solve_random'],
                        help='Operation mode: train, eval, solve_custom, or solve_random')

    parser.add_argument('--model', type=str, default='cube_solver_model.h5',
                        help='Path to save/load the model')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')

    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of samples per epoch')

    parser.add_argument('--use_oll_pll', action='store_true',
                        help='Use OLL+PLL algorithms for scrambling')

    parser.add_argument('--custom_scramble', type=str, default=None,
                        help='Custom scramble formula (e.g., "R U R\' U\')')

    parser.add_argument('--num_tests', type=int, default=10,
                        help='Number of test cases for evaluation')

    args = parser.parse_args()

    # Initialize trainer
    trainer = CubeSolverTrainer(model_path=args.model)

    # Execute requested operation
    if args.mode == 'train':
        print("Starting training...")
        trainer.train_network(
            epochs=args.epochs,
            batch_size=args.batch_size,
            samples_per_epoch=args.samples,
            use_oll_pll=args.use_oll_pll,
            custom_scramble=args.custom_scramble
        )

    elif args.mode == 'eval':
        print("Evaluating model...")
        trainer.evaluate_model(
            num_tests=args.num_tests,
            use_oll_pll=args.use_oll_pll,
            custom_scramble=args.custom_scramble
        )

    elif args.mode == 'solve_custom':
        if not args.custom_scramble:
            print("Error: Custom scramble required for solve_custom mode")
            return

        print(f"Solving custom scramble: {args.custom_scramble}")
        trainer.solve_custom_scramble(args.custom_scramble)

    elif args.mode == 'solve_random':
        print("Solving random scramble...")
        trainer.solve_random_scramble(use_oll_pll=args.use_oll_pll)

    print("Operation completed!")


if __name__ == "__main__":
    main()