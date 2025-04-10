#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training System for Rubik's Cube 1LLL Solver with Curriculum Learning

This module implements:
1. Autodidactic Iteration (ADI) - DeepCube's self-play training algorithm
2. Progressive curriculum learning from easy to hard scrambles 
3. Experience replay for stable learning
4. Integration with MCTS for solving
5. Custom scramble mode for user-defined patterns
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import random
import matplotlib.pyplot as plt
import os
from collections import deque
import argparse
from tqdm import tqdm

# Import custom modules
from cube_env import CubeEnv, create_scrambled_cube
from cube_neural_network import CubeNeuralNetwork
from mcts import MCTS

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Constants
DEFAULT_MODEL_PATH = '1lll_solver_model.h5'
EXPERIENCE_BUFFER_SIZE = 100000
BATCH_SIZE = 128
GAMMA = 0.997  # Discount factor for future rewards
MAX_STEPS = 30  # Maximum steps for solving

# Curriculum learning parameters
SUCCESS_THRESHOLD = 0.7  # Success rate required to advance level
MIN_EVALUATIONS = 20    # Minimum evaluations before advancing

class CubeTrainer:
    """
    Training system for 1LLL Rubik's Cube solver
    Uses DeepCube's Autodidactic Iteration with curriculum learning
    """

    def __init__(self, model_path=DEFAULT_MODEL_PATH, architecture='resnet'):
        """
        Initialize the trainer

        Args:
            model_path: Path to save/load model
            architecture: 'resnet' or 'lstm'
        """
        self.model_path = model_path
        self.architecture = architecture

        # Initialize neural network
        print(f"Initializing neural network with {architecture} architecture")
        self.network = CubeNeuralNetwork(
            input_shape=(54, 6),
            action_size=15,
            learning_rate=0.001,
            architecture=architecture
        )

        # Try to load existing model
        if os.path.exists(model_path):
            self.network.load_model(model_path)

        # Initialize MCTS solver
        self.mcts = MCTS(neural_network=self.network)

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=EXPERIENCE_BUFFER_SIZE)

        # Curriculum learning state
        self.curriculum_level = 1  # Start at easiest level
        self.success_history = deque(maxlen=MIN_EVALUATIONS)

        # Training metrics
        self.metrics = {
            'loss': [],
            'value_loss': [],
            'policy_loss': [],
            'success_rate': [],
            'avg_steps': [],
            'curriculum_level': []
        }

        print(f"Trainer initialized with curriculum level {self.curriculum_level}")

    def autodidactic_iteration(self, num_samples, curriculum_level=None):
        """
        Implementation of Autodidactic Iteration (ADI) algorithm from DeepCube

        Args:
            num_samples: Number of training samples to generate
            curriculum_level: Difficulty level (1-5)

        Returns:
            states, values, policies: Training data
        """
        if curriculum_level is None:
            curriculum_level = self.curriculum_level

        print(f"Running Autodidactic Iteration for {num_samples} samples at level {curriculum_level}")

        states = []
        values = []
        policies = []

        for i in tqdm(range(num_samples)):
            # Create scrambled cube at appropriate difficulty
            env = create_scrambled_cube(curriculum_level=curriculum_level)

            # Get current state (one-hot encoded)
            state = env.get_one_hot_state()

            # For each possible action, evaluate resulting state
            action_values = np.zeros(len(env.action_space))

            for action in range(len(env.action_space)):
                # Copy environment
                test_env = env.copy()

                # Take action
                _, reward, done, _ = test_env.step(action)

                # If action solves the cube, it has maximum value
                if done and test_env.is_solved():
                    action_values[action] = 1.0
                else:
                    # Otherwise, use neural network to evaluate resulting state
                    next_state = test_env.get_one_hot_state()
                    value, _ = self.network.predict(next_state)

                    # Value is immediate reward plus discounted future value
                    action_values[action] = reward + GAMMA * value

            # Best action has highest value
            best_action = np.argmax(action_values)
            best_value = action_values[best_action]

            # Create policy (one-hot encoding for best action)
            policy = np.zeros(len(env.action_space))
            policy[best_action] = 1.0

            # Add to training data
            states.append(state)
            values.append([best_value])
            policies.append(policy)

            # Add experience to replay buffer
            self.replay_buffer.append((state, best_value, policy))

            # Occasionally print example
            if i > 0 and i % 500 == 0:
                print(f"\nSample {i}:")
                print(f"Best action: {env.action_space[best_action]}")
                print(f"Action values: {action_values}")
                print(f"Best value: {best_value:.4f}")

        return np.array(states), np.array(values), np.array(policies)

    def train(self, iterations=50, samples_per_iteration=500, eval_interval=5):
        """
        Train the model with curriculum learning

        Args:
            iterations: Number of training iterations
            samples_per_iteration: ADI samples per iteration
            eval_interval: How often to evaluate and adjust curriculum

        Returns:
            Training metrics
        """
        print(f"Starting training for {iterations} iterations")
        print(f"Neural network architecture: {self.architecture}")
        print(f"Initial curriculum level: {self.curriculum_level}")

        # Main training loop
        for iteration in range(iterations):
            start_time = time.time()
            print(f"\n--- Iteration {iteration+1}/{iterations} ---")
            print(f"Current curriculum level: {self.curriculum_level}")

            # Generate training data using ADI
            states, values, policies = self.autodidactic_iteration(
                num_samples=samples_per_iteration,
                curriculum_level=self.curriculum_level
            )

            # Mix with replay buffer if available
            if len(self.replay_buffer) > BATCH_SIZE:
                num_replay = min(samples_per_iteration // 2, len(self.replay_buffer))
                print(f"Mixing with {num_replay} samples from experience replay")

                # Sample from replay buffer
                replay_samples = random.sample(self.replay_buffer, num_replay)
                replay_states, replay_values, replay_policies = zip(*replay_samples)

                # Convert to arrays
                replay_states = np.array(replay_states)
                replay_values = np.array(replay_values).reshape(-1, 1)
                replay_policies = np.array(replay_policies)

                # Combine with new samples
                states = np.concatenate([states, replay_states])
                values = np.concatenate([values, replay_values])
                policies = np.concatenate([policies, replay_policies])

            # Train neural network
            print("Training neural network...")
            history = self.network.train(
                states=states,
                values=values,
                policies=policies,
                batch_size=BATCH_SIZE,
                epochs=1
            )

            # Record metrics
            self.metrics['loss'].append(history.history['loss'][0])
            if 'value_output_loss' in history.history:
                self.metrics['value_loss'].append(history.history['value_output_loss'][0])
                self.metrics['policy_loss'].append(history.history['policy_output_loss'][0])

            # Save model
            self.network.save_model(self.model_path)

            # Evaluate and adjust curriculum
            if (iteration + 1) % eval_interval == 0 or iteration == 0:
                success_rate, avg_steps = self.evaluate(num_tests=5)

                # Record metrics
                self.metrics['success_rate'].append(success_rate)
                self.metrics['avg_steps'].append(avg_steps)
                self.metrics['curriculum_level'].append(self.curriculum_level)

                # Update curriculum level
                self._update_curriculum(success_rate)

            # Print iteration summary
            end_time = time.time()
            print(f"Iteration {iteration+1} completed in {end_time - start_time:.2f} seconds")

        # Final evaluation
        print("\n--- Final Evaluation ---")
        for level in range(1, 6):
            self.evaluate(num_tests=10, curriculum_level=level)

        # Plot training progress
        self._plot_training_progress()

        return self.metrics

    def _update_curriculum(self, success_rate):
        """
        Update curriculum level based on performance

        Args:
            success_rate: Success rate (0-100%)
        """
        # Add to success history
        self.success_history.append(success_rate)

        # Only consider advancement if we have enough evaluations
        if len(self.success_history) >= MIN_EVALUATIONS:
            # Calculate average success rate
            avg_success = sum(self.success_history) / len(self.success_history)

            if avg_success >= SUCCESS_THRESHOLD and self.curriculum_level < 5:
                # Advance to next level
                self.curriculum_level += 1
                print(f"\n*** ADVANCING TO CURRICULUM LEVEL {self.curriculum_level} ***")

                # Reset success history
                self.success_history.clear()
            elif avg_success < 0.3 and self.curriculum_level > 1:
                # If performing poorly, go back to easier level
                self.curriculum_level -= 1
                print(f"\n*** RETURNING TO CURRICULUM LEVEL {self.curriculum_level} ***")

                # Reset success history
                self.success_history.clear()

        print(f"Current curriculum level: {self.curriculum_level}")
        if self.success_history:
            avg = sum(self.success_history) / len(self.success_history)
            print(f"Average success rate: {avg:.1f}% ({len(self.success_history)}/{MIN_EVALUATIONS} evaluations)")

    def evaluate(self, num_tests=5, curriculum_level=None, use_mcts=True):
        """
        Evaluate solver on test cases

        Args:
            num_tests: Number of test cases
            curriculum_level: Difficulty level (defaults to current level)
            use_mcts: Whether to use MCTS for solving

        Returns:
            success_rate, avg_steps: Evaluation metrics
        """
        if curriculum_level is None:
            curriculum_level = self.curriculum_level

        print(f"Evaluating on {num_tests} tests at level {curriculum_level}")
        print(f"Solver: {'MCTS' if use_mcts else 'Greedy Policy'}")

        successes = 0
        total_steps = 0

        for i in range(num_tests):
            # Create scrambled cube
            env = create_scrambled_cube(curriculum_level=curriculum_level)

            # Get the scramble formula
            scramble = env.generate_curriculum_scramble()
            print(f"Test {i+1} scramble: {scramble}")

            # Try to solve
            solution, is_solved, steps = self.solve(env, use_mcts=use_mcts)

            if is_solved:
                successes += 1
                total_steps += steps
                print(f"  ✓ Test {i+1}: Solved in {steps} steps")
                print(f"  Solution: {' '.join(solution)}")
            else:
                print(f"  ✗ Test {i+1}: Failed to solve")

        # Calculate metrics
        success_rate = (successes / num_tests) * 100
        avg_steps = total_steps / successes if successes > 0 else 0

        print(f"\nEvaluation results at level {curriculum_level}:")
        print(f"Success rate: {success_rate:.1f}% ({successes}/{num_tests})")
        if successes > 0:
            print(f"Average steps for successful solves: {avg_steps:.1f}")

        return success_rate, avg_steps

    def solve(self, env, max_steps=MAX_STEPS, use_mcts=True):
        """
        Attempt to solve a cube

        Args:
            env: CubeEnv environment
            max_steps: Maximum steps to attempt
            use_mcts: Whether to use MCTS

        Returns:
            solution: List of moves
            is_solved: Whether solved successfully
            steps: Number of steps taken
        """
        solution = []
        steps = 0

        # Make a copy of the environment
        solve_env = env.copy()

        while steps < max_steps:
            # Check if already solved
            if solve_env.is_solved():
                return solution, True, steps

            if use_mcts:
                # Use MCTS to find best action
                # Limit search time to 5 seconds per move
                action = self.mcts.search(solve_env, time_limit=5)
            else:
                # Use greedy policy (no search)
                state = solve_env.get_one_hot_state()
                _, policy = self.network.predict(state)
                action = np.argmax(policy)

            # Take action
            _, _, done, _ = solve_env.step(action)
            solution.append(solve_env.action_space[action])
            steps += 1

            # Check if solved
            if done and solve_env.is_solved():
                return solution, True, steps

        # Failed to solve within max steps
        return solution, solve_env.is_solved(), steps

    def solve_custom(self, scramble, use_mcts=True):
        """
        Solve a cube with a custom scramble formula

        Args:
            scramble: Custom scramble formula (e.g., "R U R' U'")
            use_mcts: Whether to use MCTS

        Returns:
            solution, is_solved, steps
        """
        # Create a solved cube
        env = CubeEnv()

        # Apply custom scramble
        if env.custom_scramble(scramble):
            print("Applied scramble successfully")
        else:
            print("Failed to apply scramble - check formula")
            return None, False, 0

        # Show scrambled cube
        print("\nScrambled cube:")
        env.render()
        print(f"Applied scramble: {scramble}")

        # Try to solve
        print("\nAttempting to solve...")
        start_time = time.time()
        solution, is_solved, steps = self.solve(env, use_mcts=use_mcts)
        end_time = time.time()

        # Show final state
        print("\nFinal state:")
        env.render()

        if is_solved:
            print(f"✓ Solved in {steps} steps! (took {end_time - start_time:.2f} seconds)")
            print(f"Solution: {' '.join(solution)}")
        else:
            print(f"✗ Failed to solve after {steps} steps")
            print(f"Partial solution: {' '.join(solution)}")

        return solution, is_solved, steps

    def demo_solve(self, curriculum_level=None, use_mcts=True):
        """
        Demonstrate solving a cube

        Args:
            curriculum_level: Difficulty level
            use_mcts: Whether to use MCTS
        """
        if curriculum_level is None:
            curriculum_level = self.curriculum_level

        print(f"Demonstrating solve at level {curriculum_level}")
        print(f"Using {'MCTS' if use_mcts else 'Greedy Policy'}")

        # Create scrambled cube
        env = create_scrambled_cube(curriculum_level=curriculum_level)

        # Get the scramble formula
        scramble = env.generate_curriculum_scramble()

        print("Initial state:")
        env.render()
        print(f"Scramble: {scramble}")

        # Try to solve
        print("\nSolving step by step...")
        start_time = time.time()

        solve_env = env.copy()
        solution = []
        steps = 0

        while steps < MAX_STEPS:
            # Check if already solved
            if solve_env.is_solved():
                break

            if use_mcts:
                # Use MCTS to find best action
                action = self.mcts.search(solve_env, time_limit=3)
            else:
                # Use greedy policy
                state = solve_env.get_one_hot_state()
                _, policy = self.network.predict(state)
                action = np.argmax(policy)

            # Take action
            move = solve_env.action_space[action]
            _, _, done, _ = solve_env.step(action)
            solution.append(move)
            steps += 1

            # Show each step
            print(f"Step {steps}: {move}")
            if steps % 3 == 0 or done:  # Show cube state every 3 steps
                solve_env.render()

            if done and solve_env.is_solved():
                break

        end_time = time.time()

        print("\nFinal state:")
        solve_env.render()

        if solve_env.is_solved():
            print(f"\n✓ Solved in {steps} steps! (took {end_time - start_time:.2f} seconds)")
            print(f"Complete solution: {' '.join(solution)}")
        else:
            print(f"\n✗ Failed to solve after {steps} steps")
            print(f"Partial solution: {' '.join(solution)}")

        return solution, solve_env.is_solved(), steps

    def _plot_training_progress(self):
        """Plot training metrics"""
        plt.figure(figsize=(15, 12))

        # Loss plot
        plt.subplot(3, 1, 1)
        plt.plot(self.metrics['loss'], 'r-', label='Total Loss')
        if self.metrics['value_loss']:
            plt.plot(self.metrics['value_loss'], 'b-', label='Value Loss')
            plt.plot(self.metrics['policy_loss'], 'g-', label='Policy Loss')
        plt.title('Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Success rate plot
        plt.subplot(3, 1, 2)
        eval_indices = range(len(self.metrics['success_rate']))
        plt.plot(eval_indices, self.metrics['success_rate'], 'b-')
        plt.xlabel('Evaluation')
        plt.ylabel('Success Rate (%)')
        plt.title('Solver Success Rate')
        plt.grid(True)

        # Curriculum level and solution steps
        plt.subplot(3, 1, 3)
        plt.plot(eval_indices, self.metrics['curriculum_level'], 'r-', label='Curriculum Level')
        plt.plot(eval_indices, self.metrics['avg_steps'], 'g-', label='Avg. Solution Steps')
        plt.xlabel('Evaluation')
        plt.ylabel('Value')
        plt.title('Curriculum Level and Solution Steps')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_progress.png')
        print("Training progress plot saved as 'training_progress.png'")

def run_interactive_mode():
    """Run an interactive session with the cube solver"""
    # Create trainer with default model
    trainer = CubeTrainer()

    print("\n=== 1LLL Rubik's Cube Solver Interactive Mode ===\n")
    print("Options:")
    print("1. Solve with custom scramble")
    print("2. Demonstrate with random scramble")
    print("3. Train the model")
    print("4. Evaluate model performance")
    print("5. Quit")

    while True:
        try:
            choice = input("\nEnter your choice (1-5): ")

            if choice == '1':
                # Custom scramble
                scramble = input("Enter your scramble formula (e.g., 'R U R\' U\''): ")
                use_mcts = input("Use MCTS for better results? (y/n, slower but better): ").lower().startswith('y')
                trainer.solve_custom(scramble, use_mcts=use_mcts)

            elif choice == '2':
                # Demo solve
                level = input("Enter difficulty level (1-5, default=current): ")
                level = int(level) if level.isdigit() and 1 <= int(level) <= 5 else None
                use_mcts = input("Use MCTS for better results? (y/n, slower but better): ").lower().startswith('y')
                trainer.demo_solve(curriculum_level=level, use_mcts=use_mcts)

            elif choice == '3':
                # Train
                iterations = input("Enter number of training iterations (default=20): ")
                iterations = int(iterations) if iterations.isdigit() else 20
                samples = input("Enter samples per iteration (default=200): ")
                samples = int(samples) if samples.isdigit() else 200
                print(f"\nStarting training with {iterations} iterations, {samples} samples each...")
                trainer.train(iterations=iterations, samples_per_iteration=samples)

            elif choice == '4':
                # Evaluate
                level = input("Enter difficulty level to evaluate (1-5, default=current): ")
                level = int(level) if level.isdigit() and 1 <= int(level) <= 5 else None
                num_tests = 5  # Fixed number of test cases
                use_mcts = input("Use MCTS for evaluation? (y/n, slower but better): ").lower().startswith('y')
                trainer.evaluate(num_tests=num_tests, curriculum_level=level, use_mcts=use_mcts)

            elif choice == '5':
                # Quit
                print("\nExiting. Goodbye!")
                break

            else:
                print("Invalid choice. Please enter a number from 1 to 5.")

        except Exception as e:
            print(f"Error: {e}")
            print("Let's try again.")

def main():
    """Main function - parse arguments and run trainer"""
    parser = argparse.ArgumentParser(description='Train 1LLL Rubik\'s Cube Solver')

    parser.add_argument('--mode', type=str, default='interactive',
                      choices=['train', 'eval', 'demo', 'custom', 'interactive'],
                      help='Operation mode')

    parser.add_argument('--architecture', type=str, default='resnet',
                      choices=['resnet', 'lstm'],
                      help='Neural network architecture')

    parser.add_argument('--iterations', type=int, default=50,
                      help='Number of training iterations')

    parser.add_argument('--samples', type=int, default=500,
                      help='Samples per iteration')

    parser.add_argument('--level', type=int, default=None,
                      help='Curriculum level (1-5) for evaluation/demo')

    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                      help='Model file path')

    parser.add_argument('--mcts', action='store_true',
                      help='Use MCTS for evaluation/demo (slower but better)')

    parser.add_argument('--scramble', type=str, default=None,
                      help='Custom scramble formula (e.g., "R U R\' U\'") for custom mode')

    args = parser.parse_args()

    # Run in interactive mode if selected
    if args.mode == 'interactive':
        run_interactive_mode()
        return

    # Create trainer
    trainer = CubeTrainer(
        model_path=args.model,
        architecture=args.architecture
    )

    # Run in specified mode
    if args.mode == 'train':
        print(f"Training with {args.architecture} architecture for {args.iterations} iterations")
        trainer.train(
            iterations=args.iterations,
            samples_per_iteration=args.samples
        )

    elif args.mode == 'eval':
        level = args.level if args.level is not None else trainer.curriculum_level
        print(f"Evaluating at curriculum level {level}")
        trainer.evaluate(
            num_tests=5,
            curriculum_level=level,
            use_mcts=args.mcts
        )

    elif args.mode == 'demo':
        level = args.level if args.level is not None else trainer.curriculum_level
        print(f"Demonstrating solve at curriculum level {level}")
        trainer.demo_solve(
            curriculum_level=level,
            use_mcts=args.mcts
        )

    elif args.mode == 'custom':
        # If scramble is not provided via command line, ask for it
        scramble = args.scramble
        if not scramble:
            scramble = input("Enter custom scramble formula (e.g., 'R U R\\' U\\'): ")

        print(f"Solving cube with custom scramble: {scramble}")
        trainer.solve_custom(scramble, use_mcts=args.mcts)

if __name__ == "__main__":
    main()