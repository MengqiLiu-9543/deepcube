#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cube Environment Module (CubeEnv):
- Based on cube_rotation.py RubikCube class
- Provides environment interface for Reinforcement Learning
- Supports state representation, action execution, and reward computation
- Includes formula testing functionality and curriculum-based scramble generation
"""

import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# Import the original cube code
from cube_rotation import RubikCube, draw_cube, rotate_face, rotate_face_ccw, rotate_face_180
# Import the OLL+PLL generator with curriculum support
from random_cube_generator import generate_scramble_by_level, generate_oll_pll


class CubeEnv:
    """
    Cube Environment based on RubikCube class
    Suitable for Reinforcement Learning training with curriculum learning
    """

    def __init__(self, initial_state=None):
        # Initialize the cube
        if initial_state is None:
            self.cube = RubikCube()
        else:
            self.cube = copy.deepcopy(initial_state)

        # Save initial state
        self.initial_state = copy.deepcopy(self.cube)
        # Save solved state (for reward calculation and completion check)
        self.solved_state = RubikCube()

        # Define action space (excluding B moves)
        self.action_space = [
            "U",
            "D",
            "F",
            "L",
            "R",  # Clockwise rotations
            "U'",
            "D'",
            "F'",
            "L'",
            "R'",  # Counter-clockwise rotations
            "U2",
            "D2",
            "F2",
            "L2",
            "R2"  # 180-degree rotations
        ]

        # Color mapping for state representation
        self.color_map = {
            'W': 0,  # White (Up face)
            'Y': 1,  # Yellow (Down face)
            'R': 2,  # Red (Front face)
            'O': 3,  # Orange (Back face)
            'G': 4,  # Green (Left face)
            'B': 5  # Blue (Right face)
        }

        # Track current step count
        self.current_step = 0
        # Maximum steps limit
        self.max_steps = 30

        # Current curriculum level (default: 1)
        self.curriculum_level = 1

    def reset(self, scramble=None, use_oll_pll=False, curriculum_level=None):
        """
        Reset the environment

        Args:
            scramble: Optional scramble formula string, e.g., "R U F' L"
            use_oll_pll: If True, use generate_oll_pll() to create a scramble
            curriculum_level: If provided, use this level for scramble generation

        Returns:
            state: The state after reset
        """
        # Reset to initial state
        self.cube = RubikCube()

        # Update curriculum level if provided
        if curriculum_level is not None:
            self.curriculum_level = curriculum_level

        # Generate a scramble if requested
        if use_oll_pll:
            scramble = generate_oll_pll()
            print(f"Generated OLL+PLL scramble: {scramble}")
        elif curriculum_level is not None:
            # Use curriculum-based scramble generation
            scramble = generate_scramble_by_level(self.curriculum_level)
            print(
                f"Generated Level {self.curriculum_level} scramble: {scramble}"
            )

        # If a scramble formula is provided, execute it
        if scramble:
            self.execute_formula(scramble)

        # Reset step counter
        self.current_step = 0

        # Return current state
        return self.get_state()

    def step(self, action):
        """
        Execute an action

        Args:
            action: Action to take, can be an integer index or action string

        Returns:
            next_state: The state after taking the action
            reward: The reward value
            done: Whether the episode is complete
            info: Additional information
        """
        # Handle integer index actions
        if isinstance(action, int):
            if action < 0 or action >= len(self.action_space):
                raise ValueError(
                    f"Action index {action} out of range 0-{len(self.action_space)-1}"
                )
            action_str = self.action_space[action]
        else:
            action_str = action
            if action_str not in self.action_space:
                raise ValueError(f"Unknown action: {action_str}")

        # Save pre-action state
        old_state = copy.deepcopy(self.cube)

        # Execute the action
        self.cube.move(action_str)
        self.current_step += 1

        # Calculate completion state and reward
        done = self.is_solved() or self.current_step >= self.max_steps

        # Reward design:
        # - Completion: +10 (higher reward for successful completion)
        # - Each step: -0.1 (small penalty for each step to encourage efficiency)
        # - Curriculum bonus: +5 for solving level 5, +4 for level 4, etc.
        if self.is_solved():
            # Base reward plus curriculum bonus
            reward = 10.0 + (self.curriculum_level / 2.0)
        else:
            # Small penalty for each step
            reward = -0.1

            # If max steps reached without solving, larger penalty
            if self.current_step >= self.max_steps:
                reward -= 1.0

        # Return next state, reward, completion status, and info
        return self.get_state(), reward, done, {
            "action": action_str,
            "steps": self.current_step,
            "solved": self.is_solved(),
            "curriculum_level": self.curriculum_level
        }

    def get_state(self):
        """
        Get the current cube state

        Returns:
            state: Array representing the cube state
        """
        # Use 1D array to represent all face colors
        state = []

        # Traverse all faces in a specific order
        for face in ['U', 'D', 'F', 'B', 'L', 'R']:
            for i in range(3):
                for j in range(3):
                    color = self.cube.faces[face][i][j]
                    state.append(self.color_map[color])

        return np.array(state)

    def get_one_hot_state(self):
        """
        Get one-hot encoded state representation

        Returns:
            one_hot_state: One-hot encoded state
        """
        # Create 6 faces x 9 squares x 6 colors one-hot encoding
        one_hot = np.zeros((54, 6))

        # Fill in the one-hot encoding
        idx = 0
        for face in ['U', 'D', 'F', 'B', 'L', 'R']:
            for i in range(3):
                for j in range(3):
                    color = self.cube.faces[face][i][j]
                    color_idx = self.color_map[color]
                    one_hot[idx, color_idx] = 1
                    idx += 1

        return one_hot

    def set_curriculum_level(self, level):
        """
        Set the current curriculum level

        Args:
            level: Integer representing difficulty (1-5)

        Returns:
            None
        """
        if 1 <= level <= 5:
            self.curriculum_level = level
            print(f"Curriculum level set to {level}")
        else:
            print(
                f"Invalid curriculum level: {level}. Using current level: {self.curriculum_level}"
            )

    def generate_curriculum_scramble(self):
        """
        Generate a scramble based on the current curriculum level

        Returns:
            The scramble formula string
        """
        scramble = generate_scramble_by_level(self.curriculum_level)
        return scramble

    def custom_scramble(self, scramble_sequence):
        """
        Apply a custom scramble sequence to the cube

        Args:
            scramble_sequence: A string of space-separated moves, e.g. "R U R' U'"

        Returns:
            bool: Whether the scramble was successfully applied
        """
        # Reset to solved state first
        self.reset()

        # Execute the scramble sequence
        success = self.execute_formula(scramble_sequence)
        if success:
            print(f"Applied custom scramble: {scramble_sequence}")
        else:
            print("Failed to apply custom scramble - invalid moves detected")

        return success

    def is_solved(self):
        """
        Check if the cube is solved

        Returns:
            boolean: Whether the cube is solved
        """
        # Check if each face has a single color
        for face in self.cube.faces:
            center_color = self.cube.faces[face][1][1]
            for row in self.cube.faces[face]:
                for color in row:
                    if color != center_color:
                        return False
        return True

    def render(self):
        """Display the current cube state"""
        self.cube.display()

    def save_image(self, filename="current_cube.png"):
        """Save the cube state as an image"""
        draw_cube(self.cube, filename)
        return filename

    def execute_formula(self, formula):
        """
        Execute a sequence of moves

        Args:
            formula: Cube formula string, e.g., "R U R' U'"

        Returns:
            Whether execution was successful
        """
        if not formula:
            return True

        # Parse the formula string
        moves = formula.split()

        # Execute each action
        for move in moves:
            if move not in self.action_space and move != "B" and move != "B'" and move != "B2":
                print(f"Warning: Formula contains invalid action '{move}'")
                return False

            if move == "B" or move == "B'" or move == "B2":
                print(f"Warning: Skipping B-face move '{move}' as requested")
                continue

            self.cube.move(move)

        return True

    def test_formula(self, formula, show_image=True):
        """
        Test a formula's effect

        Args:
            formula: Cube formula string
            show_image: Whether to display the result image

        Returns:
            The cube state after executing the formula
        """
        # Save current state
        original_cube = copy.deepcopy(self.cube)

        # Reset to standard state
        self.reset()

        print(f"Testing formula: {formula}")
        print("Initial state:")
        self.render()

        # Execute formula
        if not self.execute_formula(formula):
            print("Formula execution failed")
            self.cube = original_cube
            return None

        print("\nState after execution:")
        self.render()

        # Save image
        if show_image:
            filename = self.save_image("formula_result.png")
            print(f"Result saved as: {filename}")

        result_cube = copy.deepcopy(self.cube)

        # Restore original state
        self.cube = original_cube

        return result_cube

    def copy(self):
        """
        Create a deep copy of the current environment

        Returns:
            A new CubeEnv instance with the same state
        """
        new_env = CubeEnv(initial_state=self.cube)
        new_env.current_step = self.current_step
        new_env.curriculum_level = self.curriculum_level
        return new_env


# Helper functions
def create_scrambled_cube(scramble_formula=None,
                          n_random_moves=0,
                          use_oll_pll=False,
                          curriculum_level=None):
    """
    Create a scrambled cube

    Args:
        scramble_formula: Specific scramble formula to use
        n_random_moves: Number of random moves for scrambling
        use_oll_pll: Whether to use OLL+PLL generator
        curriculum_level: Curriculum difficulty level (1-5)

    Returns:
        Scrambled cube environment
    """
    env = CubeEnv()

    if curriculum_level is not None:
        env.set_curriculum_level(curriculum_level)
        scramble_formula = env.generate_curriculum_scramble()
        print(
            f"Using curriculum level {curriculum_level} scramble: {scramble_formula}"
        )

    elif use_oll_pll:
        scramble_formula = generate_oll_pll()
        print(f"Using OLL+PLL scramble: {scramble_formula}")

    if scramble_formula:
        env.execute_formula(scramble_formula)

    if n_random_moves > 0:
        for _ in range(n_random_moves):
            action = random.choice(env.action_space)
            env.step(action)

    return env


# Demo function for testing curriculum learning
def curriculum_demo():
    """Demonstrate curriculum learning functionality"""
    print("Demonstrating curriculum learning levels:")

    for level in range(1, 6):
        print(f"\n--- Curriculum Level {level} ---")
        env = create_scrambled_cube(curriculum_level=level)
        print("Scrambled cube state:")
        env.render()

        # Attempt to solve with a simple algorithm (just for demo)
        solved = False
        for _ in range(10):  # Try a few random moves
            action = random.choice(env.action_space)
            state, reward, done, info = env.step(action)
            if done and info["solved"]:
                solved = True
                print(f"Solved with action: {action}, reward: {reward}")
                break

        if not solved:
            print("Not solved with random actions (expected)")

        print("Final state:")
        env.render()

    return "Curriculum demonstration complete"


if __name__ == "__main__":
    # Run curriculum demo
    curriculum_demo()
