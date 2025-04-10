#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cube Environment Module (CubeEnv):
- Based on cube_rotation.py RubikCube class
- Provides environment interface for Reinforcement Learning
- Supports state representation, action execution, and reward computation
- Includes formula testing functionality and OLL+PLL scramble generation
"""

import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# Import the original cube code
from cube_rotation import RubikCube, draw_cube, rotate_face, rotate_face_ccw, rotate_face_180
# Import the OLL+PLL generator
from random_cube_generator import generate_oll_pll


class CubeEnv:
    """
    Cube Environment based on RubikCube class
    Suitable for Reinforcement Learning training
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

    def reset(self, scramble=None, use_oll_pll=False):
        """
        Reset the environment

        Args:
            scramble: Optional scramble formula string, e.g., "R U F' L"
            use_oll_pll: If True, use generate_oll_pll() to create a scramble

        Returns:
            state: The state after reset
        """
        # Reset to initial state
        self.cube = RubikCube()

        # Generate a scramble if requested
        if use_oll_pll:
            scramble = generate_oll_pll()
            print(f"Generated OLL+PLL scramble: {scramble}")

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
        # - Completion: +1
        # - Each step: -1 (encourages fewer steps)
        if self.is_solved():
            reward = 1.0
        else:
            reward = -1

        # Return next state, reward, completion status, and info
        return self.get_state(), reward, done, {
            "action": action_str,
            "steps": self.current_step,
            "solved": self.is_solved()
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


    def custom_scramble(self, scramble_sequence):
        """
        Apply a custom scramble sequence to the cube

        Args:
            scramble_sequence: A string of space-separated moves, e.g. "R U R' U'"

        Returns:
            The cube state after applying the scramble
        """
        # Reset to solved state first
        self.reset()
        
        # Execute the scramble sequence
        if self.execute_formula(scramble_sequence):
            print(f"Applied custom scramble: {scramble_sequence}")
        else:
            print("Failed to apply custom scramble - invalid moves detected")
            
        return self.get_state()

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


# Helper functions
def create_scrambled_cube(scramble_formula=None,
                          n_random_moves=0,
                          use_oll_pll=False):
    """
    Create a scrambled cube

    Args:
        scramble_formula: Specific scramble formula to use
        n_random_moves: Number of random moves for scrambling
        use_oll_pll: Whether to use OLL+PLL generator

    Returns:
        Scrambled cube environment
    """
    env = CubeEnv()

    if use_oll_pll:
        scramble_formula = generate_oll_pll()
        print(f"Using OLL+PLL scramble: {scramble_formula}")

    if scramble_formula:
        env.execute_formula(scramble_formula)

    if n_random_moves > 0:
        for _ in range(n_random_moves):
            action = random.choice(env.action_space)
            env.step(action)

    return env


'''
The demo code is commented out as it is not needed for the current implementation.
# Demo code
def demo():
    """Demonstrate environment functionality"""
    # Create environment
    env = CubeEnv()

    # Test a formula
    test_formula = "R U R' U'"
    print(f"Testing formula: {test_formula}")
    env.test_formula(test_formula)

    # Test OLL+PLL scramble
    print("\nTesting OLL+PLL scramble:")
    scrambled_env = create_scrambled_cube(use_oll_pll=True)
    scrambled_env.render()
    scrambled_env.save_image("oll_pll_scrambled.png")

    # Execute some actions
    print("\nExecuting some actions:")
    state, reward, done, info = scrambled_env.step("F")
    print(f"Action: F, Reward: {reward}, Completed: {done}")
    scrambled_env.render()

    return env

if __name__ == "__main__":
    # Run demo
    demo()

'''
