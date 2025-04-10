#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search (MCTS) for Rubik's Cube 1LLL Solver

This module implements MCTS as described in the DeepCube paper:
- Uses a neural network to guide exploration
- Uses Upper Confidence Bound (UCB) for action selection
- Prioritizes promising moves during search
- Combines value estimation with tree search
"""

import numpy as np
import math
import random
import time

# Constants for the MCTS algorithm
DEFAULT_EXPLORATION_CONSTANT = 1.25
MAX_SIMULATIONS = 200
MAX_SEARCH_DEPTH = 20


class MCTSNode:
    """Node in the Monte Carlo Tree Search"""

    def __init__(self, state, parent=None, prior=0.0, action_taken=None):
        self.state = state  # Cube state
        self.parent = parent  # Parent node
        self.prior = prior  # Prior probability from policy network
        self.action_taken = action_taken  # Action that led to this state

        self.children = {}  # Maps actions to child nodes
        self.visit_count = 0  # Number of visits to this node
        self.value_sum = 0.0  # Sum of values from this node
        self.value = 0.0  # Average value

    def is_expanded(self):
        """Check if node has been expanded"""
        return len(self.children) > 0

    def select_child(self, exploration_constant):
        """
        Select child using the UCB formula

        Args:
            exploration_constant: Controls exploration vs exploitation

        Returns:
            action, child_node: Selected action and child node
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None

        # Calculate UCB for each child
        for action, child in self.children.items():
            # Exploration term
            # Based on UCB formula: Q(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            if child.visit_count > 0:
                # Exploitation term
                exploit = child.value

                # Exploration term
                explore = exploration_constant * child.prior * \
                         (math.sqrt(self.visit_count) / (1 + child.visit_count))

                # UCB score
                ucb = exploit + explore
            else:
                # If unvisited, prioritize by prior probability
                ucb = exploration_constant * child.prior * math.sqrt(
                    self.visit_count + 1e-8)

            # Update best child if better score found
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, actions, action_priors, next_states):
        """
        Expand node with child nodes

        Args:
            actions: List of possible actions
            action_priors: Prior probabilities for actions
            next_states: Resulting states for each action
        """
        for action, prior, next_state in zip(actions, action_priors,
                                             next_states):
            # Don't expand if already exists
            if action not in self.children:
                self.children[action] = MCTSNode(state=next_state,
                                                 parent=self,
                                                 prior=prior,
                                                 action_taken=action)

    def update(self, value):
        """
        Update statistics after simulation

        Args:
            value: Value estimate to propagate
        """
        self.visit_count += 1
        self.value_sum += value
        self.value = self.value_sum / self.visit_count

    def get_best_action(self):
        """Get action with most visits (for final move selection)"""
        visits = [(action, child.visit_count)
                  for action, child in self.children.items()]
        action, _ = max(visits, key=lambda x: x[1])
        return action


class MCTS:
    """Monte Carlo Tree Search algorithm"""

    def __init__(self,
                 neural_network,
                 exploration_constant=DEFAULT_EXPLORATION_CONSTANT,
                 max_simulations=MAX_SIMULATIONS):
        """
        Initialize MCTS algorithm

        Args:
            neural_network: Neural network for state evaluation
            exploration_constant: UCB exploration constant
            max_simulations: Maximum simulations per move
        """
        self.neural_network = neural_network
        self.exploration_constant = exploration_constant
        self.max_simulations = max_simulations

    def search(self, env, time_limit=None):
        """
        Perform MCTS search from current state

        Args:
            env: CubeEnv environment
            time_limit: Optional time limit in seconds

        Returns:
            best_action: Action with highest visit count
        """
        # Create root node
        root = MCTSNode(state=env.get_one_hot_state())

        # Evaluate root node with neural network
        value, policy = self.neural_network.predict(root.state)

        # Get all possible actions
        actions = list(range(len(env.action_space)))

        # Generate next states for all actions
        next_states = []
        for action in actions:
            # Copy environment
            test_env = env.copy()
            # Take action
            test_env.step(action)
            # Get new state
            next_states.append(test_env.get_one_hot_state())

        # Expand root with all children
        root.expand(actions, policy, next_states)

        # Record start time if using time limit
        start_time = time.time()

        # Run simulations
        num_simulations = 0

        while num_simulations < self.max_simulations:
            if time_limit and (time.time() - start_time) > time_limit:
                break

            # Simulation consists of selection, expansion, evaluation and backup
            self._simulate(root, env)
            num_simulations += 1

            # Print progress occasionally
            if num_simulations % 50 == 0:
                print(f"Completed {num_simulations} simulations")

        # Print statistics
        print(f"Completed {num_simulations} simulations")

        # Select best action from root
        best_action = root.get_best_action()

        # Return best action
        return best_action

    def _simulate(self, root, env):
        """
        Perform a single simulation

        Args:
            root: Root node of search tree
            env: CubeEnv environment
        """
        # Make a copy of the environment
        sim_env = env.copy()

        # Select path through tree (selection phase)
        path = []
        node = root

        # Selection phase - find a leaf node
        while node.is_expanded():
            action, node = node.select_child(self.exploration_constant)
            path.append((action, node))

            # Apply action in simulation environment
            sim_env.step(action)

            # If solved or reached maximum depth, stop selection
            if sim_env.is_solved() or len(path) >= MAX_SEARCH_DEPTH:
                break

        # Expansion and evaluation phase
        # If node is not terminal, expand it
        if not sim_env.is_solved() and len(path) < MAX_SEARCH_DEPTH:
            # Get state from simulation environment
            state = sim_env.get_one_hot_state()

            # Evaluate with neural network
            value, policy = self.neural_network.predict(state)

            # Get possible actions
            actions = list(range(len(sim_env.action_space)))

            # Generate next states
            next_states = []
            for action in actions:
                # Copy environment
                test_env = sim_env.copy()
                # Take action
                test_env.step(action)
                # Get new state
                next_states.append(test_env.get_one_hot_state())

            # Expand node with all children
            node.expand(actions, policy, next_states)
        else:
            # If terminal node, value is determined by whether cube is solved
            value = 1.0 if sim_env.is_solved() else -1.0

        # Backup phase - update statistics for all nodes in path
        # First update the newly expanded node
        node.update(value)

        # Then update all parent nodes
        for action, node in reversed(path):
            node.update(value)
