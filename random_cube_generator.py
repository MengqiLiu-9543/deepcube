#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains algorithms for different difficulty levels and provides functions
to generate scrambles according to a curriculum:
Level 1: Single U move (easiest)
Level 2: Single OLL algorithms
Level 3: Single PLL algorithms
Level 4: OLL+PLL combinations
Level 5: Multiple OLL+PLL combinations (hardest)

The curriculum allows progressive learning from simple to complex patterns.
"""
import random

# Single U moves for the easiest level
SINGLE_U_MOVES = ["U", "U'", "U2"]

# OLL algorithms
OLL_ALGORITHMS = [
    "R U R' U' R U' R' F' U' F R U R'",
    "F R' F R2 U' R' U' R U R' F2",
    "R U R' U R U2 R' F R U R' U' F'",
    "R' U' R U' R' U2 R F R U R' U' F'",
    "L F' L' U' L U F U' L'",
    "R' F R U R' U' F' U R",
    "R U R2 U' R' F R U R U' F'",
    "R' U' R' F R F' U R",
    "R U2 R' U' R U R' U' R U' R'",
    "R U2 R2 U' R2 U' R2 U2 R",
    "R2 D' R U2 R' D R U2 R",
    "R U2 R' U' R U' R'",
    "R U R' U R U2 R'",
    "R U2 R2 F R F' U2 R' F R F'",
    "R U R' U' R' F R2 U R' U' F'",
    "R U R' U R' F R F' R U2 R'",
    "R U2 R2 F R F' R U2 R'",
    "F R' F' R U R U' R'",
    "F U R U' R' U R U' R' F'",
    "R U R' U R U' B U' B' R'",
    "R' F R U R U' R2 F' R2 U' R' U R U R'",
    "F U R U' R2 F' R U R U' R'",
    "R' F R U R' F' R F U' F'",
    "R' U' F U R U' R' F' R",
    "L U F' U' L' U L F L'",
    "F' U' L' U L F",
    "F U R U' R' F'",
    "R' U' R' F R F' R' F R F' U R",
    "F R U R' U' R U R' U' F'",
    "R U R' U' R' F R F'",
    "F R U R' U' F'",
    "L' U' L U' L' U L U L F' L' F",
    "R U R' U R U' R' U' R' F R F'",
]

# PLL algorithms
PLL_ALGORITHMS = [
    "F R U' R' U' R U R' F' R U R' U' R' F R F'",
    "R U R' U' R' F R2 U' R' U' R U R' F'",
    "R U' R U R U R U' R' U' R2",
    "R2 U R U R' U' R' U' R' U R'",
    "R' U' F' R U R' U' R' F R2 U' R' U' R U R' U R",
    "R2 U R' U R' U' R U' R2 U' D R' U R D'",
    "R U R' F' R U R' U' R' F R2 U' R'",
    "R U' R' U' R U R D R' U' R D' R' U2 R'",
    "R U R' U' R' F R2 U' R' U' R U R' F'",
    "R U R' U R U R' F' R U R' U' R' F R2 U' R' U2 R U' R'",
    "R' U R U' R' F' U' F R U R' F R' F' R U' R",
    "F R U' R' U' R U R' F' R U R' U' R' F R F'",
]

# Curriculum levels
CURRICULUM_LEVELS = {
    1: "single_u",  # Single U move
    2: "single_oll",  # Single OLL algorithm
    3: "single_pll",  # Single PLL algorithm
    4: "oll_pll",  # OLL+PLL combination
    5: "double_oll_pll"  # Multiple OLL+PLL combinations
}


def generate_scramble_by_level(level):
    """
    Generate a scramble formula based on the curriculum level

    Args:
        level: Integer representing difficulty (1-5)

    Returns:
        formula: Generated scramble formula
    """
    if level not in CURRICULUM_LEVELS:
        level = 1  # Default to easiest level

    level_type = CURRICULUM_LEVELS[level]

    if level_type == "single_u":
        # Level 1: Single U move
        formula = random.choice(SINGLE_U_MOVES)

    elif level_type == "single_oll":
        # Level 2: Single OLL algorithm
        formula = random.choice(OLL_ALGORITHMS)

    elif level_type == "single_pll":
        # Level 3: Single PLL algorithm
        formula = random.choice(PLL_ALGORITHMS)

    elif level_type == "oll_pll":
        # Level 4: OLL+PLL combination
        oll = random.choice(OLL_ALGORITHMS)
        pll = random.choice(PLL_ALGORITHMS)
        formula = oll + " " + pll

    elif level_type == "double_oll_pll":
        # Level 5: Two OLL+PLL combinations
        oll1 = random.choice(OLL_ALGORITHMS)
        pll1 = random.choice(PLL_ALGORITHMS)
        oll2 = random.choice(OLL_ALGORITHMS)
        pll2 = random.choice(PLL_ALGORITHMS)
        formula = oll1 + " " + pll1 + " " + oll2 + " " + pll2

    return formula


def generate_oll_pll():
    """
    Legacy function for backwards compatibility
    Randomly generate formula:
      - 75% chance to combine one OLL and one PLL algorithm
      - 25% chance to select single algorithm (equal probability between OLL and PLL)
    Returns the generated formula string.
    """
    x = random.random()
    if x < 0.25:
        # Generate single formula
        if random.random() < 0.5:
            formula = random.choice(OLL_ALGORITHMS)
        else:
            formula = random.choice(PLL_ALGORITHMS)
    elif x > 0.25 and x < 0.65:
        oll = random.choice(OLL_ALGORITHMS)
        pll = random.choice(PLL_ALGORITHMS)
        formula = oll + " " + pll
    else:
        oll1 = random.choice(OLL_ALGORITHMS)
        pll1 = random.choice(PLL_ALGORITHMS)
        oll2 = random.choice(OLL_ALGORITHMS)
        pll2 = random.choice(PLL_ALGORITHMS)
        formula = oll1 + " " + pll1 + " " + oll2 + " " + pll2
    return formula


def main():
    # Test each curriculum level
    for level in range(1, 6):
        formula = generate_scramble_by_level(level)
        print(f"Level {level} ({CURRICULUM_LEVELS[level]}) scramble:")
        print(formula)
        print()


if __name__ == '__main__':
    main()
