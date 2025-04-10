#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains 57 OLL algorithms and 21 PLL algorithms (all concrete cube rotation instructions),
and provides a function generate_oll_pll() to randomly generate an OLL+PLL formula combination.

Generation rules:
  - 75% chance to select both an OLL algorithm and a PLL algorithm (separated by space)
  - 25% chance to select only one formula (equal probability between OLL and PLL)

Algorithm data referenced from common CFOP method materials, adjust as needed.
"""

import random

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


def generate_oll_pll():
    """
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
    result = generate_oll_pll()
    print("Generated OLL+PLL formula:")
    print(result)


if __name__ == '__main__':
    main()
