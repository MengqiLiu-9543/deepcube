#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code implements:
1. RubikCube class to simulate cube state, generating scrambled states based on input move sequences (e.g., "U R U' L2")
2. Using matplotlib to draw cube net diagrams _and save as images (rubik_cube.png)
Note: Left and right face colors have been adjusted, left face is green, right face is blue.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from random_cube_generator import generate_oll_pll

def rotate_face(face):
    """Rotate a 3x3 matrix face clockwise"""
    return [list(x)[::-1] for x in zip(*face)]

def rotate_face_ccw(face):
    """Rotate a 3x3 matrix face counterclockwise"""
    return [list(x) for x in zip(*face[::-1])]

def rotate_face_180(face):
    """Rotate 180 degrees"""
    return [row[::-1] for row in face[::-1]]


class RubikCube:
    def __init__(self):
        # Define six faces of the cube using standard colors:
        # U(up): White(W), D(down): Yellow(Y)
        # F(front): Red(R), B(back): Orange(O)
        # Modified left-right faces: L(left): Green(G), R(right): Blue(B)
        self.faces = {
            'U': [['W'] * 3 for _ in range(3)],
            'D': [['Y'] * 3 for _ in range(3)],
            'F': [['R'] * 3 for _ in range(3)],
            'B': [['O'] * 3 for _ in range(3)],
            'L': [['G'] * 3 for _ in range(3)],  # Left face: green
            'R': [['B'] * 3 for _ in range(3)]   # Right face: blue
        }

    def move(self, notation):
        """
        Parse single move notation
        Supports formats like "U", "U'", "U2"
        """
        if notation.endswith("2"):
            times = 2
            move = notation[0]
        elif notation.endswith("'"):
            times = 3  # Three clockwise rotations equivalent to one counterclockwise
            move = notation[0]
        else:
            times = 1
            move = notation[0]

        for _ in range(times):
            self._apply_move(move)

    def _apply_move(self, move):
        if move == 'U':
            self.U()
        elif move == 'D':
            self.D()
        elif move == 'F':
            self.F()
        elif move == 'B':
            self.B()
        elif move == 'L':
            self.L()
        elif move == 'R':
            self.R()
        else:
            print(f"Unknown move notation: {move}")

    def U(self):
        """Upper layer clockwise rotation"""
        self.faces['U'] = rotate_face(self.faces['U'])
        # U layer adjacent to first row of F, R, B, L faces
        temp = self.faces['F'][0][:]
        self.faces['F'][0] = self.faces['R'][0][:]
        self.faces['R'][0] = self.faces['B'][0][:]
        self.faces['B'][0] = self.faces['L'][0][:]
        self.faces['L'][0] = temp

    def D(self):
        """Down layer clockwise rotation"""
        self.faces['D'] = rotate_face(self.faces['D'])
        # D layer adjacent to bottom row of F, L, B, R faces
        temp = self.faces['F'][2][:]
        self.faces['F'][2] = self.faces['L'][2][:]
        self.faces['L'][2] = self.faces['B'][2][:]
        self.faces['B'][2] = self.faces['R'][2][:]
        self.faces['R'][2] = temp

    def F(self):
        """Front face clockwise rotation"""
        self.faces['F'] = rotate_face(self.faces['F'])
        # F face adjacent to U bottom row, L right column, D top row, R left column
        temp = self.faces['U'][2][:]
        for i in range(3):
            self.faces['U'][2][i] = self.faces['L'][2 - i][2]
        for i in range(3):
            self.faces['L'][i][2] = self.faces['D'][0][i]
        for i in range(3):
            self.faces['D'][0][i] = self.faces['R'][2 - i][0]
        for i in range(3):
            self.faces['R'][i][0] = temp[i]

    def B(self):
        """Back face clockwise rotation"""
        self.faces['B'] = rotate_face(self.faces['B'])
        # B face adjacent to U top row, R right column, D bottom row, L left column
        temp = self.faces['U'][0][:]
        for i in range(3):
            self.faces['U'][0][i] = self.faces['R'][i][2]
        for i in range(3):
            self.faces['R'][i][2] = self.faces['D'][2][2 - i]
        for i in range(3):
            self.faces['D'][2][i] = self.faces['L'][i][0]
        for i in range(3):
            self.faces['L'][i][0] = temp[2 - i]

    def L(self):
        """Left face clockwise rotation"""
        self.faces['L'] = rotate_face(self.faces['L'])
        # L face adjacent to U left column, B right column, D left column, F left column
        temp = [self.faces['U'][i][0] for i in range(3)]
        for i in range(3):
            self.faces['U'][i][0] = self.faces['B'][2 - i][2]
        for i in range(3):
            self.faces['B'][i][2] = self.faces['D'][2 - i][0]
        for i in range(3):
            self.faces['D'][i][0] = self.faces['F'][i][0]
        for i in range(3):
            self.faces['F'][i][0] = temp[i]

    def R(self):
        """Right face clockwise rotation"""
        self.faces['R'] = rotate_face(self.faces['R'])
        # R face adjacent to U right column, F right column, D right column, B left column
        temp = [self.faces['U'][i][2] for i in range(3)]
        for i in range(3):
            self.faces['U'][i][2] = self.faces['F'][i][2]
        for i in range(3):
            self.faces['F'][i][2] = self.faces['D'][i][2]
        for i in range(3):
            self.faces['D'][i][2] = self.faces['B'][2 - i][0]
        for i in range(3):
            self.faces['B'][i][0] = temp[2 - i]

    def display(self):
        """Print cube state to console (net diagram)"""
        # Print upper layer
        print("      Up")
        for row in self.faces['U']:
            print("      " + " ".join(row))
        print("Left    Front    Right    Back")
        for i in range(3):
            print(" ".join(self.faces['L'][i]) + "     " +
                  " ".join(self.faces['F'][i]) + "     " +
                  " ".join(self.faces['R'][i]) + "     " +
                  " ".join(self.faces['B'][i]))
        print("      Down")
        for row in self.faces['D']:
            print("      " + " ".join(row))


def draw_cube(cube, filename="rubik_cube.png"):
    """
    Draw cube net diagram using matplotlib:

           +-----+
           |  U  |
    +-----+-----+-----+-----+
    |  L  |  F  |  R  |  B  |
    +-----+-----+-----+-----+
           |  D  |
           +-----+

    Each face is a 3x3 grid, filled with corresponding colors:
      U: White, D: Yellow, F: Red, B: Orange, L: Green, R: Blue
    """
    # Define face offsets on canvas (x, y)
    face_offsets = {
        'U': (3, 6),  # Upper face, starts at (3,6)
        'L': (0, 3),
        'F': (3, 3),
        'R': (6, 3),
        'B': (9, 3),
        'D': (3, 0)
    }

    # Define color mapping, no modification needed: G for green, B for blue
    color_map = {
        'W': "#FFFFFF",  # White
        'Y': "#FFFF00",  # Yellow
        'R': "#FF0000",  # Red
        'O': "#FFA500",  # Orange
        'G': "#00FF00",  # Green
        'B': "#0000FF"   # Blue
    }

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw each face
    for face, offset in face_offsets.items():
        ox, oy = offset
        # Each face is 3x3 grid, note: row 0 in list is top row, reverse for drawing
        for i in range(3):
            for j in range(3):
                color_letter = cube.faces[face][i][j]
                rect = Rectangle((ox + j, oy + (2 - i)), 1, 1,
                                 edgecolor='black', facecolor=color_map[color_letter], lw=2)
                ax.add_patch(rect)
                ax.text(ox + j + 0.5, oy + (2 - i) + 0.5, color_letter,
                        fontsize=14, ha='center', va='center')
        ax.text(ox + 1.5, oy + 3.1, face, fontsize=16, ha='center')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Cube net diagram saved as {filename}")


def main():
    cube = RubikCube()
    scramble = generate_oll_pll()
    # scramble = "B"
    moves = scramble.split()
    for move in moves:
        cube.move(move)
    print("Scramble formula:", scramble)
    print("\nCube state after scramble:")
    cube.display()

    draw_cube(cube, filename="rubik_cube.png")


if __name__ == "__main__":
    main()
