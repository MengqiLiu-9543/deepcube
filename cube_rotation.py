#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
此代码实现了：
1. RubikCube 类，用于模拟魔方状态，根据输入的转动指令（例如 "U R U' L2"）生成打乱状态；
2. 利用 matplotlib 绘制魔方展开图，并保存为图片 (rubik_cube.png)。
注意：左右面颜色已调整，左面为绿色，右面为蓝色。
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from random_cube_generator import generate_oll_pll

def rotate_face(face):
    """顺时针旋转 3x3 矩阵表示的面"""
    return [list(x)[::-1] for x in zip(*face)]

def rotate_face_ccw(face):
    """逆时针旋转 3x3 矩阵表示的面"""
    return [list(x) for x in zip(*face[::-1])]

def rotate_face_180(face):
    """180度旋转"""
    return [row[::-1] for row in face[::-1]]


class RubikCube:
    def __init__(self):
        # 定义魔方六个面，采用常用颜色：
        # U(上): 白色(W)、D(下): 黄色(Y)
        # F(前): 红色(R)、B(后): 橙色(O)
        # 修改左右面：左面 L 为绿色(G)，右面 R 为蓝色(B)
        self.faces = {
            'U': [['W'] * 3 for _ in range(3)],
            'D': [['Y'] * 3 for _ in range(3)],
            'F': [['R'] * 3 for _ in range(3)],
            'B': [['O'] * 3 for _ in range(3)],
            'L': [['G'] * 3 for _ in range(3)],  # 左面：绿色
            'R': [['B'] * 3 for _ in range(3)]   # 右面：蓝色
        }

    def move(self, notation):
        """
        解析单个转动指令
        支持例如 "U", "U'", "U2"
        """
        if notation.endswith("2"):
            times = 2
            move = notation[0]
        elif notation.endswith("'"):
            times = 3  # 顺时针旋转三次 等价于一次逆时针
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
            print(f"未知的转动指令: {move}")

    def U(self):
        """上层顺时针转动"""
        self.faces['U'] = rotate_face(self.faces['U'])
        # U 层相邻于 F, R, B, L 四个面的第一行
        temp = self.faces['F'][0][:]
        self.faces['F'][0] = self.faces['R'][0][:]
        self.faces['R'][0] = self.faces['B'][0][:]
        self.faces['B'][0] = self.faces['L'][0][:]
        self.faces['L'][0] = temp

    def D(self):
        """下层顺时针转动"""
        self.faces['D'] = rotate_face(self.faces['D'])
        # D 层相邻于 F, L, B, R 四个面的底行
        temp = self.faces['F'][2][:]
        self.faces['F'][2] = self.faces['L'][2][:]
        self.faces['L'][2] = self.faces['B'][2][:]
        self.faces['B'][2] = self.faces['R'][2][:]
        self.faces['R'][2] = temp

    def F(self):
        """前面顺时针转动"""
        self.faces['F'] = rotate_face(self.faces['F'])
        # F 面相邻于 U 底行、L 右列、D 顶行、R 左列
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
        """后面顺时针转动"""
        self.faces['B'] = rotate_face(self.faces['B'])
        # B 面相邻于 U 顶行、R 右列、D 底行、L 左列
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
        """左面顺时针转动"""
        self.faces['L'] = rotate_face(self.faces['L'])
        # L 面相邻于 U 左列、B 右列、D 左列、F 左列
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
        """右面顺时针转动"""
        self.faces['R'] = rotate_face(self.faces['R'])
        # R 面相邻于 U 右列、F 右列、D 右列、B 左列
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
        """控制台打印魔方状态（展开图）"""
        # 打印上层
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
    利用 matplotlib 绘制魔方展开图：

           +-----+
           |  U  |
    +-----+-----+-----+-----+
    |  L  |  F  |  R  |  B  |
    +-----+-----+-----+-----+
           |  D  |
           +-----+

    每个面为 3x3 的方块，各小方块用相应颜色填充：
      U: 白色, D: 黄色, F: 红色, B: 橙色, L: 绿色, R: 蓝色
    """
    # 定义六个面的在画布上的偏移位置 (x, y)
    face_offsets = {
        'U': (3, 6),  # 上面，从 (3,6) 开始绘制
        'L': (0, 3),
        'F': (3, 3),
        'R': (6, 3),
        'B': (9, 3),
        'D': (3, 0)
    }

    # 定义颜色映射，不需修改：G 代表绿色，B 代表蓝色
    color_map = {
        'W': "#FFFFFF",  # 白色
        'Y': "#FFFF00",  # 黄色
        'R': "#FF0000",  # 红色
        'O': "#FFA500",  # 橙色
        'G': "#00FF00",  # 绿色
        'B': "#0000FF"   # 蓝色
    }

    fig, ax = plt.subplots(figsize=(6, 6))

    # 绘制每个面
    for face, offset in face_offsets.items():
        ox, oy = offset
        # 每个面 3x3 小格，注意：列表中 row 0 为顶行，绘图时反向显示
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
    print(f"魔方展开图已保存为 {filename}")


def main():
    cube = RubikCube()
    scramble = generate_oll_pll()
    # scramble = "B"
    moves = scramble.split()
    for move in moves:
        cube.move(move)
    print("打乱公式：", scramble)
    print("\n打乱后的魔方状态：")
    cube.display()

    draw_cube(cube, filename="rubik_cube.png")


if __name__ == "__main__":
    main()
