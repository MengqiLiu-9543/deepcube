#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
此文件存储了 57 个 OLL 算法和 21 个 PLL 算法（均为具体魔方转动指令），
并提供一个函数 generate_oll_pll() 用于随机生成一个 OLL+PLL 的公式组合。

生成规则：
  - 75% 概率同时选取一个 OLL 算法和一个 PLL 算法（中间以空格分隔）
  - 25% 概率只选取一个公式（在 OLL 和 PLL 中各半概率）

算法数据参考自常见 CFOP 方法资料，如需调整请自行替换。
"""

import random

#OLL 算法
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

# PLL 
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
    随机生成公式：
      - 75% 概率同时选择一个 OLL 和一个 PLL 算法，拼接为一个组合公式
      - 25% 概率只选择单个算法（在 OLL 与 PLL 中各半概率）
    返回生成的公式字符串。
    """
    x = random.random()
    if x < 0.25:
        # 只生成一个公式
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
    print("生成的 OLL+PLL 公式为:")
    print(result)


if __name__ == '__main__':
    main()
