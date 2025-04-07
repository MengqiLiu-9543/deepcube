import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class SimpleRubiksCube:
    def __init__(self, cube_size=3):
        # 初始化一个已解决状态的魔方，每个面均匀填充对应的数字
        self.cube_size = cube_size
        self.cube = np.zeros((6, cube_size, cube_size), dtype=np.int32)
        for i in range(6):
            self.cube[i, :, :] = i

    def copy_cube(self):
        return np.copy(self.cube)

    def apply_move(self, move):
        """
        执行单个基本动作，支持 12 个动作 (F, F', B, B', L, L', R, R', U, U', D, D')
        """
        # 定义动作和面索引映射
        action_map = {
            'F': 0, "F'": 1,
            'B': 2, "B'": 3,
            'L': 4, "L'": 5,
            'R': 6, "R'": 7,
            'U': 8, "U'": 9,
            'D': 10, "D'": 11
        }
        if move not in action_map:
            print("未知的动作:", move)
            return
        action = action_map[move]

        # 将动作映射到对应实际操作的面
        # 标准定义：0: 上, 1: 左, 2: 前, 3: 右, 4: 后, 5: 下
        action_to_face = {
            0: 2, 1: 2,    # F, F' -> 前面(2)
            2: 4, 3: 4,    # B, B' -> 后面(4)
            4: 1, 5: 1,    # L, L' -> 左面(1)
            6: 3, 7: 3,    # R, R' -> 右面(3)
            8: 0, 9: 0,    # U, U' -> 上面(0)
            10: 5, 11: 5   # D, D' -> 下面(5)
        }
        # 约定：对于L和R动作，偶数是顺时针（k=1），奇数是逆时针（k=3）
        # 其他动作则相反
        if action in [4, 5, 6, 7]:  # L, L', R, R'
            direction = 1 if action % 2 == 0 else 3
        else:
            direction = 3 if action % 2 == 0 else 1
        face_idx = action_to_face[action]

        # 旋转该面
        self.cube[face_idx] = np.rot90(self.cube[face_idx], k=direction)

        # 更新相邻面的边缘状态
        # 这里我们给出一个简化版的更新映射：每个面旋转会影响相邻面边缘的交换
        adjacent_faces = {
            # 上面 (0)旋转影响：前(2)、右(3)、后(4)、左(1) 的顶行
            0: [(2, 0, False), (3, 0, False), (4, 0, False), (1, 0, False)],
            # 下面 (5)旋转影响：前(2)、右(3)、后(4)、左(1) 的底行
            5: [(2, 2, False), (3, 2, False), (4, 2, False), (1, 2, False)],
            # 前面 (2)旋转影响：上(0)的底行、右(3)的左列、下(5)的顶行、左(1)的右列
            2: [(0, 2, False), (3, 0, False), (5, 0, True), (1, 2, True)],
            # 后面 (4)旋转影响：上(0)的顶行、左(1)的左列、下(5)的底行、右(3)的右列
            4: [(0, 0, True), (1, 0, True), (5, 2, False), (3, 2, False)],
            # 左面 (1)旋转影响：上(0)的左列、前(2)的左列、下(5)的左列、后(4)的右列
            1: [(0, 0, True), (2, 0, True), (5, 0, False), (4, 2, False)],
            # 右面 (3)旋转影响：上(0)的右列、前(2)的右列、下(5)的右列、后(4)的左列
            3: [(0, 2, True), (2, 2, True), (5, 2, False), (4, 0, False)]
        }
        if face_idx not in adjacent_faces:
            return

        cube_copy = self.copy_cube()
        strips = []
        # 提取受影响边缘
        for (adj_face, pos, rev) in adjacent_faces[face_idx]:
            # 这里对上和下的相邻面直接按行提取，其余一般取列
            if face_idx in [0, 5] or (face_idx in [2, 4] and adj_face in [0, 5]):
                strip = cube_copy[adj_face, pos, :].copy()
            else:
                strip = cube_copy[adj_face, :, pos].copy()
            if rev:
                strip = strip[::-1]
            strips.append(strip)

        # 根据旋转方向调整边缘顺序：顺时针右移一位；逆时针左移一位
        if direction == 1:
            strips = [strips[-1]] + strips[:-1]
        else:
            strips = strips[1:] + [strips[0]]

        # 将更新后的边缘写回相邻面
        for i, (adj_face, pos, rev) in enumerate(adjacent_faces[face_idx]):
            strip = strips[i]
            if rev:
                strip = strip[::-1]
            if face_idx in [0, 5] or (face_idx in [2, 4] and adj_face in [0, 5]):
                self.cube[adj_face, pos, :] = strip
            else:
                self.cube[adj_face, :, pos] = strip

    def apply_algorithm(self, algorithm_str):
        """
        根据空格分隔的公式字符串依次应用动作，
        支持类似 "R U R' U'" 以及 "R2"（代表执行两次 R 动作）的写法。
        """
        moves = algorithm_str.split()
        for move in moves:
            if len(move) == 2 and move[1] == '2':
                base = move[0]
                self.apply_move(base)
                self.apply_move(base)
            else:
                self.apply_move(move)

    def get_cube(self):
        return self.cube

def plot_cube(cube, save_path="scrambled_cube.png"):
    """
    使用matplotlib将魔方的六个面绘制到一个 2x3 的子图中，
    并保存为PNG图片。
    """
    # 定义面名顺序：依据初始化顺序 0: 上, 1: 左, 2: 前, 3: 右, 4: 后, 5: 下
    face_names = ["Up", "Left", "Front", "Right", "Back", "Down"]
    # 定义颜色映射（与魔方状态数值对应：0上白、1左绿、2前红、3右蓝、4后橙、5下黄）
    cmap = colors.ListedColormap(['white', 'green', 'red', 'blue', 'orange', 'yellow'])

    fig, axs = plt.subplots(2, 3, figsize=(8, 6))
    axs = axs.flatten()

    for i in range(6):
        ax = axs[i]
        ax.imshow(cube[i], cmap=cmap, vmin=-0.5, vmax=5.5)
        ax.set_title(face_names[i])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"打乱后的魔方状态已保存为 '{save_path}'.")

def main():
    # 创建魔方实例（初始为已解决状态）
    cube = SimpleRubiksCube()
    # 定义打乱公式，例如 "R U R' U'"
    algorithm = "L"
    print("应用公式:", algorithm)
    cube.apply_algorithm(algorithm)

    # 获取打乱后的魔方状态
    scrambled_cube = cube.get_cube()
    print("打乱后的魔方状态:")
    for face_idx, face in enumerate(scrambled_cube):
        print(f"Face {face_idx}:")
        print(face)

    # 绘制魔方状态，并保存为 PNG 文件
    plot_cube(scrambled_cube, save_path="scrambled_cube.png")

if __name__ == "__main__":
    main()
