import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import random
from matplotlib.patches import Rectangle

class RubiksCubeEnv(gym.Env):
    """
    魔方环境的简化版实现，基于Gymnasium框架
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    # 修改OLL/PLL算法列表，只包含确保不会影响前两层的算法
    oll_algs = [
        "R U2 R' U' R U' R'",     # OLL 27 (Sune)
        "R' U' R U' R' U2 R",     # OLL 26 (Antisune)
        "F R U R' U' F'",         # OLL 45
        "U R U2 R' U' R U' R'"    # OLL 27 with U 前缀
    ]

    pll_algs = [
        "R U' R U R U R U' R' U' R2",   # U-perm (a)
        "R2 U R U R' U' R' U' R' U R'", # U-perm (b)
        "U R U' R U R U R U' R' U' R2"  # U-perm (a) with U 前缀
    ]

    # 只改变顶层朝向的几种简单插入
    u_moves = ["", "U", "U2", "U'"]
    
    # 动作映射字典：从公式中的字符转换到数字动作
    action_map = {
        'F': 0,   # 前面顺时针
        "F'": 1,  # 前面逆时针
        'B': 2,   # 后面顺时针
        "B'": 3,  # 后面逆时针
        'L': 4,   # 左面顺时针
        "L'": 5,  # 左面逆时针
        'R': 6,   # 右面顺时针
        "R'": 7,  # 右面逆时针
        'U': 8,   # 上面顺时针
        "U'": 9,  # 上面逆时针
        'D': 10,  # 下面顺时针
        "D'": 11, # 下面逆时针
        'M': 12,  # 中层
        "M'": 13, # 中层'
        'M2': 14  # 中层2
    }
    
    def __init__(self, cube_size=3, render_mode=None, max_steps=50, scramble_moves=5, use_oll_pll=False, debug_mode=False):
        """
        初始化魔方环境
        Args:
            cube_size: 魔方大小（通常为3）
            render_mode: 渲染模式，可选 'human' 或 'rgb_array'
            max_steps: 一个episode的最大步数
            scramble_moves: 初始打乱的步数
            use_oll_pll: 是否使用OLL/PLL算法打乱（仅当scramble_moves=0时）
            debug_mode: 是否开启调试模式
        """
        self.cube_size = cube_size
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.scramble_moves = scramble_moves
        self.use_oll_pll = use_oll_pll
        self.steps = 0
        self.debug_mode = debug_mode
        
        # 动作空间：12个动作，对应6个面的顺/逆时针旋转
        # F, F', B, B', L, L', R, R', U, U', D, D'
        self.action_space = spaces.Discrete(12)
        
        # 状态空间：6个面，每个面是cube_size x cube_size的颜色值
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(6, cube_size, cube_size), dtype=np.int32
        )
        
        # 用于渲染的颜色映射
        self.color_map = {
            0: 'white',    # 上面 (U)
            1: 'yellow',   # 下面 (D)
            2: 'red',      # 前面 (F)
            3: 'orange',   # 后面 (B)
            4: 'green',    # 左面 (L)
            5: 'blue'      # 右面 (R)
        }
        
        # 初始化魔方
        self._reset_cube()
        
        # 记录最佳奖励，用于调试和评估
        self.best_reward = -np.inf
        
        # 渲染相关
        self.fig = None
        self.axs = None
    
    def _reset_cube(self):
        """初始化已解决的魔方状态"""
        self.cube = np.zeros((6, self.cube_size, self.cube_size), dtype=np.int32)
        for i in range(6):
            self.cube[i, :, :] = i
    
    def _check_first_two_layers_solved(self):
        """
        检查魔方的前两层是否已解决。
        
        前两层定义为：
        1. 下面(D=5)的所有块
        2. 前(F=2)、后(B=4)、左(L=1)、右(R=3)面的下两行（不包括与上面(U=0)相邻的第一行）
        
        返回:
            bool: 如果前两层已解决则为True，否则为False
        """
        # 标准面索引: 0=U, 1=L, 2=F, 3=R, 4=B, 5=D
        
        # 调试输出
        if self.debug_mode:
            print("\n检查前两层是否已解决:")
            print("标准面索引: 0=上(U), 1=左(L), 2=前(F), 3=右(R), 4=后(B), 5=下(D)")
        
        # 1. 检查下面 (D=5) 是否完全还原 - 所有元素应该等于5
        if not np.all(self.cube[5] == 5):
            if self.debug_mode:
                print("下面(D)未还原:")
                print(self.cube[5])
            return False
        
        # 2. 检查四个侧面的底部两行是否还原
        for face_idx, face_name in zip([1, 2, 3, 4], ["左(L)", "前(F)", "右(R)", "后(B)"]):
            face_color = face_idx  # 在标准索引中，面的颜色等于面的索引
            
            # 检查底部两行 (行索引1和2)
            for row in range(1, 3):
                if not np.all(self.cube[face_idx, row, :] == face_color):
                    if self.debug_mode:
                        print(f"{face_name}面的第{row+1}行未还原")
                        print(f"期望值: {face_color}")
                        print(f"实际值: {self.cube[face_idx, row, :]}")
                    return False
        
        # 如果所有检查都通过，则前两层已解决
        if self.debug_mode:
            print("前两层已成功解决!")
        return True
    
    def _parse_and_apply_algorithm(self, algorithm):
        """解析并应用魔方算法字符串"""
        moves = algorithm.split()
        actions = []
        
        for move in moves:
            if move == "M2":
                # M2 可以分解为两个 M（中层）动作
                self._apply_m_slice()
                self._apply_m_slice()
                actions.append(14)  # M2
            elif move == "M":
                self._apply_m_slice()
                actions.append(12)  # M
            elif move == "M'":
                self._apply_m_slice(reverse=True)
                actions.append(13)  # M'
            elif move in self.action_map:
                action = self.action_map[move]
                if action < 12:  # 只有标准12个动作能用_apply_action
                    self._apply_action(action)
                    actions.append(action)
            else:
                # 处理 U2, R2 等二阶动作
                if len(move) == 2 and move[1] == '2':
                    base_move = move[0]
                    if base_move in 'FBLRUD':
                        # 执行两次对应的基本动作
                        action = self.action_map[base_move]
                        self._apply_action(action)
                        self._apply_action(action)
                        actions.append(action)
                        actions.append(action)
                    
        return actions
    
    def _apply_m_slice(self, reverse=False):
        """应用M层（中间层）旋转
        
        M层是L和R之间的中间层，按照L的方向旋转
        注意：由于我们简化的魔方模型，M层旋转会影响前两层，因此我们在OLL/PLL中避免使用
        """
        # M = L 方向的中间层
        # 保存当前状态
        cube_copy = np.copy(self.cube)
        
        # 获取中间层索引
        mid_idx = self.cube_size // 2
        
        # 中间层会影响四个面：上(U)、前(F)、下(D)、后(B)
        # 按照列转移方式：上 -> 前 -> 下 -> 后 -> 上
        faces = [0, 2, 1, 3]  # 上, 前, 下, 后
        strips = []
        
        # 收集所有面的中间列
        for face in faces:
            strip = cube_copy[face, :, mid_idx].copy()
            # 后面需要翻转
            if face == 3:  # 后面
                strip = strip[::-1]
            strips.append(strip)
        
        # 根据方向旋转
        if not reverse:  # L方向 (逆时针看左面) - 默认
            # 移动: 上 -> 前 -> 下 -> 后 -> 上
            destination = [(2, mid_idx), (1, mid_idx), (3, mid_idx), (0, mid_idx)]
        else:  # L'方向 (顺时针看左面)
            # 移动: 上 -> 后 -> 下 -> 前 -> 上
            destination = [(3, mid_idx), (1, mid_idx), (2, mid_idx), (0, mid_idx)]
            # 翻转后面的条带
            strips[3] = strips[3][::-1]
        
        # 应用旋转
        for i, (face, idx) in enumerate(destination):
            # 获取正确的源条带
            src_idx = (i - 1) % 4
            strip = strips[src_idx]
            
            # 如果是后面，需要翻转
            if face == 3:
                strip = strip[::-1]
                
            # 应用到目标面
            self.cube[face, :, idx] = strip
        
        # 现在我们什么都不做，因为我们在OLL/PLL中避免使用M层旋转
    
    def _random_last_layer_scramble(self, num_steps=2):
        """
        生成一个让"最后一层乱序、前两层保持还原"的打乱序列。
        num_steps 指用多少次OLL/PLL算法做叠加。
        返回所有应用的动作列表。
        """
        scramble_actions = []
        scramble_algs = []
        
        # 确保魔方是已解决状态
        self._reset_cube()
        
        # 尝试的最大次数
        max_attempts = 5
        attempt = 0
        valid_scramble = False
        
        while not valid_scramble and attempt < max_attempts:
            # 重置魔方为已解决状态
            self._reset_cube()
            scramble_actions = []
            scramble_algs = []
            
            # 尝试打乱
            for _ in range(num_steps):
                # 随机插入一个 U 方向的小转动
                u_move = random.choice(self.u_moves)
                if u_move:
                    scramble_algs.append(u_move)
                    scramble_actions.extend(self._parse_and_apply_algorithm(u_move))
                
                # 随机从 OLL 或 PLL 中选一个算法
                if random.random() < 0.5:
                    alg = random.choice(self.oll_algs)
                else:
                    alg = random.choice(self.pll_algs)
                
                scramble_algs.append(alg)
                scramble_actions.extend(self._parse_and_apply_algorithm(alg))
                
                # 检查前两层是否保持还原
                if not self._check_first_two_layers_solved():
                    # 前两层被破坏，尝试下一次
                    break
            
            # 最后检查前两层是否保持还原
            if self._check_first_two_layers_solved():
                valid_scramble = True
            else:
                attempt += 1
        
        # 如果所有尝试都失败，使用备用方法
        if not valid_scramble:
            self._reset_cube()
            scramble_actions = []
            scramble_algs = []
            
            # 只使用U层旋转，这肯定不会影响前两层
            for _ in range(5):  # 随机进行5次U层旋转
                u_action = random.choice([8, 9])  # U 或 U'
                u_notation = "U" if u_action == 8 else "U'"
                scramble_algs.append(u_notation)
                scramble_actions.append(u_action)
                self._apply_action(u_action)
        
        # 打印应用的算法
        if valid_scramble:
            print("应用的打乱算法:", " ".join(scramble_algs))
        else:
            print("由于前两层保持还原的约束，只使用了U层旋转:", " ".join(scramble_algs))
            
        return scramble_actions
    
    def _scramble_cube(self):
        """打乱魔方，记录打乱步骤以便后续对比"""
        self.scramble_actions = []
        
        # 如果需要随机打乱
        if self.scramble_moves > 0:
            for _ in range(self.scramble_moves):
                action = np.random.randint(0, 12)
                self.scramble_actions.append(action)
                self._apply_action(action)
            print(f"随机打乱了魔方 {self.scramble_moves} 步")
        # 特定打乱模式
        else:
            # 将魔方重置为已解决状态
            self._reset_cube()
            
            if self.use_oll_pll:
                # 使用OLL/PLL算法进行打乱
                print("使用OLL/PLL算法打乱魔方（保持前两层还原）")
                self.scramble_actions = self._random_last_layer_scramble(num_steps=2)
                
                # 验证前两层是否保持还原
                if not self._check_first_two_layers_solved():
                    print("警告：前两层未保持还原状态！这可能是由于算法实现问题。")
                    print("将尝试使用备用方法（仅U层旋转）。")
                    # 再次重置并使用简单的U层旋转
                    self._reset_cube()
                    self.scramble_actions = []
                    # 只使用U层旋转，这肯定不会影响前两层
                    for _ in range(5):  # 随机进行5次U层旋转
                        u_action = random.choice([8, 9])  # U 或 U'
                        self.scramble_actions.append(u_action)
                        self._apply_action(u_action)
                    print("已应用备用打乱方法：5次随机U层旋转")
            else:
                # 使用预设的特定动作打乱
                print("使用预设动作序列打乱魔方（保持前两层还原）")
                # 这里模拟了一个OLL情况（只有顶层打乱，前两层还原）
                specific_actions = [
                    8, 8,       # U U (顺时针旋转上层两次)
                    6, 8, 7,    # R U R' (顶层角块置换)
                    8,          # U (继续移动顶层)
                    4, 9, 5,    # L U' L' (顶层边块置换)
                    8, 8        # U U (顺时针旋转上层两次)
                ]
                
                for action in specific_actions:
                    self.scramble_actions.append(action)
                    self._apply_action(action)
                
                # 验证前两层是否保持还原
                if not self._check_first_two_layers_solved():
                    print("警告：前两层未保持还原状态！这可能是由于预设动作序列问题。")
            
            # 最终验证
            front_two_layers_status = self._check_first_two_layers_solved()
            print(f"前两层保持还原状态: {front_two_layers_status}")
            
            # 计算当前正确位置的块数量和百分比
            correct_pieces = 0
            for i in range(6):
                correct_pieces += np.sum(self.cube[i, :, :] == i)
            total_pieces = 6 * self.cube_size * self.cube_size
            correct_percentage = correct_pieces / total_pieces * 100
            print(f"正确位置的块数: {correct_pieces}/{total_pieces} ({correct_percentage:.1f}%)")
    
    def _apply_action(self, action):
        """应用动作到魔方状态"""
        face = action // 2  # 0=U, 1=D, 2=F, 3=B, 4=L, 5=R (基于动作映射0=F,1=F',2=B,3=B',...)
        direction = action % 2  # 0=顺时针, 1=逆时针
        
        # 创建魔方副本避免直接修改
        cube_copy = np.copy(self.cube)
        
        # 动作映射到标准面索引: 0=U, 1=L, 2=F, 3=R, 4=B, 5=D
        # 我们需要将动作面映射到标准面
        face_map = {0: 2, 1: 4, 2: 1, 3: 3, 4: 0, 5: 5}  # F->2, B->4, L->1, R->3, U->0, D->5
        std_face = face_map[face]
        
        # 旋转指定面
        if direction == 0:  # 顺时针
            self.cube[std_face] = np.rot90(self.cube[std_face], k=3)  # 顺时针旋转90度
        else:  # 逆时针
            self.cube[std_face] = np.rot90(self.cube[std_face], k=1)  # 逆时针旋转90度
        
        # 使用标准面索引: 0=U, 1=L, 2=F, 3=R, 4=B, 5=D
        # 定义相邻面的关系，表示每个动作会影响的边缘
        # 格式: (面索引, 行/列索引, 是否反转)
        adjacent_faces = {
            # 上面(U)旋转时影响前(F)右(R)后(B)左(L)面的上一行
            0: [(2, 0, False), (3, 0, False), (4, 0, False), (1, 0, False)],
            
            # 下面(D)旋转时影响前(F)左(L)后(B)右(R)面的底一行
            5: [(2, 2, False), (1, 2, False), (4, 2, False), (3, 2, False)],
            
            # 前面(F)旋转时影响上(U)右(R)下(D)左(L)面的相应行/列
            2: [(0, 2, False), (3, 0, True), (5, 0, True), (1, 2, False)],
            
            # 后面(B)旋转时影响上(U)左(L)下(D)右(R)面的相应行/列
            4: [(0, 0, True), (1, 0, False), (5, 2, False), (3, 2, True)],
            
            # 左面(L)旋转时影响上(U)前(F)下(D)后(B)面的相应列
            1: [(0, 0, False), (2, 0, False), (5, 0, True), (4, 2, True)],
            
            # 右面(R)旋转时影响上(U)后(B)下(D)前(F)面的相应列
            3: [(0, 2, True), (4, 0, True), (5, 2, False), (2, 2, False)]
        }
        
        # 获取当前面相邻的面
        adj_faces = adjacent_faces[std_face]
        
        # 调试信息
        if getattr(self, 'debug_mode', False):
            face_names = "ULFRBD"
            print(f"旋转面: {face_names[std_face]} (索引:{std_face}), 方向: {'顺时针' if direction == 0 else '逆时针'}")
            print(f"相邻面: {adj_faces}")
        
        # 准备用于旋转的条带
        strips = []
        for adj_face, idx, rev in adj_faces:
            # 获取行或列
            if idx < 3 and (adj_face == 0 or adj_face == 5 or 
                           (std_face == 0 and idx == 0) or 
                           (std_face == 5 and idx == 2)):  # 行
                strip = cube_copy[adj_face, idx, :].copy()
            else:  # 列
                strip = cube_copy[adj_face, :, idx].copy()
            
            if rev:
                strip = strip[::-1]
            strips.append(strip)
        
        # 根据旋转方向调整顺序
        if direction == 0:  # 顺时针
            strips = [strips[-1]] + strips[:-1]
        else:  # 逆时针
            strips = strips[1:] + [strips[0]]
        
        # 应用旋转到相邻面
        for i, (adj_face, idx, rev) in enumerate(adj_faces):
            strip = strips[i]
            if rev:
                strip = strip[::-1]
            
            # 应用到行或列
            if idx < 3 and (adj_face == 0 or adj_face == 5 or 
                          (std_face == 0 and idx == 0) or 
                          (std_face == 5 and idx == 2)):  # 行
                self.cube[adj_face, idx, :] = strip
            else:  # 列
                self.cube[adj_face, :, idx] = strip
    
    def is_solved(self):
        """检查魔方是否已解决"""
        for i in range(6):
            if not np.all(self.cube[i, :, :] == i):
                return False
        return True
    
    def _get_reward(self):
        """
        计算奖励函数
        返回基于正确位置的块数量的奖励
        """
        if self.is_solved():
            return 10.0  # 魔方已解决，给予高奖励
        
        # 计算正确位置的方块数量
        correct_pieces = 0
        for i in range(6):
            correct_pieces += np.sum(self.cube[i, :, :] == i)
        
        # 归一化奖励
        total_pieces = 6 * self.cube_size * self.cube_size
        normalized_reward = correct_pieces / total_pieces
        
        # 添加步数惩罚
        step_penalty = self.steps / self.max_steps * 0.1
        
        reward = normalized_reward - step_penalty
        
        # 记录最佳奖励
        if reward > self.best_reward:
            self.best_reward = reward
            
        return reward
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self._reset_cube()
        self._scramble_cube()
        self.steps = 0
        self.best_reward = -np.inf
        
        observation = self.cube.copy()
        info = {"scramble_actions": self.scramble_actions}
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        """执行一步动作"""
        self._apply_action(action)
        self.steps += 1
        
        observation = self.cube.copy()
        reward = self._get_reward()
        terminated = self.is_solved()
        truncated = self.steps >= self.max_steps
        
        info = {
            "steps": self.steps,
            "best_reward": self.best_reward,
            "solved": self.is_solved()
        }
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """渲染当前状态"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        """使用matplotlib渲染魔方的一帧"""
        if self.fig is None:
            self.fig, self.axs = plt.subplots(2, 3, figsize=(10, 6))
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            self.axs = self.axs.flatten()
        
        # 清除之前的图像
        for ax in self.axs:
            ax.clear()
        
        # 面的顺序：上(0)、左(1)、前(2)、右(3)、后(4)、下(5)
        face_names = ['Up (0)', 'Left (1)', 'Front (2)', 'Right (3)', 'Back (4)', 'Down (5)']
        
        # 绘制每个面
        for i, (face, name) in enumerate(zip(self.cube, face_names)):
            ax = self.axs[i]
            
            # 创建颜色网格
            cmap = colors.ListedColormap([self.color_map[j] for j in range(6)])
            norm = colors.BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
            
            # 绘制网格
            ax.imshow(face, cmap=cmap, norm=norm)
            
            # 添加网格线
            ax.grid(color='black', linestyle='-', linewidth=2)
            
            # 设置标题和刻度
            ax.set_title(name)
            ax.set_xticks(np.arange(-0.5, self.cube_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.cube_size, 1), minor=True)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 添加网格线
            ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
            
            # 在每个方块中央显示颜色代码
            for y in range(self.cube_size):
                for x in range(self.cube_size):
                    color = face[y, x]
                    ax.text(x, y, str(color), ha="center", va="center", fontsize=12)
        
        # 添加图例
        patches = [mpatches.Patch(color=color, label=f'{i}') 
                  for i, color in self.color_map.items()]
        self.fig.legend(handles=patches, loc='lower center', ncol=6)
        
        plt.tight_layout()
        plt.suptitle(f"Steps: {self.steps}, Solved: {self.is_solved()}")
        
        plt.draw()
        plt.pause(0.1)
        
        if self.render_mode == "rgb_array":
            # 将图表转换为RGB数组
            self.fig.canvas.draw()
            img = np.array(self.fig.canvas.renderer.buffer_rgba())
            return img
            
    def visualize_cube(self, save_path=None):
        """
        使用更详细的布局可视化魔方
        可选择保存为图像文件
        
        参数:
            save_path: 要保存图像的路径，如果为None则不保存
        """
        # 使用提供的可视化代码实现
        # 定义面的颜色
        CUBE_COLORS = {
            0: 'white',      # Up face
            1: 'orange',     # Left face
            2: 'green',      # Front face
            3: 'red',        # Right face
            4: 'blue',       # Back face
            5: 'yellow'      # Down face
        }
        
        # 创建新图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 移除坐标轴
        ax.set_axis_off()
        
        # 设置图的范围
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 9)
        
        # 定义面在图中的位置
        face_positions = {
            'Up': (3, 6),
            'Left': (0, 3),
            'Front': (3, 3),
            'Right': (6, 3),
            'Back': (9, 3),
            'Down': (3, 0)
        }
        
        face_names = ['Up', 'Left', 'Front', 'Right', 'Back', 'Down']
        
        # 绘制每个面
        for face_idx, face_name in enumerate(face_names):
            x_pos, y_pos = face_positions[face_name]
            
            # 添加面的标题
            ax.text(x_pos + 1.5, y_pos + 3.2, face_name, ha='center', fontsize=12)
            
            # 绘制每个方块
            for row in range(3):
                for col in range(3):
                    # 获取颜色值
                    color_idx = self.cube[face_idx, row, col]
                    color = CUBE_COLORS[color_idx]
                    
                    # 创建方块
                    rect = Rectangle((x_pos + col, y_pos + (2 - row)), 1, 1, 
                                    facecolor=color, edgecolor='black', linewidth=2)
                    
                    # 添加方块到图
                    ax.add_patch(rect)
                    
                    # 添加颜色索引文本
                    ax.text(x_pos + col + 0.5, y_pos + (2 - row) + 0.5, str(color_idx),
                           ha='center', va='center', fontsize=10)
        
        # 添加图例
        legend_elements = [Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', label=f"{idx}: {color}")
                        for idx, color in CUBE_COLORS.items()]
        ax.legend(handles=legend_elements, loc='upper right', title="Color Mapping")
        
        plt.tight_layout()
        title = f"Rubik's Cube - Steps: {self.steps}, Solved: {self.is_solved()}"
        plt.title(title, fontsize=16, pad=20)
        
        if save_path:
            plt.savefig(save_path)
            print(f"图像已保存到: {save_path}")
            
        plt.show()
        return fig
    
    def close(self):
        """关闭环境"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axs = None

    def _print_cube_state(self):
        """打印当前魔方状态，便于调试"""
        if not self.debug_mode:
            return
            
        face_names = ["U (上)", "D (下)", "F (前)", "B (后)", "L (左)", "R (右)"]
        color_names = ["白", "黄", "绿", "蓝", "橙", "红"]
        
        print("\n当前魔方状态:")
        for i, face in enumerate(self.cube):
            print(f"\n{face_names[i]} 面:")
            for row in face:
                # 将数字转换为颜色名称
                color_row = [color_names[int(val)] for val in row]
                print(color_row)
                
        # 检查魔方是否解决
        solved = self.is_solved()
        print(f"\n魔方是否已解决: {solved}")
        
        # 检查前两层是否解决
        f2l_solved = self._check_first_two_layers_solved()
        print(f"前两层是否已解决: {f2l_solved}")
        
        # 计算正确位置的块数量和百分比
        correct_pieces = 0
        total_pieces = self.cube_size * self.cube_size * 6
        
        # 创建一个已解决的魔方作为参考
        solved_cube = np.array([np.full((self.cube_size, self.cube_size), i) for i in range(6)])
        
        for i in range(6):
            correct_pieces += np.sum(self.cube[i] == solved_cube[i])
            
        percentage = (correct_pieces / total_pieces) * 100
        print(f"正确位置的块数量: {correct_pieces}/{total_pieces} ({percentage:.1f}%)") 