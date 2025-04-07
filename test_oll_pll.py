import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
import random
from matplotlib.patches import Rectangle
import os

class RubiksCubeEnv(gym.Env):
    """
    Rubik's Cube 环境
    
    一个模拟3x3魔方的环境，可以接受12种基本动作
    (F, F', B, B', L, L', R, R', U, U', D, D')
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    # 标准的OLL/PLL算法列表 (用于解决)
    oll_algs = [
        "R U2 R' U' R U' R'",     # OLL 27 (Sune)
        "R U R' U R U2 R'",
        "R' U' R U' R' U2 R",     # OLL 26 (Antisune)
        "R' U2 R U R' U R",
        "F R U R' U' F'",         # OLL 45
        "F U R U' R' F'",
        #"U R U2 R' U' R U' R'"    # OLL 27 with U 前缀 - 避免重复
        #"R U R' U R U2 R' U'"     # 重复
    ]

    pll_algs = [
        "R U' R U R U R U' R' U' R2",   # U-perm (a)
        "R2 U R U R' U' R' U' R' U R'", # U-perm (b)
        #"U R U' R U R U R U' R' U' R2"  # U-perm (a) with U 前缀 - 避免重复
        #"R2 U R U R' U' R' U' R' U R' U'" # 重复
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
        'M': 12,  # 中层 - 暂不直接支持
        "M'": 13, # 中层' - 暂不直接支持
        'M2': 14  # 中层2 - 暂不直接支持
    }
    
    # 逆动作名称映射，用于_reverse_algorithm
    reversed_action_names = {v: k for k, v in action_map.items()}

    def __init__(self, cube_size=3, render_mode=None, max_steps=50, scramble_moves=5, use_oll_pll=False, debug_mode=False, custom_algorithm=None):
        """
        初始化魔方环境
        
        参数:
        cube_size: 魔方的尺寸(3代表3x3x3魔方)
        render_mode: 渲染模式
        max_steps: 每个episode的最大步数
        scramble_moves: 初始打乱的动作数量
        use_oll_pll: 是否使用OLL/PLL的 *逆算法* 打乱(只影响最后一层)
        debug_mode: 是否打印详细信息
        custom_algorithm: 自定义算法，可以是字符串或动作列表，用于精确控制打乱
        """
        self.cube_size = cube_size
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.scramble_moves = scramble_moves
        self.use_oll_pll = use_oll_pll # True表示使用OLL/PLL的逆算法
        self.debug_mode = debug_mode
        self.custom_algorithm = custom_algorithm
        
        # 强制检查：如果提供了自定义算法，自动启用OLL/PLL模式 (即假设自定义算法也是用于顶层操作)
        if custom_algorithm is not None:
            self.use_oll_pll = True # 假设自定义算法也是作用于顶层
        
        # 窗口尺寸
        self.window_size = 512
        
        self.steps = 0
        
        # 动作空间：12个动作，对应6个面的顺/逆时针旋转
        # F, F', B, B', L, L', R, R', U, U', D, D'
        self.action_space = spaces.Discrete(12)
        
        # 状态空间：6个面，每个面是cube_size x cube_size的颜色值
        # 标准面索引: 0=U(白), 1=L(橙), 2=F(绿), 3=R(红), 4=B(蓝), 5=D(黄)
        self.observation_space = spaces.Box(
            low=0, high=5, shape=(6, cube_size, cube_size), dtype=np.int32
        )
        
        # 用于渲染的颜色映射 (标准配色)
        self.color_map = {
            0: 'white',    # 上面 (U)
            1: 'orange',   # 左面 (L)
            2: 'green',    # 前面 (F)
            3: 'red',      # 右面 (R)
            4: 'blue',     # 后面 (B)
            5: 'yellow'    # 下面 (D)
        }
        
        # 初始化魔方
        self._reset_cube()
        
        # 记录最佳奖励，用于调试和评估
        self.best_reward = -np.inf
        
        # 渲染相关
        self.fig = None
        self.axs = None
    
    def _reset_cube(self):
        """初始化已解决的魔方状态
        
        在已解决的状态中，每个面都应该有统一的颜色：
        - 0面(上U): 所有块为0 (白)
        - 1面(左L): 所有块为1 (橙)
        - 2面(前F): 所有块为2 (绿)
        - 3面(右R): 所有块为3 (红)
        - 4面(后B): 所有块为4 (蓝)
        - 5面(下D): 所有块为5 (黄)
        """
        self.cube = np.zeros((6, self.cube_size, self.cube_size), dtype=np.int32)
        for i in range(6):
            self.cube[i, :, :] = i
        
        if self.debug_mode:
            print("魔方已重置为初始已解决状态")
            self._print_cube_state() # 使用更新后的打印函数
    
    def _check_first_two_layers_solved(self):
        """
        检查魔方的前两层是否已解决。
        
        前两层定义为：
        1. 下面(D=5)的所有块
        2. 前(F=2)、后(B=4)、左(L=1)、右(R=3)面的下两行（不包括与上面(U=0)相邻的第一行）
        
        返回:
            bool: 如果前两层已解决则为True，否则为False
        """
        if self.debug_mode:
            print("\n检查前两层是否已解决:")
            print("标准面索引: 0=U(白), 1=L(橙), 2=F(绿), 3=R(红), 4=B(蓝), 5=D(黄)")
        
        # 1. 检查下面 (D=5) 是否完全还原 (黄色)
        if not np.all(self.cube[5] == 5):
            if self.debug_mode:
                print("下面(D)未还原:")
                print(self.cube[5])
            return False
        
        # 2. 检查四个侧面的底部两行是否还原 (L=橙, F=绿, R=红, B=蓝)
        for face_idx, face_name in zip([1, 2, 3, 4], ["左(L)", "前(F)", "右(R)", "后(B)"]):
            face_color = face_idx # 面的颜色等于面的索引
            for row in range(1, 3): # 检查行索引1和2 (底部两行)
                if not np.all(self.cube[face_idx, row, :] == face_color):
                    if self.debug_mode:
                        print(f"{face_name}面的第{row+1}行未还原 (期望: {face_color})")
                        print(f"实际值: {self.cube[face_idx, row, :]}")
                    return False
        
        if self.debug_mode:
            print("前两层已成功解决!")
        return True

    def _reverse_algorithm(self, algorithm):
        """
        反转一个魔方算法字符串。
        例如: "R U R'" -> "R U' R'"
        """
        moves = algorithm.split()
        moves.reverse() # 1. 翻转动作顺序
        
        reversed_moves = []
        for move in moves:
            # 2. 翻转每个动作
            if move.endswith("'"):
                # F' -> F
                reversed_moves.append(move[:-1])
            elif move.endswith("2"):
                # F2 不变
                reversed_moves.append(move)
            else:
                # F -> F'
                reversed_moves.append(move + "'")
                
        return " ".join(reversed_moves)

    def _parse_and_apply_algorithm(self, algorithm):
        """解析并应用魔方算法字符串，返回执行的动作列表"""
        moves = algorithm.split()
        actions = []
        action_names = self.reversed_action_names # 用于调试
        
        for move in moves:
            if move in self.action_map:
                action = self.action_map[move]
                if action < 12:  # 只应用标准12个动作
                    self._apply_action(action)
                    actions.append(action)
                    if self.debug_mode:
                         print(f"  应用解析动作: {move} (索引: {action})")
                # M层动作暂不处理
            elif len(move) == 2 and move[1] == '2':
                # 处理 U2, R2 等二阶动作
                base_move = move[0]
                if base_move in 'FBLRUD':
                    action = self.action_map[base_move]
                    self._apply_action(action)
                    self._apply_action(action)
                    actions.append(action)
                    actions.append(action)
                    if self.debug_mode:
                         print(f"  应用解析动作: {move} (执行两次 {base_move}, 索引: {action})")
            else:
                 if self.debug_mode:
                     print(f"  警告: 无法解析或应用动作 '{move}'")
                    
        return actions

    def _apply_m_slice(self):
        """M层旋转 - 目前是空操作，因为不保证 F2L 不变"""
        # M层旋转会影响F2L，在需要保持F2L时避免使用
        if self.debug_mode:
            print("警告: M层旋转被跳过，因为它会影响前两层。")
        pass # 保持为空操作
    
    def _random_last_layer_scramble(self, num_steps=2):
        """
        生成一个让"最后一层乱序、前两层保持还原"的打乱序列。
        通过应用 OLL/PLL 算法的 *逆算法* 来实现。
        num_steps 指用多少次逆OLL/PLL算法做叠加。
        返回所有应用的动作列表。
        """
        scramble_actions = []
        applied_reverse_algs = [] # 记录应用的逆算法名称
        
        # 确保魔方是已解决状态
        self._reset_cube()
        
        max_attempts = 10 # 增加尝试次数
        attempt = 0
        valid_scramble = False
        
        while not valid_scramble and attempt < max_attempts:
            # 重置魔方为已解决状态
            self._reset_cube()
            current_scramble_actions = []
            current_applied_reverse_algs = []
            
            f2l_preserved_throughout = True # 假设F2L全程保持

            # 尝试打乱 num_steps 次
            for step in range(num_steps):
                # 随机插入一个 U 方向的小转动 (U转不影响F2L)
                u_move = random.choice(self.u_moves)
                if u_move:
                    current_applied_reverse_algs.append(f"(U-move: {u_move})")
                    parsed_actions = self._parse_and_apply_algorithm(u_move)
                    current_scramble_actions.extend(parsed_actions)
                    # U转理论上不影响F2L，但为保险起见可以检查
                    if not self._check_first_two_layers_solved():
                         if self.debug_mode: print(f"警告：U转 ({u_move}) 后F2L被破坏？")
                         f2l_preserved_throughout = False
                         break # 如果U转破坏了F2L，说明有基础问题

                # 随机从 OLL 或 PLL 中选一个 *解决* 算法
                if random.random() < 0.5:
                    alg = random.choice(self.oll_algs)
                    alg_type = "OLL"
                else:
                    alg = random.choice(self.pll_algs)
                    alg_type = "PLL"
                
                # 获取该算法的 *逆算法*
                reversed_alg = self._reverse_algorithm(alg)
                current_applied_reverse_algs.append(f"Rev({alg_type}:{alg}) -> {reversed_alg}")
                
                # 解析并应用 *逆算法*
                if self.debug_mode: print(f"步骤 {step+1}: 应用逆算法: {reversed_alg} (来自 {alg_type}: {alg})")
                parsed_actions = self._parse_and_apply_algorithm(reversed_alg)
                current_scramble_actions.extend(parsed_actions)
                
                # **关键检查**: 在应用每个逆算法后检查 F2L 是否保持还原
                if not self._check_first_two_layers_solved():
                    if self.debug_mode: 
                        print(f"错误: 应用逆算法 '{reversed_alg}' 后，F2L 被破坏了！")
                        print(f"       原始算法是: {alg}")
                    f2l_preserved_throughout = False
                    break # F2L 被破坏，停止当前尝试

            # 如果循环正常结束 (没有break)，并且最终 F2L 保持还原
            if f2l_preserved_throughout: # 检查标志
                 # 再次最终确认
                 if self._check_first_two_layers_solved():
                     valid_scramble = True
                     scramble_actions = current_scramble_actions
                     applied_reverse_algs = current_applied_reverse_algs
                     if self.debug_mode: print("成功找到保持F2L的打乱序列。")
                 else:
                      if self.debug_mode: print(f"尝试 {attempt+1} 失败: 最终F2L检查未通过。")
                      attempt += 1
            else:
                 if self.debug_mode: print(f"尝试 {attempt+1} 失败: F2L在过程中被破坏。")
                 attempt += 1
        
        # 如果所有尝试都失败，使用备用方法 (只用U转)
        if not valid_scramble:
            if self.debug_mode: print(f"\n所有 {max_attempts} 次尝试均未能保证F2L，回退到仅使用U层旋转。")
            self._reset_cube()
            scramble_actions = []
            applied_reverse_algs = []
            
            for i in range(5): # 进行5次随机U层旋转
                u_action = random.choice([8, 9]) # U 或 U'
                u_notation = "U" if u_action == 8 else "U'"
                applied_reverse_algs.append(f"(Fallback U-move: {u_notation})")
                scramble_actions.append(u_action)
                self._apply_action(u_action)
            
            # U转肯定保持F2L
            if not self._check_first_two_layers_solved() and self.debug_mode:
                 print("严重错误：仅U转后F2L未解决！")

        # 打印应用的算法 (现在是逆算法或U转)
        if self.debug_mode:
             print("\n最终应用的打乱序列:")
             for alg_str in applied_reverse_algs:
                 print(f"  - {alg_str}")
        
        self.scramble_actions = scramble_actions # 保存最终的动作列表
        return self.scramble_actions
    
    
    
    def _scramble_cube(self):
        """打乱魔方，记录打乱步骤以便后续对比"""
        self.scramble_actions = []
        action_names = ["F", "F'", "B", "B'", "L", "L'", "R", "R'", "U", "U'", "D", "D'"] # 用于调试
        
        # 检查是否有自定义算法
        if self.custom_algorithm is not None:
            self._reset_cube()
            if self.debug_mode: print("\n使用自定义算法打乱魔方")
            
            if isinstance(self.custom_algorithm, str):
                 if self.debug_mode: print(f"应用自定义算法: {self.custom_algorithm}")
                 # 注意：这里假设自定义算法是直接应用的，而不是逆算法
                 # 如果需要自定义算法也反转，需要额外逻辑
                 self.scramble_actions = self._parse_and_apply_algorithm(self.custom_algorithm)
            elif isinstance(self.custom_algorithm, list):
                 if self.debug_mode: print(f"应用自定义动作列表: {[action_names[a] for a in self.custom_algorithm]}")
                 for action in self.custom_algorithm:
                     if 0 <= action < 12:
                         self.scramble_actions.append(action)
                         self._apply_action(action)
            
            # 如果使用OLL/PLL模式，验证自定义算法是否破坏F2L
            if self.use_oll_pll and not self._check_first_two_layers_solved():
                print("\n警告：自定义算法破坏了前两层！")
            
            return self.scramble_actions

        # 如果不是自定义算法
        self._reset_cube() # 确保从解决状态开始

        if self.use_oll_pll:
             # 使用 OLL/PLL 的 *逆算法* 进行打乱
             if self.debug_mode: print("\n使用 OLL/PLL 的逆算法打乱魔方（尝试保持前两层还原）")
             self.scramble_actions = self._random_last_layer_scramble(num_steps=2) # 调用使用逆算法的函数

        elif self.scramble_moves > 0:
             # 使用指定步数的随机动作打乱 (会破坏F2L)
             if self.debug_mode: print(f"\n使用 {self.scramble_moves} 步随机动作打乱魔方 (会影响所有层)")
             for _ in range(self.scramble_moves):
                 action = np.random.randint(0, 12)
                 self.scramble_actions.append(action)
                 self._apply_action(action)
        else:
             # 如果 scramble_moves=0 且 use_oll_pll=False，则不打乱
             if self.debug_mode: print("\n不进行打乱，保持魔方已解决状态。")
             self.scramble_actions = []

        # 最终验证 (如果进行了打乱)
        if self.debug_mode and self.scramble_actions:
             final_f2l_status = self._check_first_two_layers_solved()
             print(f"\n打乱完成后，前两层保持还原状态: {final_f2l_status}")
             
             correct_pieces = 0
             total_pieces = 6 * self.cube_size * self.cube_size
             for i in range(6):
                 correct_pieces += np.sum(self.cube[i] == i)
             correct_percentage = correct_pieces / total_pieces * 100
             print(f"正确位置的块数: {correct_pieces}/{total_pieces} ({correct_percentage:.1f}%)")
             
             actions_str = " ".join([action_names[a] for a in self.scramble_actions])
             print(f"最终应用的动作序列: {actions_str}")
        
        return self.scramble_actions
    
    def _apply_action(self, action):
        """应用动作到魔方状态 (核心旋转逻辑)"""
        action_to_face = {0: 2, 1: 2, 2: 4, 3: 4, 4: 1, 5: 1, 6: 3, 7: 3, 8: 0, 9: 0, 10: 5, 11: 5}
        direction = 1 if action % 2 == 0 else 3
        face = action_to_face[action]
        
        if self.debug_mode:
            face_names = "ULFRBD"; dir_name = "顺时针" if direction == 1 else "逆时针"
            #print(f"  执行旋转: {face_names[face]}({face}) {dir_name}") # 在parse_and_apply中打印更清晰

        cube_copy = np.copy(self.cube)
        self.cube[face] = np.rot90(self.cube[face], k=direction)
        
        adjacent_faces = {
            0: [(2, 0, False), (3, 0, False), (4, 0, False), (1, 0, False)], # U -> F,R,B,L (顶行)
            5: [(2, 2, False), (1, 2, False), (4, 2, False), (3, 2, False)], # D -> F,L,B,R (底行)
            2: [(0, 2, False), (3, 0, True), (5, 0, False), (1, 2, True)],  # F -> U(底), R(左,rev), D(顶), L(右,rev) -- 修正: D(顶)不反转, L(右)反转
            4: [(0, 0, False), (1, 0, True), (5, 2, True), (3, 2, False)], # B -> U(顶), L(左,rev), D(底,rev), R(右) -- 修正: U(顶)不反转, D(底)反转
            1: [(0, 0, False), (4, 2, True), (5, 0, False), (2, 0, True)],  # L -> U(左), B(右,rev), D(左), F(左,rev) -- 修正: D(左)不反转
            3: [(0, 2, False), (2, 2, True), (5, 2, False), (4, 0, True)]   # R -> U(右), F(右,rev), D(右), B(左,rev) -- 修正: D(右)不反转
        }
        # --- 再次仔细检查和修正 adjacent_faces ---
        # F(2): 影响 U(0)底行， R(3)左列(需反转)， D(5)顶行(需反转)， L(1)右列(需反转)
        # B(4): 影响 U(0)顶行， L(1)左列(需反转)， D(5)底行(需反转)， R(3)右列(需反转)
        # L(1): 影响 U(0)左列， F(2)左列， D(5)左列， B(4)右列(需反转)
        # R(3): 影响 U(0)右列， B(4)左列(需反转)， D(5)右列， F(2)右列
        # 修正后的 adjacent_faces (基于标准视图和右手定则)
        adjacent_faces = {
             0: [(4, 0, False), (3, 0, False), (2, 0, False), (1, 0, False)], # U -> B, R, F, L (顶行, 逆时针看) - 修正顺序匹配顺时针旋转
             5: [(1, 2, False), (2, 2, False), (3, 2, False), (4, 2, False)], # D -> L, F, R, B (底行, 逆时针看) - 修正顺序匹配顺时针旋转
             2: [(0, 2, False), (3, 0, False), (5, 0, False), (1, 2, True)],  # F -> U(底), R(左), D(顶), L(右, rev)
             4: [(0, 0, False), (1, 0, False), (5, 2, False), (3, 2, True)],  # B -> U(顶), L(左), D(底), R(右, rev)
             1: [(0, 0, False), (2, 0, False), (5, 0, False), (4, 2, True)],  # L -> U(左), F(左), D(左), B(右, rev)
             3: [(0, 2, False), (4, 0, True), (5, 2, False), (2, 2, False)]   # R -> U(右), B(左, rev), D(右), F(右)
        }
        # --- 最终确认 adjacent_faces (再次核对常见资料) ---
        # U(0): B(4,0), R(3,0), F(2,0), L(1,0) -> OK
        # D(5): L(1,2), F(2,2), R(3,2), B(4,2) -> OK
        # F(2): U(0,2), R(3,:,0), D(5,0), L(1,:,2) -> R列需要从上到下读，所以L列也是。D行需反转，L列需反转?
        #       标准操作: U底->R左, R左->D顶, D顶->L右, L右->U底
        #       U底行 -> R左列 (方向OK)
        #       R左列 -> D顶行 (方向OK, 但D顶行需反转?)
        #       D顶行 -> L右列 (方向OK, 但L右列需反转?)
        #       L右列 -> U底行 (方向OK?)
        # 让我们遵循维基百科或类似来源的精确映射:
        # F: U(2,:) -> R(:,0); R(:,0) -> D(0,:); D(0,:) -> L(:,2); L(:,2) -> U(2,:)
        #    Indices: U(0,2,:), R(3,:,0), D(5,0,:), L(1,:,2)
        #    Need reversal? R(3,:,0) NO, D(5,0,:) YES, L(1,:,2) YES?
        # F: [(0, 2, False), (3, 0, False), (5, 0, True), (1, 2, True)] # 假设D顶行反转, L右列反转
        # B: U(0,:) -> L(:,0); L(:,0) -> D(2,:); D(2,:) -> R(:,2); R(:,2) -> U(0,:)
        #    Indices: U(0,0,:), L(1,:,0), D(5,2,:), R(3,:,2)
        #    Need reversal? L(1,:,0) YES, D(5,2,:) YES, R(3,:,2) NO?
        # B: [(0, 0, False), (1, 0, True), (5, 2, True), (3, 2, False)] # 假设L左列反转, D底行反转
        # L: U(:,0) -> F(:,0); F(:,0) -> D(:,0); D(:,0) -> B(:,2); B(:,2) -> U(:,0)
        #    Indices: U(0,:,0), F(2,:,0), D(5,:,0), B(4,:,2)
        #    Need reversal? F(2,:,0) NO, D(5,:,0) NO, B(4,:,2) YES?
        # L: [(0, 0, False), (2, 0, False), (5, 0, False), (4, 2, True)] # 假设B右列反转
        # R: U(:,2) -> B(:,0); B(:,0) -> D(:,2); D(:,2) -> F(:,2); F(:,2) -> U(:,2)
        #    Indices: U(0,:,2), B(4,:,0), D(5,:,2), F(2,:,2)
        #    Need reversal? B(4,:,0) YES, D(5,:,2) NO, F(2,:,2) NO?
        # R: [(0, 2, False), (4, 0, True), (5, 2, False), (2, 2, False)] # 假设B左列反转

        # 使用最终确认的 adjacent_faces
        adjacent_faces = {
             0: [(4, 0, False), (3, 0, False), (2, 0, False), (1, 0, False)], # U -> B,R,F,L (顶行)
             5: [(1, 2, False), (2, 2, False), (3, 2, False), (4, 2, False)], # D -> L,F,R,B (底行)
             2: [(0, 2, False), (3, 0, False), (5, 0, True),  (1, 2, True) ], # F -> U(底), R(左), D(顶,rev), L(右,rev)
             4: [(0, 0, False), (1, 0, True),  (5, 2, True),  (3, 2, False)], # B -> U(顶), L(左,rev), D(底,rev), R(右)
             1: [(0, 0, False), (2, 0, False), (5, 0, False), (4, 2, True) ], # L -> U(左), F(左), D(左), B(右,rev)
             3: [(0, 2, False), (4, 0, True),  (5, 2, False), (2, 2, False)]  # R -> U(右), B(左,rev), D(右), F(右)
        }

        adj_faces = adjacent_faces[face]
        strips = []
        for adj_face, idx_spec, rev in adj_faces:
            if isinstance(idx_spec, int): # 行或列索引
                is_row = (face in [0, 5]) or (adj_face in [0, 5]) # 粗略判断是行还是列
                if face in [2, 4]: is_row = (adj_face in [0, 5])
                if face in [1, 3]: is_row = False # L/R 旋转侧面总是影响列

                if is_row:
                    strip = cube_copy[adj_face, idx_spec, :].copy()
                else:
                    strip = cube_copy[adj_face, :, idx_spec].copy()
            else: # 应该是切片 (未在此版本中使用，保持简单)
                 continue # 跳过

            if rev:
                strip = strip[::-1]
            strips.append(strip)
        
        if direction == 1: # 顺时针
            strips = [strips[-1]] + strips[:-1]
        else: # 逆时针
            strips = strips[1:] + [strips[0]]
        
        for i, (adj_face, idx_spec, rev) in enumerate(adj_faces):
            strip = strips[i]
            if rev:
                strip = strip[::-1]

            if isinstance(idx_spec, int):
                is_row = (face in [0, 5]) or (adj_face in [0, 5])
                if face in [2, 4]: is_row = (adj_face in [0, 5])
                if face in [1, 3]: is_row = False

                if is_row:
                    self.cube[adj_face, idx_spec, :] = strip
                else:
                    self.cube[adj_face, :, idx_spec] = strip
            else:
                 continue # 跳过

        # # 调试：检查逆操作是否恢复原状 (可选)
        # if self.debug_mode and action % 2 == 0:
        #     inv_action = action + 1
        #     temp_cube_before_inv = np.copy(self.cube)
        #     self._apply_action(inv_action) # 应用逆动作
        #     if not np.array_equal(self.cube, cube_copy):
        #          print(f"错误: 动作 {action} 和其逆动作 {inv_action} 不可逆!")
        #          # 可以选择恢复到应用逆操作前的状态
        #          self.cube = temp_cube_before_inv
        #     else:
        #          # 测试通过，需要恢复到原状态（应用正向动作后的状态）
        #          self.cube = temp_cube_before_inv
        #          if self.debug_mode: print(f"动作 {action} 与逆动作 {inv_action} 可逆性验证通过。")
        return
    
    def is_solved(self):
        """检查魔方是否已解决"""
        for i in range(6):
            if not np.all(self.cube[i] == i): return False
        return True
    
    def _get_reward(self):
        """计算奖励函数 (简化版)"""
        if self.is_solved(): return 10.0
        correct_pieces = 0
        for i in range(6): correct_pieces += np.sum(self.cube[i] == i)
        total_pieces = 6 * self.cube_size * self.cube_size
        normalized_reward = correct_pieces / total_pieces
        step_penalty = self.steps / self.max_steps * 0.1 # 轻微步数惩罚
        reward = normalized_reward - step_penalty
        if reward > self.best_reward: self.best_reward = reward
        return reward
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self._reset_cube() # 重置为已解决
        
        # 打乱魔方 (根据设置)
        self._scramble_cube() 
        
        self.steps = 0
        self.best_reward = -np.inf
        observation = self.cube.copy()
        info = {"scramble_actions": getattr(self, 'scramble_actions', [])} # 确保scramble_actions存在
        if self.render_mode == "human": self._render_frame()
        return observation, info
    
    def step(self, action):
        """执行一步动作"""
        self._apply_action(action)
        self.steps += 1
        observation = self.cube.copy()
        reward = self._get_reward()
        terminated = self.is_solved()
        truncated = self.steps >= self.max_steps
        info = {"steps": self.steps, "best_reward": self.best_reward, "solved": terminated}
        if self.render_mode == "human": self._render_frame()
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """渲染当前状态"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
             self._render_frame() # 在step中调用，这里可以空置或只处理rgb_array
    
    def _render_frame(self):
        """使用matplotlib渲染魔方的一帧"""
        if self.render_mode not in ["human", "rgb_array"]: return
        
        # 标准面索引: 0=U(白), 1=L(橙), 2=F(绿), 3=R(红), 4=B(蓝), 5=D(黄)
        # 面名称映射到索引
        face_name_to_idx = {'Up': 0, 'Left': 1, 'Front': 2, 'Right': 3, 'Back': 4, 'Down': 5}
        
        if self.fig is None or self.axs is None:
            # 调整布局以匹配常见魔方展开图 (十字形)
            self.fig, self.axs = plt.subplots(3, 4, figsize=(12, 9))
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            # 隐藏不需要的格子
            for ax in self.axs.flat: ax.set_visible(False)
            # 定义哪些格子可见及其对应的面
            self.ax_map = {
                (0, 1): face_name_to_idx['Up'],    # R0, C1 -> 上
                (1, 0): face_name_to_idx['Left'],  # R1, C0 -> 左
                (1, 1): face_name_to_idx['Front'], # R1, C1 -> 前
                (1, 2): face_name_to_idx['Right'], # R1, C2 -> 右
                (1, 3): face_name_to_idx['Back'],  # R1, C3 -> 后
                (2, 1): face_name_to_idx['Down']   # R2, C1 -> 下
            }
            for (r, c), face_idx in self.ax_map.items():
                self.axs[r, c].set_visible(True)

        # 绘制每个可见的面
        for (r, c), face_idx in self.ax_map.items():
            ax = self.axs[r, c]
            ax.clear() # 清除之前的绘制内容
            face_data = self.cube[face_idx]
            
            cmap = colors.ListedColormap([self.color_map[j] for j in range(6)])
            norm = colors.BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
            
            ax.imshow(face_data, cmap=cmap, norm=norm, interpolation='nearest')
            ax.grid(color='black', linestyle='-', linewidth=2)
            ax.set_xticks(np.arange(-0.5, self.cube_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.cube_size, 1), minor=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
            
            # 设置标题 (可选)
            face_names_list = ['Up(0)', 'Left(1)', 'Front(2)', 'Right(3)', 'Back(4)', 'Down(5)']
            ax.set_title(face_names_list[face_idx], fontsize=10)

            # # 在每个方块中央显示颜色代码 (调试用)
            # for y in range(self.cube_size):
            #     for x in range(self.cube_size):
            #         color_idx = face_data[y, x]
            #         ax.text(x, y, str(color_idx), ha="center", va="center", fontsize=8, color='black' if color_idx != 5 else 'white')

        plt.suptitle(f"Steps: {self.steps}, Solved: {self.is_solved()}", fontsize=14)
        plt.draw()
        plt.pause(0.01) # 暂停一小段时间以便显示更新

        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            img_data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_data = img_data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img_data
            
    def visualize_cube(self, save_path=None):
        """使用详细布局可视化魔方并可选保存"""
        
        # 标准面索引: 0=U(白), 1=L(橙), 2=F(绿), 3=R(红), 4=B(蓝), 5=D(黄)
        CUBE_COLORS = self.color_map # 使用实例的颜色映射
        
        FACE_NAMES_CN = {0: '上(U)', 1: '左(L)', 2: '前(F)', 3: '右(R)', 4: '后(B)', 5: '下(D)'}
        COLOR_NAMES_CN = {0: '白', 1: '橙', 2: '绿', 3: '红', 4: '蓝', 5: '黄'}

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_axis_off()
        ax.set_xlim(0, 12); ax.set_ylim(0, 9)

        # 十字展开图位置
        face_positions = {'Up': (3, 6), 'Left': (0, 3), 'Front': (3, 3), 
                          'Right': (6, 3), 'Back': (9, 3), 'Down': (3, 0)}
        face_indices = {'Up': 0, 'Left': 1, 'Front': 2, 'Right': 3, 'Back': 4, 'Down': 5}

        for face_name, (x_pos, y_pos) in face_positions.items():
            face_idx = face_indices[face_name]
            ax.text(x_pos + 1.5, y_pos + 3.2, f"{face_name} ({FACE_NAMES_CN[face_idx]})", ha='center', fontsize=12)
            
            for row in range(3):
                for col in range(3):
                    color_idx = int(self.cube[face_idx, row, col]) # 确保是整数
                    color = CUBE_COLORS[color_idx]
                    rect = Rectangle((x_pos + col, y_pos + (2 - row)), 1, 1, 
                                     facecolor=color, edgecolor='black', linewidth=1.5)
                    ax.add_patch(rect)
                    # # 显示颜色中文名 (可选)
                    # ax.text(x_pos + col + 0.5, y_pos + (2 - row) + 0.5, 
                    #        f"{COLOR_NAMES_CN[color_idx]}",
                    #        ha='center', va='center', fontsize=9, color='black' if color_idx != 5 else 'white') # 黄色用白色字

        legend_elements = [Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', 
                                     label=f"{idx}: {color} ({COLOR_NAMES_CN[idx]})")
                           for idx, color in CUBE_COLORS.items()]
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, title="颜色表")
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # 调整布局防止图例重叠
        title = f"魔方状态 - 步数: {self.steps}, 是否解决: {self.is_solved()}"
        plt.title(title, fontsize=16, pad=20)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            if self.debug_mode: print(f"图像已保存到: {save_path}")
            
        plt.close(fig) # 关闭图形，释放内存

        # 输出魔方状态信息（便于调试）
        if self.debug_mode: self._print_cube_state()
        
        # 注意：此函数现在不返回fig对象，如果需要请修改
    
    def close(self):
        """关闭环境"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axs = None

    def _print_cube_state(self):
        """打印当前魔方状态，便于调试"""
        if not self.debug_mode: return
            
        # 标准面索引: 0=U(白), 1=L(橙), 2=F(绿), 3=R(红), 4=B(蓝), 5=D(黄)
        FACE_NAMES_CN = {0: '上(U)', 1: '左(L)', 2: '前(F)', 3: '右(R)', 4: '后(B)', 5: '下(D)'}
        COLOR_NAMES_CN = {0: '白', 1: '橙', 2: '绿', 3: '红', 4: '蓝', 5: '黄'}
        
        print("\n--- 当前魔方状态 ---")
        for i in range(6):
             print(f"\n{FACE_NAMES_CN[i]} 面:")
             face_data = self.cube[i]
             for row in face_data:
                 color_row = [COLOR_NAMES_CN[int(val)] for val in row]
                 print(f"  {color_row}")
                
        solved = self.is_solved()
        print(f"\n魔方是否已解决: {solved}")
        f2l_solved = self._check_first_two_layers_solved() # 调用检查函数
        # print(f"前两层是否已解决: {f2l_solved}") # 在check函数内部已有打印

        correct_pieces = 0; total_pieces = 54
        for i in range(6): correct_pieces += np.sum(self.cube[i] == i)
        percentage = (correct_pieces / total_pieces) * 100
        print(f"正确位置块数: {correct_pieces}/{total_pieces} ({percentage:.1f}%)")
        print("--------------------\n")