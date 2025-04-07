#!/usr/bin/env python3
"""
测试魔方环境
这个脚本只测试魔方环境的功能，不依赖DQN训练
"""
from rubiks_cube_env import RubiksCubeEnv
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import os

def test_visualize_magic_cube():
    """测试魔方环境的可视化"""
    print("\n===== 魔方环境可视化测试 =====")
    
    # 创建输出目录
    os.makedirs("cube_images/visualization", exist_ok=True)
    
    # 创建魔方环境并重置
    env = RubiksCubeEnv(cube_size=3, scramble_moves=0, use_oll_pll=False, debug_mode=True)
    env.reset()
    
    # 可视化初始状态
    print("初始状态 - 已解决的魔方")
    env.visualize_cube(save_path="cube_images/visualization/initial_state.png")
    
    # 执行一系列动作以显示不同状态
    actions = [
        {"name": "F (前面顺时针)", "action": 0},
        {"name": "R U R' (常见插入)", "actions": [6, 8, 7]},
        {"name": "F R U R' U' F' (OLL算法)", "actions": [0, 6, 8, 7, 9, 1]}
    ]
    
    # 重置魔方
    env.reset()
    
    # 执行和可视化第一个动作
    print(f"\n执行动作: {actions[0]['name']}")
    env.step(actions[0]['action'])
    env.visualize_cube(save_path="cube_images/visualization/action_F.png")
    
    # 重置魔方
    env.reset()
    
    # 执行和可视化第二个动作序列
    print(f"\n执行动作序列: {actions[1]['name']}")
    for action in actions[1]['actions']:
        env.step(action)
    env.visualize_cube(save_path="cube_images/visualization/sequence_R_U_R_prime.png")
    
    # 重置魔方
    env.reset()
    
    # 执行和可视化第三个动作序列
    print(f"\n执行算法: {actions[2]['name']}")
    for action in actions[2]['actions']:
        env.step(action)
    env.visualize_cube(save_path="cube_images/visualization/algorithm_F_R_U_R_prime_U_prime_F_prime.png")
    
    print("\n环境可视化测试完成。图像已保存到 cube_images/visualization 目录。")

def test_specific_algorithm():
    """测试特定的算法序列"""
    print("\n=== 测试特定的算法序列 ===")
    
    # 创建环境
    env = RubiksCubeEnv(
        render_mode="human",
        scramble_moves=0,
        use_oll_pll=True
    )
    
    # 重置环境
    observation, info = env.reset()
    print("魔方已打乱，现在尝试执行一个简单的顶层算法")
    time.sleep(2)
    
    # 反向执行原先的Sune算法: R' U2 R U R' U R
    # 动作对应: 7(R'), 8, 8(U2), 6(R), 8(U), 7(R'), 8(U), 6(R)
    algorithm = [7, 8, 8, 6, 8, 7, 8, 6]
    algorithm_notation = "R' U2 R U R' U R"
    
    print(f"执行反向算法: {algorithm_notation}")
    for i, action in enumerate(algorithm):
        action_names = ["F", "F'", "B", "B'", "L", "L'", "R", "R'", "U", "U'", "D", "D'"]
        print(f"  步骤 {i+1}: {action_names[action]}")
        
        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.5)
        
        if terminated:
            print("  魔方已解决!")
            break
    
    print("算法执行完毕，关闭环境")
    env.close()
    
def test_scramble_modes():
    """测试不同的打乱模式"""
    # 创建输出目录
    os.makedirs("cube_images/scramble_modes", exist_ok=True)
    
    print("\n===== 测试不同打乱模式 =====")
    
    # 测试1: 随机打乱3步
    env = RubiksCubeEnv(cube_size=3, scramble_moves=3, use_oll_pll=False, debug_mode=True)
    env.reset()
    print("\n使用3个随机动作打乱:")
    env.visualize_cube(save_path="cube_images/scramble_modes/random_3moves.png")
    
    # 测试2: 随机打乱10步
    env = RubiksCubeEnv(cube_size=3, scramble_moves=10, use_oll_pll=False, debug_mode=True)
    env.reset()
    print("\n使用10个随机动作打乱:")
    env.visualize_cube(save_path="cube_images/scramble_modes/random_10moves.png")
    
    # 测试3: OLL/PLL算法打乱
    env = RubiksCubeEnv(cube_size=3, scramble_moves=0, use_oll_pll=True, debug_mode=True)
    env.reset()
    print("\n使用OLL/PLL算法打乱:")
    env.visualize_cube(save_path="cube_images/scramble_modes/oll_pll.png")
    
    # 测试4: 不打乱
    env = RubiksCubeEnv(cube_size=3, scramble_moves=0, use_oll_pll=False, debug_mode=True)
    env.reset()
    print("\n不打乱魔方:")
    env.visualize_cube(save_path="cube_images/scramble_modes/no_scramble.png")

def test_check_first_two_layers():
    """测试检查前两层是否解决的逻辑"""
    # 创建输出目录
    os.makedirs("cube_images/first_two_layers", exist_ok=True)
    
    print("\n===== 测试前两层检查逻辑 =====")
    
    env = RubiksCubeEnv(cube_size=3, scramble_moves=0, use_oll_pll=False, debug_mode=True)
    env.reset()
    
    # 首先检查已解决的魔方
    print("\n初始状态 - 已解决的魔方:")
    is_f2l_solved = env._check_first_two_layers_solved()
    print(f"前两层是否已解决: {is_f2l_solved}")
    env.visualize_cube(save_path="cube_images/first_two_layers/initial_state.png")
    
    # 测试不同动作对前两层的影响
    test_actions = [
        {"name": "U (上面顺时针)", "action": 8, "should_preserve": True},
        {"name": "D (下面顺时针)", "action": 10, "should_preserve": False},
        {"name": "F (前面顺时针)", "action": 0, "should_preserve": False}
    ]
    
    for test in test_actions:
        # 重置魔方
        env.reset()
        
        # 执行动作
        print(f"\n执行动作: {test['name']}")
        env.step(test['action'])
        
        # 检查前两层
        is_f2l_solved = env._check_first_two_layers_solved()
        print(f"前两层是否保持不变: {is_f2l_solved}")
        
        # 验证结果是否符合预期
        if is_f2l_solved == test["should_preserve"]:
            print(f"正确: {test['name']} {'应该' if test['should_preserve'] else '不应该'}保持前两层不变")
        else:
            print(f"错误: {test['name']} {'应该' if test['should_preserve'] else '不应该'}保持前两层不变，但结果相反")
        
        # 可视化结果
        env.visualize_cube(save_path=f"cube_images/first_two_layers/after_{test['name'].split()[0]}.png")
    
    print("\n前两层检查逻辑测试完成。图像已保存到 cube_images/first_two_layers 目录。")

def test_action_reversibility():
    """测试动作的可逆性"""
    # 创建输出目录
    os.makedirs("cube_images/reversibility", exist_ok=True)
    
    print("\n===== 测试动作可逆性 =====")
    
    env = RubiksCubeEnv(cube_size=3, scramble_moves=0, use_oll_pll=False, debug_mode=True)
    
    action_pairs = [
        {"name": "F & F'", "actions": [0, 1]},
        {"name": "U & U'", "actions": [8, 9]},
        {"name": "R & R'", "actions": [6, 7]},
        {"name": "L & L'", "actions": [4, 5]},
        {"name": "B & B'", "actions": [2, 3]},
        {"name": "D & D'", "actions": [10, 11]}
    ]
    
    for pair in action_pairs:
        # 重置魔方
        env.reset()
        original_state = np.copy(env.cube)
        
        print(f"\n测试动作对: {pair['name']}")
        
        # 执行第一个动作
        print(f"执行动作1: {pair['name'].split('&')[0].strip()}")
        env.step(pair["actions"][0])
        
        # 保存中间状态
        env.visualize_cube(save_path=f"cube_images/reversibility/{pair['name'].split('&')[0].strip()}.png")
        
        # 执行第二个动作
        print(f"执行动作2: {pair['name'].split('&')[1].strip()}")
        env.step(pair["actions"][1])
        
        # 保存最终状态
        env.visualize_cube(save_path=f"cube_images/reversibility/{pair['name'].replace('&', 'and')}_final.png")
        
        # 检查是否回到原始状态
        is_same = np.array_equal(original_state, env.cube)
        
        if is_same:
            print(f"成功: {pair['name']} 是完全可逆的")
        else:
            print(f"失败: {pair['name']} 不能完全还原原始状态")
            mismatch_count = np.sum(original_state != env.cube)
            print(f"不匹配的位置数量: {mismatch_count}/{env.cube.size}")
    
    print("\n动作可逆性测试完成。图像已保存到 cube_images/reversibility 目录。")

def test_basic_actions():
    """测试所有基本动作的正确性，包括F, F', B, B', L, L', R, R', U, U', D, D'"""
    env = RubiksCubeEnv(cube_size=3, scramble_moves=0, use_oll_pll=False, debug_mode=True)
    env.reset()
    
    # 创建输出目录
    os.makedirs("cube_images", exist_ok=True)
    
    print("\n===== 测试所有基本动作 =====")
    
    # 动作映射: 0=F, 1=F', 2=B, 3=B', 4=L, 5=L', 6=R, 7=R', 8=U, 9=U', 10=D, 11=D'
    action_names = ["F", "F'", "B", "B'", "L", "L'", "R", "R'", "U", "U'", "D", "D'"]
    
    # 保存初始状态
    print("初始状态 - 已解决的魔方")
    env.visualize_cube(save_path="cube_images/initial_state.png")
    
    for action in range(12):
        # 重置魔方
        env.reset()
        
        # 执行动作
        print(f"\n执行动作: {action_names[action]} (动作索引: {action})")
        observation, reward, done, truncated, info = env.step(action)
        
        # 标准面索引: 0=U, 1=L, 2=F, 3=R, 4=B, 5=D
        print(f"标准面索引: 0=上(U), 1=左(L), 2=前(F), 3=右(R), 4=后(B), 5=下(D)")
        
        # 可视化动作后的魔方状态
        action_filename = action_names[action].replace("'", "p")
        env.visualize_cube(save_path=f"cube_images/action_{action_filename}.png")
        
        # 检查动作后的状态
        is_first_two_layers_solved = env._check_first_two_layers_solved()
        
        # 检查该动作是否应该保持第一和第二层不变
        # 只有U和U'（动作8和9）应该保持前两层不变
        should_preserve = action in [8, 9]
        
        if should_preserve and not is_first_two_layers_solved:
            print(f"错误: {action_names[action]} 应该保持前两层不变，但现在它们被改变了")
        elif not should_preserve and is_first_two_layers_solved:
            print(f"意外结果: {action_names[action]} 保持了前两层不变")
        else:
            expected_result = "保持前两层不变" if should_preserve else "改变前两层"
            actual_result = "前两层不变" if is_first_two_layers_solved else "前两层已改变"
            print(f"结果: {action_names[action]} 预期结果: {expected_result}, 实际结果: {actual_result}")
    
    print("\n所有动作测试完成。图像已保存到 cube_images 目录。")
    
    # 测试动作的可逆性：执行动作然后执行其反向动作应该回到原始状态
    print("\n===== 测试动作的可逆性 =====")
    for action in range(0, 12, 2):  # 只测试偶数索引动作，其反向动作是下一个索引
        # 重置魔方
        env.reset()
        original_state = np.copy(env.cube)
        
        # 执行动作和其反向动作
        print(f"\n测试: {action_names[action]} + {action_names[action+1]}")
        env.step(action)      # 执行动作
        
        # 保存中间状态
        action_filename = action_names[action].replace("'", "p")
        env.visualize_cube(save_path=f"cube_images/reverse_test_{action_filename}_step1.png")
        
        env.step(action + 1)  # 执行反向动作
        
        # 保存最终状态
        env.visualize_cube(save_path=f"cube_images/reverse_test_{action_filename}_step2.png")
        
        # 检查是否回到原始状态
        is_same = np.array_equal(original_state, env.cube)
        if is_same:
            print(f"成功: {action_names[action]} 和 {action_names[action+1]} 互为逆操作")
        else:
            print(f"失败: {action_names[action]} 和 {action_names[action+1]} 不能正确还原")
            
            # 计算不匹配的位置数量
            mismatch_count = np.sum(original_state != env.cube)
            print(f"  不匹配的位置数量: {mismatch_count}/{env.cube.size}")

def test_specific_algorithm():
    """测试特定算法 - 顺时针旋转上层角块 (Sune)"""
    env = RubiksCubeEnv(cube_size=3, scramble_moves=0, use_oll_pll=False, debug_mode=True)
    env.reset()
    
    # 创建输出目录
    os.makedirs("cube_images", exist_ok=True)
    
    print("\n===== 测试特定魔方算法 =====")
    
    # 保存初始状态
    print("初始状态 - 已解决的魔方")
    env.visualize_cube(save_path="cube_images/algorithm_initial.png")
    
    # 执行反转后的Sune公式 (R' U2 R U R' U R)
    # 映射到动作: 7, 8, 8, 6, 8, 7, 8, 6
    algorithm = [7, 8, 8, 6, 8, 7, 8, 6]  # R' U2 R U R' U R
    algorithm_name = "反转Sune算法 (R' U2 R U R' U R)"
    
    print(f"执行算法: {algorithm_name}")
    
    for i, action in enumerate(algorithm):
        action_names = ["F", "F'", "B", "B'", "L", "L'", "R", "R'", "U", "U'", "D", "D'"]
        print(f"步骤 {i+1}: {action_names[action]}")
        env.step(action)
        
        # 每两步保存一次状态
        if (i+1) % 2 == 0:
            env.visualize_cube(save_path=f"cube_images/algorithm_step{i+1}.png")
    
    # 保存最终状态
    print("算法执行后的状态:")
    env.visualize_cube(save_path="cube_images/algorithm_final.png")
    
    # 检查前两层是否保持不变
    is_f2l_solved = env._check_first_two_layers_solved()
    print(f"前两层是否保持不变: {is_f2l_solved}")

def test_full_algorithm_visualization():
    """完整测试一系列魔方算法，并生成每步的可视化图像"""
    env = RubiksCubeEnv(cube_size=3, scramble_moves=0, use_oll_pll=False, debug_mode=True)
    env.reset()
    
    # 创建输出目录
    os.makedirs("cube_images/full_algorithm", exist_ok=True)
    
    print("\n===== 完整算法测试和可视化 =====")
    
    # 保存初始状态
    print("初始状态 - 已解决的魔方")
    env.visualize_cube(save_path="cube_images/full_algorithm/0_initial.png")
    
    # 定义要测试的算法序列
    algorithms = [
        {"name": "F", "actions": [0]},                  # 单个前面旋转
        {"name": "U R U' R'", "actions": [8, 6, 9, 7]}, # 常见的插入公式
        {"name": "F R U R' U' F'", "actions": [0, 6, 8, 7, 9, 1]} # OLL算法
    ]
    
    step_counter = 1
    
    for alg in algorithms:
        print(f"\n执行算法: {alg['name']}")
        
        for action in alg['actions']:
            action_names = ["F", "F'", "B", "B'", "L", "L'", "R", "R'", "U", "U'", "D", "D'"]
            print(f"  执行动作: {action_names[action]}")
            env.step(action)
            
            # 保存每一步的状态
            action_filename = action_names[action].replace("'", "p")
            env.visualize_cube(save_path=f"cube_images/full_algorithm/{step_counter}_{action_filename}.png")
            step_counter += 1
        
        # 检查前两层状态
        is_f2l_solved = env._check_first_two_layers_solved()
        print(f"算法 '{alg['name']}' 执行后，前两层是否保持不变: {is_f2l_solved}")
    
    print("\n完整算法测试完成。所有图像已保存到 cube_images/full_algorithm 目录。")

if __name__ == "__main__":
    # 运行测试
    test_visualize_magic_cube()    # 测试环境可视化
    test_check_first_two_layers()  # 测试前两层检查逻辑
    test_action_reversibility()    # 测试动作可逆性
    test_basic_actions()           # 测试所有基本动作
    test_specific_algorithm()      # 测试特定算法
    test_scramble_modes()          # 测试不同打乱模式
    test_full_algorithm_visualization()  # 测试完整算法序列并可视化 