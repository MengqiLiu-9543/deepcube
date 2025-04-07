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
    
    # 测试3: OLL/PLL算法打乱（使用自定义算法）
    # 定义几种常用OLL/PLL算法
    oll_pll_algorithms = [
        {"name": "Sune (OLL)", "notation": "R U R' U R U2 R'", "moves": [6, 8, 7, 8, 6, 8, 8, 7]},
        {"name": "Anti-Sune (OLL)", "notation": "R' U' R U' R' U2 R", "moves": [7, 9, 6, 9, 7, 8, 8, 6]},
        {"name": "F R U R' U' F' (OLL)", "notation": "F R U R' U' F'", "moves": [0, 6, 8, 7, 9, 1]},
        {"name": "T-Perm (PLL)", "notation": "R U R' U' R' F R2 U' R' U' R U R' F'", "moves": [6, 8, 7, 9, 7, 0, 6, 6, 9, 7, 9, 6, 8, 7, 1]}
    ]
    
    # 选择要使用的算法
    selected_alg_idx = 2  # 选择F R U R' U' F'算法
    selected_alg = oll_pll_algorithms[selected_alg_idx]
    
    # 使用选定算法
    env = RubiksCubeEnv(
        cube_size=3, 
        scramble_moves=0,  # 这里设置为0，因为我们使用自定义算法
        use_oll_pll=True, 
        debug_mode=True,
        custom_algorithm=selected_alg["moves"]  # 使用动作列表
    )
    env.reset()
    
    print(f"\n使用{selected_alg['name']}算法打乱（{selected_alg['notation']}, {len(selected_alg['moves'])}步）:")
    env.visualize_cube(save_path=f"cube_images/scramble_modes/oll_pll_{selected_alg_idx}.png")
    
    # 测试4: 使用字符串格式的自定义算法
    string_algorithm = "R U R' U R U2 R'"  # Sune算法
    env = RubiksCubeEnv(
        cube_size=3,
        scramble_moves=0,
        use_oll_pll=True,
        debug_mode=True,
        custom_algorithm=string_algorithm  # 使用字符串格式
    )
    env.reset()
    
    print(f"\n使用字符串格式的Sune算法打乱（{string_algorithm}）:")
    env.visualize_cube(save_path="cube_images/scramble_modes/string_algorithm.png")
    
    # 测试5: 不打乱
    env = RubiksCubeEnv(cube_size=3, scramble_moves=0, use_oll_pll=False, debug_mode=True)
    env.reset()
    print("\n不打乱魔方:")
    env.visualize_cube(save_path="cube_images/scramble_modes/no_scramble.png")

def test_check_first_two_layers():
    """测试检查魔方前两层是否已解决的逻辑"""
    print("\n===== 测试前两层检查逻辑 =====")
    
    # 创建输出目录
    os.makedirs("cube_images/first_two_layers", exist_ok=True)
    
    # 初始化环境
    env = RubiksCubeEnv(cube_size=3, scramble_moves=0, use_oll_pll=False, debug_mode=True)
    env.reset()
    
    # 保存初始状态
    print("保存初始状态图像...")
    image_path = "cube_images/first_two_layers/initial_state.png"
    env.visualize_cube(save_path=image_path)
    print(f"初始状态图像已保存到: {image_path}")
    
    # 初始检查
    print("检查初始状态的前两层:")
    is_f2l_solved = env._check_first_two_layers_solved()
    print(f"前两层是否已解决: {is_f2l_solved}")
    
    # 测试不同动作对前两层的影响
    test_actions = {
        "U": 8,     # 只旋转顶层，前两层应该保持不变
        "D": 10,    # 旋转底层，会改变前两层
        "F": 0      # 旋转前面，会改变前两层
    }
    
    for action_name, action_idx in test_actions.items():
        # 重置魔方
        env.reset()
        
        # 执行动作
        print(f"\n执行动作: {action_name}")
        env.step(action_idx)
        
        # 保存状态
        image_path = f"cube_images/first_two_layers/after_{action_name}.png"
        print(f"保存动作后状态图像: {image_path}")
        env.visualize_cube(save_path=image_path)
        
        # 检查前两层
        is_f2l_solved = env._check_first_two_layers_solved()
        print(f"执行 {action_name} 后，前两层是否保持不变: {is_f2l_solved}")
        
        # 输出期望结果
        should_preserve = action_name == "U"
        if should_preserve and not is_f2l_solved:
            print(f"错误: {action_name} 应该保持前两层不变，但未能做到")
        elif not should_preserve and is_f2l_solved:
            print(f"意外结果: {action_name} 意外地保持了前两层不变")
        else:
            expected = "保持前两层不变" if should_preserve else "改变前两层"
            actual = "前两层不变" if is_f2l_solved else "前两层已改变"
            print(f"结果正确: {action_name} 预期 {expected}，实际 {actual}")
    
    print("\n前两层检查测试完成。图像已保存到 cube_images/first_two_layers 目录。")

def test_action_reversibility():
    """测试魔方动作的可逆性"""
    print("\n===== 测试魔方动作可逆性 =====")
    
    # 创建输出目录
    os.makedirs("cube_images/reversibility", exist_ok=True)
    
    # 初始化环境
    env = RubiksCubeEnv(cube_size=3, scramble_moves=0, use_oll_pll=False, debug_mode=True)
    
    # 动作名称
    action_names = ["F", "F'", "B", "B'", "L", "L'", "R", "R'", "U", "U'", "D", "D'"]
    
    # 测试每对动作的可逆性
    for action in range(0, 12, 2):  # 偶数索引的动作和它们的逆操作
        # 重置魔方
        env.reset()
        original_state = np.copy(env.cube)
        
        # 保存初始状态
        image_path = f"cube_images/reversibility/pair_{action}_initial.png"
        print(f"\n测试动作对: {action_names[action]} 和 {action_names[action+1]}")
        print(f"保存初始状态图像: {image_path}")
        env.visualize_cube(save_path=image_path)
        
        # 第一步：执行动作
        print(f"Step 1: 执行动作 {action_names[action]}")
        env.step(action)
        
        # 保存中间状态
        image_path = f"cube_images/reversibility/pair_{action}_after_first.png"
        print(f"保存中间状态图像: {image_path}")
        env.visualize_cube(save_path=image_path)
        
        # 第二步：执行逆动作
        print(f"Step 2: 执行逆动作 {action_names[action+1]}")
        env.step(action + 1)
        
        # 保存最终状态
        image_path = f"cube_images/reversibility/pair_{action}_after_both.png"
        print(f"保存最终状态图像: {image_path}")
        env.visualize_cube(save_path=image_path)
        
        # 验证是否恢复到原始状态
        is_same = np.array_equal(env.cube, original_state)
        if is_same:
            print(f"✓ 成功：{action_names[action]} 和 {action_names[action+1]} 是完美的逆操作对")
        else:
            # 计算多少块不在原始位置
            different_blocks = np.sum(env.cube != original_state)
            total_blocks = env.cube.size
            print(f"✗ 失败：{action_names[action]} 和 {action_names[action+1]} 未能恢复原始状态")
            print(f"  {different_blocks}/{total_blocks} 个块不在原始位置 ({different_blocks/total_blocks*100:.1f}%)")
    
    print("\n可逆性测试完成。所有测试图像已保存到 cube_images/reversibility 目录。")

def test_basic_actions():
    """测试所有基本魔方动作"""
    print("\n===== 测试所有基本魔方动作 =====")
    
    # 创建输出目录
    os.makedirs("cube_images/basic_actions", exist_ok=True)
    
    # 初始化环境
    env = RubiksCubeEnv(cube_size=3, scramble_moves=0, use_oll_pll=False, debug_mode=True)
    env.reset()
    
    # 初始状态
    print("保存初始状态图像...")
    env.visualize_cube(save_path="cube_images/basic_actions/initial_state.png")
    print("初始状态图像已保存到: cube_images/basic_actions/initial_state.png")
    
    # 定义动作映射
    action_names = ["F", "F'", "B", "B'", "L", "L'", "R", "R'", "U", "U'", "D", "D'"]
    
    # 测试每个基本动作
    for action in range(12):
        # 重置魔方
        env.reset()
        
        # 执行动作
        print(f"\n执行动作: {action_names[action]} (索引: {action})")
        env.step(action)
        
        # 可视化动作后的魔方状态
        action_filename = action_names[action].replace("'", "p")
        image_path = f"cube_images/basic_actions/action_{action_filename}.png"
        print(f"保存动作后状态图像: {image_path}")
        env.visualize_cube(save_path=image_path)
        
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
    
    print("\n所有动作测试完成。图像已保存到 cube_images/basic_actions 目录。")
    
    # 测试动作的可逆性：执行动作然后执行其反向动作应该回到原始状态
    print("\n===== 测试动作的可逆性 =====")
    for action in range(0, 12, 2):  # 只测试偶数索引动作，其反向动作是下一个索引
        # 重置魔方
        env.reset()
        original_state = np.copy(env.cube)
        
        # 执行动作和其反向动作
        print(f"\n测试: {action_names[action]} + {action_names[action+1]}")
        
        # 执行第一个动作
        print(f"第1步: 执行动作 {action_names[action]}")
        env.step(action)      # 执行动作
        
        # 保存中间状态
        action_filename = action_names[action].replace("'", "p")
        image_path = f"cube_images/basic_actions/reverse_test_{action_filename}_step1.png"
        print(f"保存中间状态图像: {image_path}")
        env.visualize_cube(save_path=image_path)
        
        # 执行反向动作
        print(f"第2步: 执行反向动作 {action_names[action+1]}")
        env.step(action + 1)  # 执行反向动作
        
        # 保存最终状态
        image_path = f"cube_images/basic_actions/reverse_test_{action_filename}_step2.png"
        print(f"保存最终状态图像: {image_path}")
        env.visualize_cube(save_path=image_path)
        
        # 检查是否回到原始状态
        is_same = np.array_equal(env.cube, original_state)
        if not is_same:
            print(f"错误: {action_names[action]} + {action_names[action+1]} 未能恢复原始状态!")
            # 计算有多少块不在正确位置
            different_blocks = np.sum(env.cube != original_state)
            print(f"有 {different_blocks} 个块不在正确位置。")
        else:
            print(f"成功: {action_names[action]} + {action_names[action+1]} 正确恢复了原始状态。")
    
    print("\n可逆性测试完成。所有图像已保存到 cube_images/basic_actions 目录。")

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

def test_initial_state():
    """详细测试魔方的初始状态"""
    print("\n===== 测试魔方初始状态 =====")
    
    # 创建输出目录
    os.makedirs("cube_images/initial_state", exist_ok=True)
    
    # 创建环境，确保不进行任何打乱
    env = RubiksCubeEnv(cube_size=3, scramble_moves=0, use_oll_pll=False, debug_mode=True)
    observation, info = env.reset()
    
    # 进一步确保我们在测试纯净的初始状态
    env._reset_cube()
    
    # 详细打印魔方各面状态
    print("魔方各面状态（应该每个面都只有一种颜色）:")
    face_names = ["上(U)", "左(L)", "前(F)", "右(R)", "后(B)", "下(D)"]
    
    is_solved = True
    for face_idx in range(6):
        face = env.cube[face_idx]
        expected_color = face_idx
        print(f"\n{face_names[face_idx]}面 (期望值全为{expected_color}):")
        print(face)
        
        # 检查该面是否所有块都是同一颜色
        if not np.all(face == expected_color):
            print(f"错误: {face_names[face_idx]}面有块不是正确颜色!")
            is_solved = False
    
    # 可视化初始状态
    env.visualize_cube(save_path="cube_images/initial_state/initial_state_detail.png")
    
    print(f"\n魔方初始状态是否正确还原: {is_solved}")
    
    if not is_solved:
        print("\n注意: 魔方初始状态未正确还原，这可能导致所有其他测试失败。")
        print("请检查rubiks_cube_env.py中的_reset_cube方法实现。")
    else:
        print("\n成功: 魔方初始状态已正确还原，每个面都有正确的颜色。")

if __name__ == "__main__":
    # 运行测试
    # 首先测试初始状态
    test_initial_state()          # 测试初始状态详情
    test_visualize_magic_cube()    # 测试环境可视化
    test_check_first_two_layers()  # 测试前两层检查逻辑
    test_action_reversibility()    # 测试动作可逆性
    test_basic_actions()           # 测试所有基本动作
    test_specific_algorithm()      # 测试特定算法
    test_scramble_modes()          # 测试不同打乱模式
    test_full_algorithm_visualization()  # 测试完整算法序列并可视化 