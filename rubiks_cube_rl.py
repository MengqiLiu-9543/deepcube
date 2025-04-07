import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from collections import deque
import time
import matplotlib.pyplot as plt
import os

from rubiks_cube_env import RubiksCubeEnv

# 设置随机种子以便复现结果
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# 创建DQN代理
class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=20000)  # 记忆回放缓冲区
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.learning_rate = 0.001  # 学习率
        self.update_target_counter = 0  # 目标网络更新计数器
        self.update_target_frequency = 10  # 目标网络更新频率
        self.batch_size = 32  # 批量大小
        
        # 创建神经网络模型
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """构建神经网络模型"""
        model = Sequential()
        
        # 首先将输入重塑为正确的形状
        # 原始输入形状是(6, 3, 3)，代表6个面，每个面是3x3的魔方
        model.add(Reshape((6, 3, 3, 1), input_shape=self.state_shape))
        
        # 将6个面视为6个独立的通道
        # 这里需要使用Conv3D而不是Conv2D，因为我们有3D输入
        # 或者我们可以先展平6个面然后使用Dense层
        model.add(Flatten())  # 展平为 (6*3*3*1) = 54 个特征
        
        # 使用全连接层
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        
        # 输出层
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss=Huber(), optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def update_target_model(self):
        """更新目标网络的权重"""
        self.target_model.set_weights(self.model.get_weights())
        print("目标网络已更新")
    
    def remember(self, state, action, reward, next_state, done):
        """将经验存储到记忆缓冲区"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """根据当前状态选择动作"""
        if training and np.random.rand() <= self.epsilon:
            # 探索：随机选择动作
            return random.randrange(self.action_size)
        
        # 利用：选择Q值最高的动作
        # 不再需要添加额外的维度，直接使用原始状态
        state = np.expand_dims(state, axis=0)  # 添加批处理维度
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        """从记忆缓冲区中抽取经验并训练模型"""
        if len(self.memory) < self.batch_size:
            return 0  # 经验不足，跳过训练
        
        # 随机抽取经验批次
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = []
        target_fs = []
        
        for state, action, reward, next_state, done in minibatch:
            # 计算目标Q值
            target = reward
            if not done:
                # 使用目标网络预测下一状态的最大Q值
                # 不再需要额外添加维度，直接使用原始状态
                next_state_expanded = np.expand_dims(next_state, axis=0)
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state_expanded, verbose=0)[0]
                )
            
            # 获取当前状态的Q值预测
            # 不再需要额外添加维度，直接使用原始状态
            state_expanded = np.expand_dims(state, axis=0)
            target_f = self.model.predict(state_expanded, verbose=0)
            
            # 更新所选动作的Q值
            target_f[0][action] = target
            
            states.append(state)
            target_fs.append(target_f[0])
        
        # 将批次转换为numpy数组
        states = np.array(states)
        target_fs = np.array(target_fs)
        
        # 训练模型
        history = self.model.fit(states, target_fs, epochs=1, verbose=0, batch_size=self.batch_size)
        
        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 计数并在需要时更新目标网络
        self.update_target_counter += 1
        if self.update_target_counter >= self.update_target_frequency:
            self.update_target_model()
            self.update_target_counter = 0
        
        return history.history['loss'][0]
    
    def load(self, name):
        """加载模型权重"""
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name):
        """保存模型权重"""
        self.model.save_weights(name)

def train_dqn(env, agent, episodes, render=False, checkpoint_dir="checkpoints"):
    """
    使用DQN训练魔方环境
    
    Args:
        env: 魔方环境
        agent: DQN代理
        episodes: 训练的episode数量
        render: 是否渲染环境
        checkpoint_dir: 保存模型检查点的目录
    
    Returns:
        训练历史记录
    """
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 记录训练历史
    history = {
        'episode_rewards': [],
        'episode_steps': [],
        'episode_losses': [],
        'solve_rate': [],
        'epsilon': []
    }
    
    # 统计变量
    solved_episodes = 0
    
    # 训练循环
    for e in range(episodes):
        # 重置环境
        state, _ = env.reset()
        
        total_reward = 0
        losses = []
        
        start_time = time.time()
        
        # 单个episode循环
        for time_step in range(env.max_steps):
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 训练模型
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                losses.append(loss)
            
            # 更新状态和奖励
            state = next_state
            total_reward += reward
            
            # 如果需要渲染，并且是指定的episode，则渲染环境
            if render and e % 50 == 0:
                env.render()
            
            # 如果episode结束，则打印信息
            if done:
                episode_time = time.time() - start_time
                
                # 如果魔方已解决，则增加计数
                if info.get('solved', False):
                    solved_episodes += 1
                
                # 打印episode信息
                print(f"Episode: {e+1}/{episodes} | "
                      f"Steps: {time_step+1} | "
                      f"Reward: {total_reward:.4f} | "
                      f"Best Reward: {info.get('best_reward', 0):.4f} | "
                      f"Epsilon: {agent.epsilon:.4f} | "
                      f"Solved: {info.get('solved', False)} | "
                      f"Time: {episode_time:.2f}s")
                
                break
        
        # 更新历史记录
        history['episode_rewards'].append(total_reward)
        history['episode_steps'].append(time_step + 1)
        history['episode_losses'].append(np.mean(losses) if losses else 0)
        history['epsilon'].append(agent.epsilon)
        history['solve_rate'].append(solved_episodes / (e + 1))
        
        # 每100个episode保存模型和训练历史
        if (e + 1) % 100 == 0:
            agent.save(f"{checkpoint_dir}/dqn_model_episode_{e+1}.h5")
            np.save(f"{checkpoint_dir}/history_{e+1}.npy", history)
            
            # 绘制训练历史图表
            plot_training_history(history, e+1)
    
    # 保存最终模型
    agent.save(f"{checkpoint_dir}/dqn_model_final.h5")
    np.save(f"{checkpoint_dir}/history_final.npy", history)
    
    return history

def test_agent(env, agent, episodes=5, render=True):
    """
    测试已训练的代理
    
    Args:
        env: 魔方环境
        agent: DQN代理
        episodes: 测试的episode数量
        render: 是否渲染环境
    """
    success = 0
    
    for e in range(episodes):
        state, info = env.reset()
        total_reward = 0
        
        for t in range(env.max_steps):
            # 使用代理选择动作，不进行探索
            action = agent.act(state, training=False)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 更新状态和奖励
            state = next_state
            total_reward += reward
            
            # 渲染环境
            if render:
                env.render()
                time.sleep(0.1)  # 添加延迟以便观察
            
            # 如果episode结束，打印信息
            if done:
                if info.get('solved', False):
                    success += 1
                    print(f"Episode {e+1}: 成功! 步数: {t+1}, 总奖励: {total_reward:.4f}")
                else:
                    print(f"Episode {e+1}: 失败. 步数: {t+1}, 总奖励: {total_reward:.4f}")
                break
    
    # 打印成功率
    success_rate = success / episodes
    print(f"测试完成. 成功率: {success_rate:.2f}")
    
    return success_rate

def plot_training_history(history, episode):
    """绘制训练历史图表"""
    plt.figure(figsize=(15, 10))
    
    # 绘制奖励图表
    plt.subplot(2, 2, 1)
    plt.plot(history['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # 绘制步数图表
    plt.subplot(2, 2, 2)
    plt.plot(history['episode_steps'])
    plt.title('Episode Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # 绘制损失图表
    plt.subplot(2, 2, 3)
    plt.plot(history['episode_losses'])
    plt.title('Average Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # 绘制解决率图表
    plt.subplot(2, 2, 4)
    plt.plot(history['solve_rate'])
    plt.title('Solve Rate')
    plt.xlabel('Episode')
    plt.ylabel('Solve Rate')
    
    plt.tight_layout()
    plt.savefig(f'training_history_episode_{episode}.png')
    plt.close()

def main():
    # 创建魔方环境
    # 设置scramble_moves=0将使用特定打乱模式（只打乱顶层）
    # 设置scramble_moves>0将随机打乱指定步数
    use_specific_scramble = True  # 设置为True使用特定打乱模式，False使用随机打乱
    use_oll_pll = True  # 使用OLL/PLL算法进行打乱（只在use_specific_scramble=True时有效）

    env = RubiksCubeEnv(
        cube_size=3,
        render_mode="human",
        max_steps=50,
        scramble_moves=0 if use_specific_scramble else 3,  # 0表示使用特定打乱模式
        use_oll_pll=use_oll_pll  # 是否使用OLL/PLL算法打乱
    )
    
    # 获取状态和动作空间
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    
    print(f"状态空间: {state_shape}")
    print(f"动作空间: {action_size}")
    if use_specific_scramble:
        if use_oll_pll:
            print("打乱模式: 使用OLL/PLL算法打乱（只打乱顶层）")
        else:
            print("打乱模式: 预设动作打乱（只打乱顶层）")
    else:
        print(f"打乱模式: 随机打乱 {env.scramble_moves} 步")
    
    # 创建DQN代理
    agent = DQNAgent(state_shape, action_size)
    
    # 训练模式
    train_mode = True
    
    if train_mode:
        # 训练代理
        history = train_dqn(
            env, 
            agent, 
            episodes=1000,  # 训练的总episode数
            render=True,  # 是否渲染
            checkpoint_dir="rubiks_cube_checkpoints"  # 检查点保存目录
        )
        
        # 绘制训练历史
        plot_training_history(history, "final")
    else:
        # 加载预训练模型
        agent.load("rubiks_cube_checkpoints/dqn_model_final.h5")
        
        # 测试代理
        success_rate = test_agent(env, agent, episodes=10, render=True)

if __name__ == "__main__":
    main() 