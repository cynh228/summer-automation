import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
from collections import deque

# ========== 1. 神经网络定义 ==========
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ========== 2. 经验回放缓冲区 ==========
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ========== 3. 环境与超参数设置 ==========
env = gym.make("CartPole-v1", render_mode="human")  # 如果想看画面，用 "human"
# 如果不想看画面（加速训练），用 "rgb_array" 或 None
# env = gym.make("CartPole-v1", render_mode=None)

n_observations = env.observation_space.shape[0]  # 4
n_actions = env.action_space.n                   # 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
BATCH_SIZE = 64              # 每次训练抽取的样本数
GAMMA = 0.99                 # 折扣因子（未来奖励的折扣率）
EPS_START = 0.9              # 初始探索率（epsilon）
EPS_END = 0.05               # 最终探索率
EPS_DECAY = 1000             # 探索率衰减步数（越大衰减越慢）
TARGET_UPDATE = 10           # 目标网络更新频率（每 10 个 episode 更新一次）
MEMORY_CAPACITY = 10000      # 记忆库容量

# 创建两个网络（当前网络 + 目标网络）
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())  # 初始时让目标网络等于当前网络
target_net.eval()  # 目标网络只用于计算，不训练

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)  # 优化器
memory = ReplayBuffer(MEMORY_CAPACITY)

steps_done = 0  # 全局步数计数器（用于 epsilon 衰减）

# ========== 4. 动作选择函数（epsilon-greedy） ==========
def select_action(state):
    global steps_done
    # 计算当前 epsilon（探索率）
    epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)
    steps_done += 1
    # 以 epsilon 概率随机探索
    if random.random() < epsilon:
        return env.action_space.sample()  # 随机动作
    else:
        # 否则用当前网络选择最优动作
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = policy_net(state_tensor)
            return q_values.argmax().item()

# ========== 5. 训练函数（从经验中学习） ==========
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return  # 记忆不够，不训练
    
    # 从记忆库采样一批经验
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
    
    # 转换为 PyTorch 张量
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    
    # 1. 计算当前 Q 值（从当前网络）
    current_q_values = policy_net(states).gather(1, actions)
    
    # 2. 计算目标 Q 值（从目标网络）
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]  # 取下一个状态的最大 Q 值
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)  # 贝尔曼方程
    
    # 3. 计算损失（Huber Loss，比 MSE 更鲁棒）
    loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
    
    # 4. 反向传播
    optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪（防止梯度爆炸）
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

# ========== 6. 训练主循环 ==========
num_episodes = 200  # 训练 200 个回合
episode_rewards = []  # 记录每个回合的总奖励

for episode in range(num_episodes):
    state, info = env.reset()
    total_reward = 0
    
    for t in range(500):  # 每个回合最多 500 步
        # 1. 选择动作
        action = select_action(state)
        
        # 2. 执行动作
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 3. 存储经验
        memory.push(state, action, reward, next_state, done)
        
        # 4. 更新状态
        state = next_state
        total_reward += reward
        
        # 5. 训练模型（从经验中学习）
        optimize_model()
        
        if done:
            break
    
    episode_rewards.append(total_reward)
    
    # 每 10 个 episode 更新目标网络
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    # 打印进度
    if episode % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode:3d} | 最近10轮平均得分: {avg_reward:.2f} | 本轮得分: {total_reward}")

print("训练完成！")
env.close()