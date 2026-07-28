import gymnasium as gym
import time

# 1. 创建 CartPole 环境（控制杆直立平衡）
env = gym.make("CartPole-v1", render_mode="human")  # 开启可视化窗口

# 2. 初始化环境，获得初始观测
observation, info = env.reset()
print(f"初始观测: {observation}")  # 四维向量：[小车位置, 速度, 杆角度, 角速度]

# 3. 设置计时器
total_steps = 0
done = False

while not done:
    # 3.1 随机选择一个动作（0: 左推, 1: 右推）
    action = env.action_space.sample()  # 随机采样
    
    # 3.2 执行动作，获得新状态、奖励、终止标志、额外信息
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated  # 终止或截断都算结束
    
    # 3.3 可视化（渲染）
    env.render()
    
    total_steps += 1
    # 每 10 步打印一次状态
    if total_steps % 10 == 0:
        print(f"Step {total_steps}: 角度={observation[2]:.3f}, 位置={observation[0]:.3f}")
    
    # 放慢速度，让你看得清（可选）
    time.sleep(0.02)  # 20ms 延迟

print(f"\n🎮 游戏结束！总共坚持了 {total_steps} 步。")
env.close()