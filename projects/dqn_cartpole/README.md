# DQN 解决 CartPole 倒立摆问题

本项目使用深度 Q 网络（Deep Q-Network）训练智能体在 Gymnasium 的 CartPole-v1 环境中保持平衡。

## 环境与依赖
- Python 3.14+
- PyTorch
- Gymnasium
- NumPy, Matplotlib

安装：
```bash
pip install torch gymnasium numpy matplotlib
方法
网络结构：3层全连接 (128, 128) 输出2个动作值

优化器：Adam，学习率 1e-4

经验回放：容量 10000，采样批次 64

目标网络更新频率：每10个 episode

探索策略：ε-greedy，初始 ε=0.9，终值 0.05，衰减步长 1000

调参实验
我们进行了多组对照实验，分别修改学习率、记忆库容量和网络层数，观察对训练性能的影响。

实验	参数	最终平均得分	收敛速度（达200步）
基准	–	205	140 episode
lr=1e-3	学习率	180	150 episode
...	...	...	...


结论
学习率过大导致训练不稳定，过小则收敛缓慢。

记忆库容量适中有助于稳定训练。

加深网络层数对 CartPole 这种简单任务提升有限，甚至可能增加过拟合风险。

运行
bash
python main.py