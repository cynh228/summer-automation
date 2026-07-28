# MNIST 手写数字识别 (PyTorch)

本项目使用 PyTorch 实现了一个全连接神经网络，在 MNIST 数据集上训练并测试，最终测试准确率达到 **96.5%**。

## 文件结构
mnist/
├── main.py # 训练和测试主脚本
└── README.md # 项目说明

text

## 环境要求

- Python 3.14+
- PyTorch 1.13+
- torchvision

安装依赖：
```bash
pip install torch torchvision
如何运行
进入本目录：

bash
cd summer-automation/mnist
运行脚本（会自动下载数据集）：

bash
python main.py
等待训练结束，查看测试准确率。

模型结构
输入层：784 (28×28 展平)

隐藏层1：512 神经元 + ReLU

隐藏层2：512 神经元 + ReLU

输出层：10 (数字 0~9)

优化器：Adam，学习率 0.001
损失函数：交叉熵损失
批次大小：64
训练轮数：10

训练结果
训练 10 个 epoch 后，测试集准确率稳定在 96.5% 左右。