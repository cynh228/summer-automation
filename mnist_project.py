import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 创建模型实例，并将其移动到我们选择的设备（CPU 或 GPU）
model = NeuralNetwork().to(device)
print(model)  # 打印模型结构

# --- 3. 定义损失函数和优化器 ---
# 损失函数: 用于分类任务的交叉熵损失
loss_fn = nn.CrossEntropyLoss()
# 优化器: 随机梯度下降 (SGD)，学习率 lr=0.001
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# --- 4. 训练和测试函数 ---
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # 将模型设置为训练模式（对某些层如 Dropout 有影响）
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # 将数据移动到选定的设备 (CPU/GPU)
        X, y = X.to(device), y.to(device)

        # 1. 计算预测和损失
        pred = model(X)# 用当前这64张图做预测
        loss = loss_fn(pred, y) # 计算当前这64张图的损失（自动取平均）

        # 2. 反向传播
        # 清空上一步的梯度
        optimizer.zero_grad()
        # 计算当前梯度的“误差反向传播”
        loss.backward()
        # 3. 更新参数
        optimizer.step()

        # 每 100 个批次打印一次损失
        if batch % 100 == 0:
            loss_value = loss.item()
            current = batch * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
#zero_grad 是“清理旧账”，backward 是“调查原因”，step 是“实际改正”。这三行永远绑在一起，以后你写任何 PyTorch 训练代码，都会像刻在肌肉记忆里一样抄这三行。

def test_loop(dataloader, model, loss_fn):
    # 将模型设置为评估模式
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # 在不计算梯度的上下文下进行评估，可以节省内存和计算
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # 获取预测结果中概率最高的类别
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"测试结果: \n 准确率: {(100*correct):>0.1f}%, 平均损失: {test_loss:>8f} \n")


# --- 5. 开始训练 ---
if __name__ == "__main__":
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("完成！")