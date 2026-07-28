# 1. 导入 PyTorch 核心库和辅助库
import torch                      # PyTorch 主库
from torch import nn              # nn 包含所有神经网络层（Linear, ReLU 等）
from torch.utils.data import DataLoader  # 数据打包器，负责批量喂数据
from torchvision import datasets  # 内置经典数据集（MNIST, CIFAR 等）
from torchvision.transforms import ToTensor  # 将图片转成张量并归一化

# 2. 准备训练数据（下载 MNIST）
training_data = datasets.MNIST(
    root="data",          # 存到当前目录的 data 文件夹
    train=True,           # 加载训练集（60000 张）
    download=True,        # 如果本地没有，自动下载
    transform=ToTensor()  # 将图片转为 0~1 之间的张量
)

# 3. 准备测试数据（评估模型最终效果）
test_data = datasets.MNIST(
    root="data",
    train=False,          # 加载测试集（10000 张）
    download=True,
    transform=ToTensor()
)

# 4. 配置数据加载器（食堂阿姨）
batch_size = 64                   # 每次给模型喂 64 张图片
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)  # 训练集打乱顺序
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)      # 测试集不打乱

# 5. 检测运行设备（有 GPU 就用，没有就用 CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用: {device}")

# 6. 定义神经网络模型结构（这是核心积木）
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()                # 必须调用父类构造方法
        self.flatten = nn.Flatten()       # 把 28x28 的图片压平成一维向量（784 个像素）
        # Sequential 是一个顺序容器，数据会依次经过里面的每一层
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),        # 全连接层1：输入 784，输出 512
            nn.ReLU(),                    # 激活函数：把负数变成 0，引入非线性
            nn.Linear(512, 512),          # 全连接层2：输入 512，输出 512
            nn.ReLU(),
            nn.Linear(512, 10),           # 输出层：输入 512，输出 10（对应 0~9 十个数字的得分）
        )
    
    # 前向传播（数据怎么走）
    def forward(self, x):
        x = self.flatten(x)               # 先压平
        logits = self.linear_relu_stack(x) # 再经过全连接堆栈
        return logits                      # 返回原始得分（不用 softmax，因为 CrossEntropyLoss 自带）

# 7. 实例化模型并搬到指定设备
model = NeuralNetwork().to(device)
print(model)  # 打印模型结构看一眼

# 8. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()           # 阅卷老师：计算预测得分和真实标签的差距
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 家教：根据损失调整模型参数

# 9. 训练函数（一节课）
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)        # 总样本数（60000）
    model.train()                         # 开启训练模式（对 Dropout 等层有效）
    for batch, (X, y) in enumerate(dataloader):  # 每次循环取 64 张图
        X, y = X.to(device), y.to(device) # 把数据搬到设备（CPU/GPU）
        
        # 前向传播：模型做预测
        pred = model(X)                   # 输入 64 张图，输出 64 个 10 维得分向量
        loss = loss_fn(pred, y)           # 计算这 64 张图的平均损失
        
        # 反向传播：计算梯度并更新参数
        optimizer.zero_grad()             # 清空上一批次的梯度（防累积）
        loss.backward()                   # 反向传播，计算每个参数的梯度
        optimizer.step()                  # 根据梯度更新参数
        
        # 每 100 批打印一次当前损失（6400 张图打印一次）
        if batch % 100 == 0:
            loss_value = loss.item()      # 提取标量损失值
            current = batch * len(X)      # 已处理的图片张数
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

# 10. 测试函数（月考）
def test_loop(dataloader, model, loss_fn):
    model.eval()                          # 开启评估模式（不启用 Dropout）
    size = len(dataloader.dataset)        # 测试集总样本数（10000）
    num_batches = len(dataloader)         # 总批次数
    test_loss, correct = 0, 0             # 累计损失和正确数
    
    with torch.no_grad():                 # 禁用梯度计算（省内存、省时间）
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)               # 预测
            test_loss += loss_fn(pred, y).item()  # 累加损失
            # pred.argmax(1) 取每张图得分最高的类别索引
            # 与真实标签 y 比较，相等则累加正确数
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches              # 平均损失
    correct /= size                       # 准确率（0~1）
    print(f"测试结果: \n 准确率: {(100*correct):>0.1f}%, 平均损失: {test_loss:>8f} \n")

# 11. 主程序入口（按课程表上课）
if __name__ == "__main__":
    epochs = 10                           # 总共上 10 节课
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)  # 上课
        test_loop(test_dataloader, model, loss_fn)               # 月考
    print("完成！")