import torch

# 1. 准备数据
x = torch.tensor([1.0])
target = torch.tensor([2.0])

# 2. 初始化权重 w，我们给它一个离谱的初始值 10.0
w = torch.tensor([10.0], requires_grad=True)

# 3. 设置学习率 (步长)
learning_rate = 0.1

print(f"开始训练，初始 w = {w.item():.2f}")
print("-" * 30)

# 4. 开始迭代（下山）
for epoch in range(20):
    # --- 前向传播 ---
    output = x * w
    """
    在这种情况下，loss 虽然在技术上仍然是一个张量，但因为它只有一个值，
    PyTorch 会把它视作标量，允许你直接运行 .backward()。
    
    """
    print("output:", output,output.item())

    loss = (output - target) ** 2  # 平方误差


    # --- 反向传播 ---
    loss.backward()
    print("梯度：", w.grad)

    # --- 梯度下降（更新参数） ---
    # 使用 torch.no_grad() 是因为更新参数本身不需要计算梯度
    with torch.no_grad():
        w -= learning_rate * w.grad

        # ！！！重要：清空梯度 ！！！
        # PyTorch 默认会累加梯度，如果不清零，下次计算就会出错
        w.grad.zero_()

    print(f"第 {epoch + 1} 次尝试: w = {w.item():.4f}, Loss = {loss.item():.4f}")

print("-" * 30)
print(f"训练结束，最终 w = {w.item():.2f}")