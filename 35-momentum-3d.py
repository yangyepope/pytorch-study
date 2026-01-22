import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 定义那个“狭长山谷”函数
def f(x, y):
    return 0.05 * x**2 + y**2

# 2. 生成网格数据
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 3. 绘图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# 设置标签
ax.set_xlabel('X0 (Flat)')
ax.set_ylabel('X1 (Steep)')
ax.set_zlabel('Loss (Height)')
ax.set_title('The "Narrow Valley" Function')

plt.show()