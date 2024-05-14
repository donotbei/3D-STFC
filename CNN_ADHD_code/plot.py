import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


data = np.array([
    [[1,1,1,0,0],[1,1,1,0,0],[1,1,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
    [[1,1,1,0,0],[1,1,1,0,0],[1,1,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
    [[1,1,1,0,0],[1,1,1,0,0],[1,1,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
    [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
    [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
    [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
])

# 创建一个新的Figure对象并添加Axes3D子图
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# 将三维数组展开成二维平面，然后绘制每个点的位置
x = np.arange(len(data))
y = np.arange(len(data[0]))
z = np.arange(len(data[0][0]))

for i in range(len(data)):
    for j in range(len(data[0])):
        for s in range(len(data[0][0])):
            ax.scatter(x[i], y[j], z[s], c='w', s=0, alpha=0.4)
            if int(data[i][j][s]) == 1:
                #p = [((i-min(x))/(max(x)-min(x)))*(2-1)+1 for i in x]
                #q = [((i-min(y))/(max(y)-min(y)))*(2-1)+1 for i in y]
                ax.text(x[i], y[j], z[s], int(data[i][j][s]), c='red',
                        fontsize='xx-large', ha='center', va='center')
            elif int(data[i][j][s]) == 0:
                ax.text(x[i], y[j], z[s], int(data[i][j][s]), c='black',
                        fontsize='xx-large', ha='center', va='center')
            # ax.scatter(x[i], y[j], z[s], c=color, s=0, alpha=0.4)
            #ax.text(x[i], y[j], z[s], int(data[i][j][s]), c=color, fontsize='xx-large', ha='center', va='center')

# 设置轴标签、标题等属性
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
#ax.set_title('Three-dimensional Decision Array', fontweight='bold', fontsize='xx-large')
ax.grid(None)
ax.axis('off')

# 显示图形
plt.show()
