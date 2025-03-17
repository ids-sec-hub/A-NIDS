import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font = fm.FontProperties(fname=font_path)


# 定义参数 A 和 B
A = 1
B = 0.1
C = 1000
D = 50000

def plot_dlg(x, y1, y2, xlabel, y1label, y2label, img_path, anchor=(1.0, 1.0)):
    fig, ax1 = plt.subplots()

    ax1.plot(x, y1, linewidth=1, label=y1label, color='#104680')
    ax1.set_xlabel(xlabel, fontproperties=font)
    ax1.set_ylabel(y1label, fontproperties=font)

    # 显示网格
    ax1.set_yticks([])

    ax2 = ax1.twinx()  # 共存的右侧Y轴
    ax2.plot(x, y2, linewidth=1, label=y2label, color='#6D011F')
    ax2.set_ylabel(y2label, fontproperties=font)
    ax2.set_yticks([])

    # 获取两个图的图例项
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # 合并图例项
    lines = lines1 + lines2
    labels = labels1 + labels2

    # 绘制合并的图例
    ax1.legend(lines, labels, prop=font, bbox_to_anchor=anchor)

    plt.savefig(img_path, dpi=256)

# 定义 alpha 的范围
alpha = np.linspace(0.01, 5, 100)
outliers_alpha = np.exp(-(A * alpha + B)**2) / ((A * alpha + B) * np.sqrt(np.pi))

alpha1 = np.linspace(0.1, 1.0, 20)
or1_alpha = C / (np.exp(-(A * alpha1 + B)**2) / ((A * alpha1 + B) * np.sqrt(np.pi)))

alpha2 = np.linspace(1.0, 5.0, 80)
or2_alpha = D * (np.exp(-(A * alpha2 + B)**2) / ((A * alpha2 + B) * np.sqrt(np.pi)))

or_alpha = np.concatenate((or1_alpha, or2_alpha))
print(alpha.shape, or_alpha.shape)

plot_dlg(alpha, outliers_alpha, or_alpha, r'$\alpha$', r'$Outliers(\alpha)$', r'$OR(\alpha)$', './SimulateOutliersAlpha.jpg')
