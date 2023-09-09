# -*- coding: utf-8 -*-
"""
二维承压含水层稳定流，宽200m，高100m，离散成20*10个网格
modflow2005
来自 https://zhuanlan.zhihu.com/p/416569496
Created on 2023/09/04
@author: 猛
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import flopy
import flopy.utils.binaryfile as bf

config = {
    "font.family": "Times New Roman",
    "font.size": 15,
    "font.serif": ["SimSun"],
    "xtick.direction": "in",
    "ytick.direction": "in",
    "mathtext.fontset": "stix",  # 设置数学符号的字体
    "savefig.dpi": 300,
}
plt.rcParams.update(config)
###################################################################################################
# 1. 创建模型
modelname = "SM_model"
work_space = Path("eg1_workspace")
mf = flopy.modflow.Modflow(modelname, model_ws=work_space, exe_name="mf2005")

###################################################################################################
# 2. 定义模型网格
Lx = 200.0
Ly = 100.0
ztop = 0.0
zbot = -10.0
nlay = 1
nrow = 10
ncol = 20
delr = Lx / ncol  # x方向网格间距
delc = Ly / nrow  # y方向网格间距
delv = (ztop - zbot) / nlay  # z方向网格间距
botm = np.linspace(ztop, zbot, nlay + 1)
# Discretization Input File (DIS)
dis = flopy.modflow.ModflowDis(
    mf, nlay, nrow, ncol, delr=delr, delc=delc, top=ztop, botm=botm[1:]
)

###################################################################################################
# 3. Basic Package (BAS)
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
ibound[:, :, 0] = -1
ibound[:, :, -1] = -1
ibound[:, -1, 10:16] = -1
strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
strt[:, :, 0] = 1.0
strt[:, 0:9, -1] = 4.0
strt[:, -1, 10:16] = 3.0
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

###################################################################################################
# 4. Layer-Property Flow Package (LPF)
hk = np.ones((nlay, nrow, ncol), dtype=np.float32)
hk[:, :, 0:5] = 10.0
hk[:, 0:2, 5:10] = 1.0
hk[:, 2:7, 5:10] = 5.0
hk[:, 7:10, 5:10] = 2.0
hk[:, :, 10:20] = 8.0
lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=10.0, ipakcb=1)

###################################################################################################
# 5. Well
spd = {0: [[0, 8, 18, -5000], [0, 4, 4, 4000], [0, 5, 10, -3000], [0, 1, 15, 6000]]}
wel = flopy.modflow.ModflowWel(mf, stress_period_data=spd)

###################################################################################################
# 6. Output Control (OC)
spd = {(0, 0): ["print head", "print budget", "save head", "save budget"]}
oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)
pcg = flopy.modflow.ModflowPcg(mf)
mf.write_input()

###################################################################################################
# 7. run
success, buff = mf.run_model()
if not success:
    raise Exception("MODFLOW did not terminate normally.")

"""************************************************************************************************
               结果分析与可视化
***************************************************************************************************"""
headobj = bf.HeadFile(Path(work_space) / f"{modelname}.hds")

times = headobj.get_times()

head = headobj.get_data(totim=1.0)
df = pd.DataFrame(head[0])

###################################################################################################
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(1, 1, 1)
image = ax.imshow(df, extent=[0, 200, 0, 100])
"""extent参数是用来设置图像显示的区域，extent=[xmin, xmax, ymin, ymax]，
note 一定要记住extent这个参数，很好用"""

# 设置x轴和y轴的刻度所在的位置和刻度标签
ax.set_xticks([0, 100, 200])  # 显示刻度的位置（在这些位置显示刻度）
ax.set_xlabel(r"$\mathit{x}$/m")  # 设置x轴的标签

ax.set_yticks([0, 50, 100])
ax.set_ylabel(r"$\mathit{y}$/m")

divider = make_axes_locatable(ax)
"""make_axes_locatable函数接受一个Axes对象作为参数，并返回一个Axes对象的分离器（divider）。
分离器可以用于创建新的轴（axes）对象，并将其放置在原始轴的附加位置上。
通常，make_axes_locatable函数与append_axes方法一起使用，以在绘图中创建附加的轴，例如颜色条（colorbar）轴。
通过分离器，可以将颜色条轴放置在原始轴的一侧或底部，从而实现更灵活的布局和可视化效果。"""
cax = divider.append_axes("right", size="3%", pad=0.10)  # pad是颜色条与图像的距离，取值范围是0-1
cbar = plt.colorbar(image, cax=cax)  # 设置颜色条
# 将颜色条的标题放在颜色条的顶部
cbar.ax.set_title('Head', pad=2, fontsize=15)
fig.tight_layout()  # 调整图片边缘
plt.show()
