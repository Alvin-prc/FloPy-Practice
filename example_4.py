# -*- coding: utf-8 -*-
"""
二维承压含水层溶质运行模拟
来自 https://zhuanlan.zhihu.com/p/429847479
Created on 2023/09/05
@author: 猛
"""
from pathlib import Path
import numpy as np
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
    "mathtext.fontset": "stix",
    "savefig.dpi": 300,
}
plt.rcParams.update(config)

###################################################################################################
# 1. MODFLOW
model_name = "mf"
work_space = Path("eg4_workspace")
mf = flopy.modflow.Modflow(modelname=model_name, exe_name="mf2005.exe", model_ws=work_space)

###################################################################################################
# 2. DIS
Lx = 1600
Ly = 2000
nrow = 40
ncol = 32
nlay = 1
delr = Lx / ncol  # x方向网格间距
delc = Ly / nrow  # y方向网格间距
ztop = 10
zbot = 0
botm = np.linspace(ztop, zbot, nlay + 1)
dis = flopy.modflow.ModflowDis(
    mf,
    nlay,
    nrow,
    ncol,
    delr=delr,
    delc=delc,
    top=ztop,
    botm=botm[1:],
    perlen=365,
    nstp=30,
    itmuni=4,
)

###################################################################################################
# 3. BAS
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
ibound[0, 0, :] = -1
ibound[0, nrow-1, :] = -1
# ibound[0, 1:-1, 0] = 0
# ibound[0, 1:-1, -1] = 0
strt = 145 * np.ones((nlay, nrow, ncol), dtype=np.float32)
strt[0, 0, :] = 250
strt[0, -1, :] = 36.25
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

###################################################################################################
# 4. LPF
laytyp = 0  # 承压
hk = np.ones((nlay, nrow, ncol), dtype=np.float32)
hk[:, :, :] = 12.7  # 1.474 * (10 ** (-4))
hk[:, 10:18, 4:18] = 0.0127  # 1.474 * (10 ** (-7))
lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=hk, laytyp=laytyp, ipakcb=1)

###################################################################################################
# 5. WEL
wel_spd = {0: [[0, 9, 15, 86.4], [0, 23, 15, -1633.0]]}
wel = flopy.modflow.ModflowWel(mf, stress_period_data=wel_spd)

###################################################################################################
# 6. OC
oc_spd = {(0, 0): ["print head", "print budget", "save head", "save budget"]}
oc = flopy.modflow.ModflowOc(mf, stress_period_data=oc_spd, compact=True)

###################################################################################################
# 7. PCG求解器
pcg = flopy.modflow.ModflowPcg(mf)

###################################################################################################
# 8. LMT link to MT3DMS
lmt = flopy.modflow.ModflowLmt(mf, output_file_name="mt3d_link.ftl")

###################################################################################################
# 9. write
mf.write_input()

# 10. run
success, mfoutput = mf.run_model(pause=False, report=True)
if not success:
    raise Exception("MODFLOW did not terminate normally.")

"""***********************************************************************************************
*             溶质运移模拟
*************************************************************************************************"""
# 1. MT3DMS
mt = flopy.mt3d.Mt3dms(
    modelname="mt",
    version="mt3dms",
    exe_name="mt3dms5b",
    modflowmodel=mf,
    model_ws=work_space,
)

###################################################################################################
# 2. 基本 BTN
icbund = np.ones((nlay, nrow, ncol))
btn = flopy.mt3d.Mt3dBtn(
    mt,
    sconc=0,  # 场地内初始浓度
    prsity=0.3,  # 孔隙率
    thkmin=0.01,  # 单元格中饱和部分厚度与单元格厚度的比值小于thkmin将该单元格视为不活跃
    tunit="D",  # 时间单位
    munit="mg",  # 质量单位
    nprs=5,  # 保存几次结果
    timprs=[0, 20, 60, 120, 360],  # 保存结果的时间点，长度应该等于nprs
    icbund=icbund,
)

###################################################################################################
# 3. 对流 ADV
adv = flopy.mt3d.Mt3dAdv(mt, mixelm=-1, percel=1)
# 弥散 DSP
dsp = flopy.mt3d.Mt3dDsp(mt, al=20, dmcoef=0, trpt=0.2, trpv=0.01)

###################################################################################################
# 4. 源汇项 SSM
ssm_data = {0: [(0, 9, 15, 57.87, 2)]}
ssm = flopy.mt3d.Mt3dSsm(mt, stress_period_data=ssm_data)

###################################################################################################
# 5. GCG求解器
gcg = flopy.mt3d.Mt3dGcg(mt, mxiter=1, iter1=50, isolve=1, cclose=0.0001)

###################################################################################################
# 6. 写入
mt.write_input()

###################################################################################################
# 7. 运行
mt.run_model()

"""**************************************************************************************************
               数据分析与可视化
*************************************************************************************************"""
headobj = bf.HeadFile(Path(work_space) / f"{model_name}.hds")
budgobj = bf.CellBudgetFile(Path(work_space) / f"{model_name}.cbc")

head = headobj.get_data()[0]
frf = budgobj.get_data(text="FLOW RIGHT FACE")[0]
fff = budgobj.get_data(text="FLOW FRONT FACE")[0]

###################################################################################################
# 1. 绘制流向图
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(1, 2, 1)
head_p = ax1.imshow(head)
plt.title("Head (m)")
plt.colorbar(head_p, fraction=0.05, pad=0.05)

ax2 = fig.add_subplot(1, 2, 2)
modelmap = flopy.plot.PlotMapView(model=mf, layer=0)
grid = modelmap.plot_grid()
qm = modelmap.plot_ibound()
lc = modelmap.plot_grid()  # grid
qm1 = modelmap.plot_bc("WEL", alpha=0.5)

quiver = modelmap.plot_vector(frf, fff)
cs = modelmap.contour_array(head)
plt.clabel(cs, inline=1, fmt="%1.1f")

plt.title("Flow Direction")
# plt.show()
# fig.savefig('flow_direction.svg', dpi=300)
###################################################################################################
conc = flopy.utils.UcnFile(Path(work_space) / "MT3D001.UCN")
times = conc.get_times()
conc = conc.get_alldata()

fig = plt.figure(figsize=(20, 5))
ax = [i for i in range(10)]
for i in range(5):
    ax[i] = fig.add_subplot(1, 5, i + 1)
    image = ax[i].imshow(conc[i, 0], cmap="viridis")
    if i == 4:
        divider = make_axes_locatable(ax[i])
        """make_axes_locatable函数接受一个Axes对象作为参数，并返回一个Axes对象的分离器（divider）。
        分离器可以用于创建新的轴（axes）对象，并将其放置在原始轴的附加位置上。
        通常，make_axes_locatable函数与append_axes方法一起使用，以在绘图中创建附加的轴，例如颜色条（colorbar）轴。
        通过分离器，可以将颜色条轴放置在原始轴的一侧或底部，从而实现更灵活的布局和可视化效果。"""
        cax = divider.append_axes("right", size="5%", pad=0.20)  # pad是颜色条与图像的距离，取值范围是0-1
        cbar = plt.colorbar(image, cax=cax)  # 设置颜色条
        # 将颜色条的标题放在颜色条的顶部
        cbar.ax.set_title("Conc", pad=2, fontsize=15)
plt.show()
