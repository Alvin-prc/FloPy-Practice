# -*- coding: utf-8 -*-
"""
二维潜水含水层非稳定流模型，宽750m，高1000m，离散成15*20个网格
来自 https://zhuanlan.zhihu.com/p/427533365
Created on 2023/09/04
@author: 猛
"""
import os
import flopy
import flopy.utils.binaryfile as bf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###################################################################################################
# 1. 创建模型
modelname = "SM_model"
workspace = "./eg2_workspace"
mf = flopy.modflow.Modflow(
    modelname=modelname, exe_name="mf2005", model_ws=workspace
)

###################################################################################################
# 2. 离散 DIS
Lx = 750.0  # x 长度
Ly = 1000.0  # y 长度
ztop = 5.0  # z 顶部高程
zbot = -50.0  # z 底部高程
nlay = 1  # 含水层层数
nrow = 20  # 行数
ncol = 15  # 列数
delr = Lx / ncol  # x 方向步长
delc = Ly / nrow  # y 方向步长
botm = np.linspace(ztop, zbot, nlay + 1)  # 第0个元素是顶层高层，后面是每一层底部的高程
nper = 3
steady = [True, False, False]
perlen = [1, 100, 100]
nstp = [1, 100, 100]
dis = flopy.modflow.ModflowDis(
    mf,
    nlay,
    nrow,
    ncol,
    delr=delr,
    delc=delc,
    top=ztop,
    botm=botm[1:],  # 每一层的底部高程
    nper=nper,
    perlen=perlen,
    nstp=nstp,
    steady=steady,
)

###################################################################################################
# 3. BAS
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
strt = 10.0 * np.ones((nlay, nrow, ncol), dtype=np.float32)  # 初始水头都是10m
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

###################################################################################################
# 4. LPF
laytyp = 1  # 含水层类型是可变的，我理解就潜水
vka = 1.0  # 垂直渗透系数，单位：m/d
hk = np.ones((nlay, nrow, ncol), dtype=np.float32)  # 水平渗透系数，单位：m/d
hk[:, 0:11, 0:8] = 1.0
hk[:, 0:13, 8:16] = 3.0
hk[:, 11:21, 0:8] = 5.0
hk[:, 13:21, 8:16] = 5.0
sy = 0.1  # 给水度，单位：1/m
ss = 1.0e-4  # 单位储水量，单位：1/m
lpf = flopy.modflow.ModflowLpf(
    model=mf, hk=hk, vka=vka, sy=sy, ss=ss, laytyp=laytyp, ipakcb=1
)

###################################################################################################
# 5. 指定水头边界CHD
# 应力周期 0
shead_0 = 3
ehead_0 = 3
bound_sp_0 = []
# 应力周期 1
shead_1 = 3
ehead_1 = 6
bound_sp_1 = []
# 应力周期 2
shead_2 = 6
ehead_2 = 1
bound_sp_2 = []
for lay in range(nlay):
    for col in range(ncol):
        bound_sp_0.append([lay, 0, col, shead_0, shead_0])
        bound_sp_1.append([lay, 0, col, shead_1, ehead_1])
        bound_sp_2.append([lay, 0, col, shead_2, ehead_2])

chd_spd = {0: bound_sp_0, 1: bound_sp_1, 2: bound_sp_2}
chd = flopy.modflow.ModflowChd(model=mf, stress_period_data=chd_spd)

###################################################################################################
# 6. 井 WEL
wel_spd = {
    0: [[0, 3, 3, -500], [0, 5, 11, -100], [0, 18, 5, 600]],
    1: [[0, 3, 3, -200], [0, 5, 11, 200], [0, 18, 5, -350]],
    2: [[0, 3, 3, -100], [0, 5, 11, 500], [0, 18, 5, 100]],
}
wel = flopy.modflow.ModflowWel(mf, stress_period_data=wel_spd)

###################################################################################################
# 7. 河流 RIV
k_rivbott = 1  # 河床底部材料的渗透系数，m/d
thick_rivbott = 1  # 河床沉积物厚度，m
cond = k_rivbott * delr * delc / thick_rivbott  # conductance, m2/d
r_bott = 0  # 河底高程
riv_stage = [1, 5, 2]  # 河流水位
riv_sp_0 = []  # 应力周期 0
riv_sp_1 = []  # 应力周期 1
riv_sp_2 = []  # 应力周期 2
for i in range(ncol):
    riv_sp_0.append([0, 14, i, riv_stage[0], cond, r_bott])
    riv_sp_1.append([0, 14, i, riv_stage[1], cond, r_bott])
    riv_sp_2.append([0, 14, i, riv_stage[2], cond, r_bott])
riv_spd = {0: riv_sp_0, 1: riv_sp_1, 2: riv_sp_2}
riv = flopy.modflow.ModflowRiv(model=mf, stress_period_data=riv_spd)

###################################################################################################
# 8. 输出控制OC
stress_period_data = {}
for kper in range(nper):
    for kstp in range(nstp[kper]):
        stress_period_data[(kper, kstp)] = [
            "save head",
            "save drawdown",
            "save budget",
            "print head",
            "print budget",
        ]
oc = flopy.modflow.ModflowOc(
    mf, stress_period_data=stress_period_data, compact=True
)

###################################################################################################
# 9. PCG求解器
pcg = flopy.modflow.ModflowPcg(model=mf)

###################################################################################################
# 10. RUN
mf.write_input()
success, mfoutput = mf.run_model(pause=False, report=True)
if not success:
    raise Exception("MODFLOW did not terminate normally!")

"""*************************************************************************************************
                   结果分析与可视化
***************************************************************************************************"""
headobj = bf.HeadFile(os.path.join(workspace, modelname+".hds"))
budgobj = bf.CellBudgetFile(os.path.join(workspace, modelname+".cbc"))

times = [perlen[0], perlen[0] + perlen[1], perlen[0] + perlen[1] + perlen[2]]
"""选的时间点是每个应力周期结束"""

head = {}
frf = {}
fff = {}

for stress_per, time in enumerate(times):
    head["sp%s" % stress_per] = headobj.get_data(totim=time)
    frf["sp%s" % stress_per] = budgobj.get_data(text="FLOW RIGHT FACE", totim=time)
    fff["sp%s" % stress_per] = budgobj.get_data(text="FLOW FRONT FACE", totim=time)

###################################################################################################
# 画出每个应力周期结束时的水头分布
# fig = plt.figure(figsize=(30, 10))
# ax0 = fig.add_subplot(1, 3, 1)
# im0 = ax0.imshow(head["sp0"][0])
# plt.colorbar(im0, fraction=0.06, pad=0.05)
# ax1 = fig.add_subplot(1, 3, 2)
# im1 = ax1.imshow(head["sp1"][0])
# plt.colorbar(im1, fraction=0.06, pad=0.05)
# ax2 = fig.add_subplot(1, 3, 3)
# im2 = ax2.imshow(head["sp2"][0])
# plt.colorbar(im2, fraction=0.06, pad=0.05)
# plt.show()

###################################################################################################
# fd
fig = plt.figure(figsize=(30, 10))
mytimes = [perlen[0], perlen[0] + perlen[1], perlen[0] + perlen[1] + perlen[2]]
"""选的时间点是每个应力周期结束"""
for iplot, time in enumerate(mytimes):
    head = headobj.get_data(totim=time)
    frf = budgobj.get_data(text="FLOW RIGHT FACE", totim=time)[0]
    fff = budgobj.get_data(text="FLOW FRONT FACE", totim=time)[0]

    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(
        (frf, fff, None), mf, head)
    """这一行代码使用Flopy的get_specific_discharge函数计算了特定的排水（流出）量。
    frf和fff分别是右侧面和前方面的流量数据，mf是模型对象，head是模型的水头数据。
    这将计算出x、y和z方向的排水速度分量。这里因为是二维的，所以qz是None吧"""

    ax = fig.add_subplot(1, len(mytimes), iplot + 1, aspect="equal")
    ax.set_title("stress period " + str(iplot))

    pmv = flopy.plot.PlotMapView(model=mf, layer=0, ax=ax)
    """创建一个PlotMapView对象，用于在地图上可视化地下水模型。
    model参数是MODFLOW模型对象（mf），layer参数指定了要可视化的模型层次，ax参数是前面创建的子图。"""
    # qm = pmv.plot_ibound()
    lc = pmv.plot_grid()  # 绘制网格线
    qm = pmv.plot_bc("CHD", alpha=0.5)  # 绘制边界条件
    riv = pmv.plot_bc("RIV", alpha=0.5)

    if head.min() != head.max():
        # 检查场地内水头的最小值和最大值是否相等，如果不相等，就绘制水头等值线和流向
        # note 非常重要的一点，流向、等值线、等值线标签的绘制顺序错误会导致结果错误
        # note 我觉得正确的顺序应该是先绘制流向，再绘制等值线，最后给等值线加标签
        quiver = pmv.plot_vector(qx, qy)  # 绘制流向
        cs = pmv.contour_array(head)  # 绘制水头等值线
        plt.clabel(cs, inline=1, fontsize=10, fmt="%1.1f")  # 给水头等值线加标签

# well 井的坐标
    wpt0 = (150.0, 850.0)
    wpt1 = (550.0, 750.0)
    wpt2 = (250.0, 100.0)
    mfc = "None"
    if (iplot + 1) == len(mytimes):
        mfc = "black"
    ax.plot(wpt0[0], wpt0[1], lw=0, marker="o")  # lw 线宽，marker 点的形状
    ax.text(wpt0[0] + 25, wpt0[1] - 25, "well_A", size=12, zorder=12)
    """zorder 指定文本在图形中的绘制顺序的参数。zorder 数值越大，文本在其他图形元素上方显示的几率越大。"""
    ax.plot(wpt1[0], wpt1[1], lw=0, marker="o")
    ax.text(wpt1[0] + 25, wpt1[1] - 25, "well_B", size=12, zorder=12)
    ax.plot(wpt2[0], wpt2[1], lw=0, marker="o")
    ax.text(wpt2[0] + 25, wpt2[1] - 25, "well_C", size=12, zorder=12)
plt.show()
