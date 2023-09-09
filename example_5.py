# -*- coding: utf-8 -*-
"""
MODFLOW6示例，和其他版本最大的区别是：它不再是一个模型（model），
而是一个理论上可以整合任意多个渗流模型和溶质运移模型的模拟（simulation）
二维稳定流潜水含水层
来自 https://zhuanlan.zhihu.com/p/469717393
Created on 2023/09/05
@author: 猛
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import flopy

"""***********************************************************************************************
   给参数赋值
*************************************************************************************************"""
# 首先定义模型的名称，时间单位，长度单位
sim_name = "my_mf6"  # 总模拟的名称
length_units = "meters"  # 长度单位
time_units = "seconds"  # 时间单位

# 加载定义底部高程的 bottom.txt 文件，渗透系数的 hydraulic_conductivity.txt
# 和用于指示含水层网格是否参与模拟的 idomain.txt 文件
bottom = np.loadtxt(os.path.join("data", "bottom.txt"))

idomain = np.loadtxt(os.path.join("data", "idomain.txt"), dtype=np.int32)

# 定义模拟的基本参数
nper = 1  # 应力周期数
nlay = 1  # 含水层层数
nrow = 40  # 行数
ncol = 20  # 列数
delr = 250.0  # 列宽 (m)
delc = 250.0  # 行宽 (m)
top = 35.0  # 顶部高程 (m)


recharge = 1.60000000e-09  # 补给率 (m/s)

###################################################################################################
# 定义井的边界条件
wel_spd = {
    0: [
        [0, 8, 15, -0.00820000],
        [0, 10, 12, -0.00410000],
        [0, 19, 13, -0.00390000],
        [0, 25, 9, -8.30000000e-04],
        [0, 28, 5, -7.20000000e-04],
        [0, 33, 11, -0.00430000],
    ]
}

###################################################################################################
# 定水头边界条件
chd_spd = {
    0: [
        [0, 39, 5, 16.90000000],
        [0, 39, 6, 16.40000000],
        [0, 39, 7, 16.10000000],
        [0, 39, 8, 15.60000000],
        [0, 39, 9, 15.10000000],
        [0, 39, 10, 14.00000000],
        [0, 39, 11, 13.00000000],
        [0, 39, 12, 12.50000000],
        [0, 39, 13, 12.00000000],
        [0, 39, 14, 11.40000000],
    ]
}

###################################################################################################
# 河流边界条件
rbot = np.linspace(20.0, 10.25, num=nrow)  # 河底底部高程
rstage = np.linspace(20.1, 11.25, num=nrow)  # 河流水位
riv_spd = []
for idx, (s, b) in enumerate(zip(rstage, rbot)):
    riv_spd.append([0, idx, 14, s, 0.05, b])
    """[层, 行, 列, 水位, 导水性, 底部高程]，这个导水性应该就是河流向含水层（向下补给）补给的一种系数吧"""

riv_spd = {0: riv_spd}

"""***********************************************************************************************
创建总模拟 Simulation，该模拟可以整合任意多个渗流模型和溶质运移模型
*************************************************************************************************"""
sim_ws = "eg5_workspace"
sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name="mf6")

###################################################################################################
# 时间离散化
tdis_ds = [(1.0, 1, 1.0), ]  # 元组中是（应力期长度，时间步数，时间步数的乘数）
flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_ds, time_units=time_units)

###################################################################################################
# 创建迭代模型求解器对象
ims = flopy.mf6.ModflowIms(
    sim,
    linear_acceleration="BICGSTAB",
    outer_maximum=100,
    outer_dvclose=1e-9 * 10.0,
    inner_maximum=25,
    inner_dvclose=1e-9,
    rcloserecord="{} strict".format(1e-3),
)  # 迭代模型求解

"""***********************************************************************************************
        建立子模型1：渗流模型
*************************************************************************************************"""
# 渗流模型是gwf，本例就一个子模型，就把总模拟的名字也当做这个子模型的名字
gwf = flopy.mf6.ModflowGwf(
    sim, modelname=sim_name, newtonoptions="NEWTON UNDER_RELAXATION", save_flows=True
)

# 对模型空间离散化
dis = flopy.mf6.ModflowGwfdis(
    gwf,
    length_units=length_units,
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    delr=delr,
    delc=delc,
    top=top,
    botm=bottom,
    idomain=idomain,
)

"""赋予模型的特性，如渗透系数等。其中，单元转换类型 icelltype 是指如何去处理每个单元的饱和厚度。
0 表示饱和厚度是定值，个人认为这应该算是指代承压含水层。
>0 表示当水头低于单元所在含水层顶部高程时，饱和厚度随着计算水头发生改变，个人认为这应该时指代潜水含水层。
<0 表示只有当 THICKSTRT 参数开启时，饱和厚度才会随着计算水头发生改变。"""
icelltype = 1  # 单元转换类型
k11 = np.loadtxt(os.path.join("data", "hydraulic_conductivity.txt"))
npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=icelltype, k=k11,)

# 设置初始水头
strt = 45.0  # 初始水头 (m)
ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)  # 初始条件 Initial Conditions 包

# 设置河流、井、降雨补给和定水头。这些与 MODFLOW 2005 类似
riv = flopy.mf6.ModflowGwfriv(
    gwf, stress_period_data=riv_spd, pname="RIV-1", print_flows=True, save_flows=True
)
wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd, pname="WEL-1", save_flows=True)
rcha = flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)  # 补给率 Recharge
chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)

# 设置输出控制
head_filerecord = [f"{sim_name}.hds"]  # 列表里面字符串的形式
budget_filerecord = [f"{sim_name}.cbb"]
saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]  # 指定保存什么数据，"ALL"关键字表示保存所有数据
oc = flopy.mf6.ModflowGwfoc(
    gwf,
    head_filerecord=head_filerecord,  # 水头输出文件的名称，列表里面字符串的形式
    saverecord=saverecord,
    budget_filerecord=budget_filerecord,
    printrecord=saverecord,  # 要打印的数据，这里就是保存什么就打印什么
)

"""***********************************************************************************************
       运行的是总模拟Simulation，而不是子模型
*************************************************************************************************"""
sim.write_simulation()
success, buff = sim.run_simulation()
assert success, "MODFLOW 6 did not terminate normally!"

"""***********************************************************************************************
           数据分析与可视化
*************************************************************************************************"""
head = gwf.output.head().get_data(kstpkper=(0, 0))
plt.imshow(head[0], vmax=28.9251027, vmin=11.1838201)
plt.colorbar()
plt.show()
b = gwf.output.budget().get_data(kstpkper=(0, 0))

print('end')
