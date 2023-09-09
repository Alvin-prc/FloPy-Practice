# -*- coding: utf-8 -*-
"""
对MODFLOW-6模型进行参数敏感性分析
来自 https://zhuanlan.zhihu.com/p/477189865
"""
import os
import flopy
import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

# 文件名
sim_name = "mymodel"  # 模型名称
sim_ws = "eg7_workspace"  # 工作目录
# 单位
length_units = "feet"
time_units = "seconds"
# 空间离散和初始条件
nper = 1
nlay = 5
ncol = 15
nrow = 15
delr = 5000.0
delc = 5000.0
top = 200.0
botm = [-150.0, -200.0, -300.0, -350.0, -450.0]
strt = 0.0
icelltype = [1, 0, 0, 0, 0]  # 1是潜水，0是承压
recharge = 3e-8  # 补给率
k33 = [1.0e-3, 1.0e-8, 1.0e-4, 5.0e-7, 2.0e-4]  # 垂直方向上的渗透系数
# 周期设置
perlen = 8.640e04
nstp = 1
tsmult = 1.0
tdis_ds = ((perlen, nstp, tsmult),)  # 一个应力期对应里面的一个元组，元组里面是应力期长、时间步数、乘数
# 边界条件
# CHD
chd_spd = []
for k in (0, 2):
    chd_spd += [[k, i, 0, 0.0] for i in range(nrow)]
chd_spd = {0: chd_spd}
# 井
wel_spd = {
    0: [
        [4, 4, 10, -5.0],
        [2, 3, 5, -5.0],
        [2, 5, 11, -5.0],
        [0, 8, 7, -5.0],
        [0, 8, 9, -5.0],
        [0, 8, 11, -5.0],
        [0, 8, 13, -5.0],
        [0, 10, 7, -5.0],
        [0, 10, 9, -5.0],
        [0, 10, 11, -5.0],
        [0, 10, 13, -5.0],
        [0, 12, 7, -5.0],
        [0, 12, 9, -5.0],
        [0, 12, 11, -5.0],
        [0, 12, 13, -5.0],
    ]
}
# 排水沟
drn_spd = {
    0: [
        [0, 7, 1, 0.0, 1.0e0],
        [0, 7, 2, 0.0, 1.0e0],
        [0, 7, 3, 10.0, 1.0e0],
        [0, 7, 4, 20.0, 1.0e0],
        [0, 7, 5, 30.0, 1.0e0],
        [0, 7, 6, 50.0, 1.0e0],
        [0, 7, 7, 70.0, 1.0e0],
        [0, 7, 8, 90.0, 1.0e0],
        [0, 7, 9, 100.0, 1.0e0],
    ]
}
# 求解器
nouter = 50
ninner = 100
hclose = 1e-9
rclose = 1e-6


def build_model(k11):
    """
    创建模型
    :param k11:
    :return:
    """
    sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name="mf6")

    tdis = flopy.mf6.ModflowTdis(
        sim, nper=nper, perioddata=tdis_ds, time_units=time_units
    )

    ims = flopy.mf6.ModflowIms(
        sim,
        outer_maximum=nouter,
        outer_dvclose=hclose,
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord="{} strict".format(rclose),
    )

    gwf = flopy.mf6.ModflowGwf(sim, modelname=sim_name, save_flows=True)

    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        length_units=length_units,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )

    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        cvoptions="perched",
        perched=True,
        icelltype=icelltype,
        k=k11,
        k33=k33,
        save_specific_discharge=True,
    )

    ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)  # 初始条件
    chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd)  # 指定水头
    drn = flopy.mf6.ModflowGwfdrn(gwf, stress_period_data=drn_spd)  # 排水沟
    wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wel_spd)  # 井
    rcha = flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge)  # 补给

    head_filerecord = "{}.hds".format(sim_name)
    budget_filerecord = "{}.cbc".format(sim_name)
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    return sim, gwf


def run_sim(sim):
    """
    写入和运行模型
    :param sim:
    :return:
    """
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    if not success:
        raise Exception("MODFLOW 6 did not terminate normally.")

    return None


def read_results(gwf):
    """
    读取模型结果
    :param gwf:
    :return:
    """
    head = gwf.oc.output.head().get_data(kstpkper=(0, 0))
    budget_list = gwf.oc.output.budget().get_data(text="FLOW-JA-FACE")[0]
    grb_file = f"{sim_ws}/{sim_name}.dis.grb"
    residual = flopy.mf6.utils.get_residuals(budget_list, grb_file=grb_file)

    return head, residual


def runrunrun(param_values):
    total_heads = np.array([])
    for x in range(param_values.shape[0]):
        sim, gwf = build_model(param_values[x])
        run_sim(sim)
        head, residual = read_results(gwf)
        total_heads = np.append(total_heads, head[4, 5, 5])
    #         if x % 10 == 0:
    #             print(x,'/',param_values.shape[0])
    #         plot_results(sim)
    return total_heads


"""**********************************************************************************************
      敏感性分析
***********************************************************************************************"""
# 1. Define the model inputs，定义模型的输入，参数的个数，名称，范围
problem = {
    "num_vars": 5,
    "names": ['k11_1', 'k11_2', 'k11_3', 'k11_4', 'k11_5'],
    "bounds": [
        [0.99e-3, 1.11e-3],
        [0.9e-8, 1.02e-8],
        [0.85e-4, 1.2e-4],
        [4.8e-7, 5.13e-7],
        [1.9e-4, 2.12e-4],
    ],
}

# 2. Generate samples，生成样本
param_values = saltelli.sample(problem, 2**4)
"""萨尔泰利采样器会生成N*(2D+2)个样本，其中N是我们提供的参数，D是参数个数。
如果输入keyword argument clac_second_order=False，那么就会不会计算二阶灵敏度指标，
会生成N*(D+2)个样本。"""

# 3. Run model (example)，得到模型输出
head = runrunrun(param_values)

# 4. Perform analysis
Si = sobol.analyze(problem, head, print_to_console=True)

# 绘图
Si.plot()
plt.show()
















