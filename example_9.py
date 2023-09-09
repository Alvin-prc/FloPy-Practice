# -*- coding: utf-8 -*-
"""
MODFLOW-6 quick start
"""
import flopy
from pathlib import Path
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt

temp_dir = TemporaryDirectory()  # 创建临时文件夹
workspace = Path(temp_dir.name)  # 返回临时文件夹的路径
name = 'mymodel'  # 模拟的名称

sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=workspace, exe_name='mf6')
tdis = flopy.mf6.ModflowTdis(sim)
ims = flopy.mf6.ModflowIms(sim)

gwf = flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)
dis = flopy.mf6.ModflowGwfdis(gwf, nrow=10, ncol=10)
ic = flopy.mf6.ModflowGwfic(gwf)
npf = flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True)
chd = flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), 1.],
                                                       [(0, 9, 9), 0.]])
budget_file = name + '.bud'
head_file = name + '.hds'
oc = flopy.mf6.ModflowGwfoc(gwf,
                            budget_filerecord=budget_file,
                            head_filerecord=head_file,
                            saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
sim.write_simulation()
sim.run_simulation()

head = gwf.output.head().get_data()
bud = gwf.output.budget()

spdis = bud.get_data(text='DATA-SPDIS')[0]
qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)
pmv = flopy.plot.PlotMapView(gwf)  # 创建绘图对象
pmv.plot_array(head)  # 绘制热图
pmv.plot_grid(colors='white')  # 绘制网格
pmv.contour_array(head, levels=[.2, .4, .6, .8], linewidths=3.)  # 绘制等值线
pmv.plot_vector(qx, qy, normalize=True, color="white")  # 绘制向量箭头表示水流方向
plt.show()

# 删除临时文件夹
try:
    temp_dir.cleanup()
except:
    # prevent windows permission error
    pass
