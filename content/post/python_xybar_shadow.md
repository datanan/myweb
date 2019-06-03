---
author: "xiaoxie"
date: 2019-05-09
title: "XY Plot with Error Bar"
tags: [
    "matplotlib",
    "xy",
]
categories: [
    "python",
]
---


# 【写在前面】

调用matplotlib绘制模式和观测对比的xy曲线图，其中模式误差范围用阴影表示


## 这是代码

```

import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt

sim = pd.read_csv("sim.csv")
obs = pd.read_csv("obs.csv")
err = pd.read_csv("variance.csv")

x = sim.time
for i in range(120):
    t = time.strptime(str(x[i]), "%Y%m%d")
    y,m,d,H = t[0:4]
    x[i] = datetime.datetime(y,m,d,H)
x = x.astype('O')

var = ['WLG','TAP','UUM','LLN','YON','SDI','NJ']
lwth = 1
iax = 0
for iax in range(7):
    fig = plt.figure( figsize=(8, 4))
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.9])
    g_s_m = sim[var[iax]]
    g_a_d = obs[var[iax]]
    r = round(g_s_m.corr(g_a_d), 2)

    # ax = fig.subplots(111)
    ax.plot(x,sim[var[iax]],"k-",label='SIM',linewidth=lwth)
    ax.plot(x,obs[var[iax]],"r-",label='OBS',linewidth=lwth)
    ax.fill_between(x, sim[var[iax]]-err[var[iax]], 
        sim[var[iax]]+err[var[iax]],color='gray')
    plt.text(0.05, 0.9, var[iax]+'  r='+str(r), transform=ax.transAxes)
    fig.autofmt_xdate()
    # legend = plt.legend(loc=4, shadow=True, fontsize='small')
    # ax.yaxis.tick_right()
    plt.xticks(rotation=15)
    plt.savefig('figure/'+var[iax]+'.png',dpi=300)
```