---
author: "xiaoxie"
date: 2019-05-28
title: "QGIS根据经纬度判断站点在哪个省份"
tags: [
    "qgis",
]
categories: [
    "QGIS",
]
---


# 【写在前面】

有全国的站点经纬度，根据省份的shp文件判断每个站点在哪个省份，并输出到excel中。
思路： 利用qgis中的join attributes by location进行空间连接。详细操作可以参考：[QGIS教程和技巧](http://www.qgistutorials.com/zh_TW/docs/3/performing_spatial_joins.html)

## 空间连接
```
Input layer： 省份shp
Join layer：站点数据
```
![fig](/image/join_by_attribute.png)