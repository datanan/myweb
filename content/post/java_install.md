---
author: "xiaoxie"
date: 2019-06-02
title: "Java安装和配置"
tags: [
    "java",
]
categories: [
    "Java",
]
---


# windows安装

## 下载JDK
首先我们需要下载java开发工具包JDK，[下载地址](http://www.oracle.com/technetwork/java/javase/downloads/index.html)

## 配置环境变量
 "系统变量" 中设置 3 项属性，JAVA_HOME、PATH、CLASSPATH(大小写无所谓),若已存在则点击"编辑"，不存在则点击"新建"
变量设置参数如下：
```
变量名：JAVA_HOME
变量值：C:\Program Files (x86)\Java\jdk1.8.0_91        // 要根据自己的实际路径配置

变量名：CLASSPATH
变量值：.;%JAVA_HOME%\lib\dt.jar;%JAVA_HOME%\lib\tools.jar;         //记得前面有个"."

变量名：Path
变量值1：%JAVA_HOME%\bin
变量值2：%JAVA_HOME%\jre\bin     // 注意win10中要分开写
```

## 测试JDK是否安装成功
```
1、"开始"->"运行"，键入"cmd"；

2、键入命令: java -version、java、javac 几个命令，出现以下信息，说明环境变量配置成功；
```

## 解决没有jre目录

 jdk11和jdk12在以前版本基础上，改动有点大，安装后默认是没有jre的。

【解决方法】：
cmd进入java安装主目录，使用
```
bin\jlink.exe --module-path jmods --add-modules java.desktop --output jre
```
命令手动生成jre

# 【Java IDE】 eclipse安装
