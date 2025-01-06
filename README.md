# Codes for Programming Massively Parallel Processors (PMPP)

## Table of Contents
- [English](#english)
- [中文](#中文)

## English

This repository collects the main algorithm implementations from the fourth edition of the PPMP book. Most of the code is referenced from the original book and has been modified in the following ways:

1. A UnifiedPtr template class with reference counting is used to unify the management of memory allocation and deallocation for both CPU and GPU;
2. Errors in the code from the online PDF version of the fourth edition have been corrected;
3. Some of the code that was not provided in the book (e.g., the GPU merge sort algorithm) has been implemented.

## 中文

这个仓库收集了PMPP第四版书中的主要算法实现，大部分代码都从原书中做为参考，并进行了一些修改，主要修改如下。

1. 使用一个带有引用计数的UnifiedPtr模板类，来统一管理CPU和GPU的内存申请与释放；
2. 对于第四版网上PDF版本展示的代码的一些错误进行修改；
3. 对书中部分未给出的代码（比如GPU归并排序算法）进行实现。
