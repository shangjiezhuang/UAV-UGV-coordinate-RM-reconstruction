# UAV/UGV 协同无线电地图重建与 ODU-TD 训练

仓库保存 ODU-TD 训练代码、两套数据集，以及分别对应 2026 年 8 月底和 9 月 22 日结果的两套完整 UAV/UGV 实验代码。

## 目录

- [`training_20260827/`](training_20260827/README.md)：8 月 27 日完成的固定 1 dBm 版本，由完整备份和 5 个修改前文件恢复，20 个学习方法 + 6 个基线。
- [`training_20260922/`](training_20260922/EXPERIMENT.md)：9 月 22 日完成的正式版本，2 m/格、UAV 50 m、建筑 47–53 m、9/12/15 dBm，12 个学习方法 + 4 个基线；129 个原始源码文件均与训练时校验清单一致。

- `odu_td_training/DU_IIBTD_res_Sr_learn_nu/`：lab 原始训练源码。代码内部沿用 DU-IIBTD 命名；包括模型、训练入口和求解器接口，未修改原文件。
- `odu_td_training/configs/h04/`、`h05/`、`h06/`：2026-06-23 训练保存的配置和划分，对应核带宽 0.4、0.5、0.6。
- `data/RadioSeerDPM100PSD/`、`data/FARMOmniDPM100PSD_251/`：从 station 复制的完整数据目录，含 NPZ、清单、元数据和配套图像。

## ODU-TD 代码

本次仅选取 lab 上 `DU_IIBTD_res_Sr_learn_nu` 的模型训练版本。
`h04`、`h05`、`h06` 是同一套源码分别使用核带宽 0.4、0.5、0.6 的配置，不是三个算法源码版本。

## 数据与划分

| 数据集 | 样本数 | 训练 | 验证 | 测试 |
| --- | ---: | ---: | ---: | ---: |
| RadioSeerDPM100PSD | 10024 | 8024 | 1000 | 1000 |
| FARMOmniDPM100PSD_251 | 2008 | 1608 | 200 | 200 |
| 合计 | 12032 | 9632 | 1200 | 1200 |

每个样本空间网格为 100 × 100，频率维度为 30，R=1。按 base crop 划分，seed=42，同一 base crop 的不同 PSD 变体属于同一划分。训练脚本按数据清单重新生成划分；保存的 CSV 用于核对。

## 运行

在仓库根目录执行。依赖包括 Python、NumPy、PyTorch、Matplotlib。lab 核验环境为 PyTorch 2.10.0+cu128、NumPy 2.4.2、Matplotlib 3.10.8；CUDA 训练需要匹配的 GPU 环境。
