# 2026-08-27 完成的完整 UAV/UGV 训练版本

对应 `complete_qnq_cleanobs_h160_uimp008_192k_s42`，包含 20 个学习方法结果和 6 个基线结果。训练种子 42，各学习方法 192,000 步；8 个测试场景、每场景 3 回合、每回合 160 步。

固定发射功率 **1 dBm**；带宽比例 **0.2/0.3/0.5/0.6**；总带宽 100 MHz；源参考 16 bit；UAV 30 m、建筑 25 m；能量 9000 J；通信有效带宽使用当时的 **0.8 系数**。

## 恢复方式与可信范围

这是从备份恢复的历史版本，不是凭配置反推的新实现。基于 station 的 129 文件完整备份 `before_bw3_power_review_20260914_230629/source_code.tar.gz`，核验全部文件校验值，再恢复 5 个后续修改前的文件：量化与非量化环境、双路径控制器、MODIFICATIONS.md 和 test_optimization_contract.py。前四个文件的前后 SHA-256 都与备份清单一致；备份中仅这五个文件的修改时间晚于 8 月 27 日。

八月底没有保存完整源码的同期校验清单，因此不能将该恢复版本宣称为经过同期清单逐字节证明的原始快照。后端来自 station 原始 `DU_IIBTD_res_Sr_learn_nu`，不是 9 月累计覆盖掩码修复版本。26 份结果均与原结果清单的 SHA-256 一致。

## 完整训练入口

依次执行 `du_iibtd_based_fading_delta/run_astar4_cleanobs_h160_uimp008_192k_s42.sh` 和 `du_iibtd_based_fading_delta/run_remaining16_qnq_clean160_192k_s42.sh`：先训练 4 个量化 A* 方法，再训练其余 16 个学习方法并测试 6 个基线。第二个脚本最后调用 `merge_complete_qnq_clean160.py`；迁移机器时也需修改该文件的 `PKG_ROOT`。

## 数据、模型与运行环境

两套源码独立使用，必须从本版本目录运行，避免 Python 导入另一版本的同名包。依赖见 `requirements.txt`；原训练环境是 Linux + CUDA。仓库根目录 `data/` 已有两个数据集，不在版本文件夹重复上传。

在本版本目录建立数据链接：

```bash
ln -s ../data/RadioSeerDPM100PSD RadioSeerDPM100PSD
ln -s ../data/FARMOmniDPM100PSD_251 FARMOmniDPM100PSD_251
```

重建必须另行准备三个预训练 ODU-TD 权重及其 `train_config.json`。默认位置为 `DU_IIBTD_res_Sr_learn_nu/20260623/runs_t3_h0{4,5,6}_res_balance_bw_learnNu_radioseer_farmo/checkpoints/best_nmse.pth`，`train_config.json` 在各 runs 目录下。仓库的 `odu_td_training/` 保存模型训练代码和配置；本文件夹保存本次 UAV/UGV 实验实际使用的重建模型和适配器。不要用别的目录的同名模型覆盖它们。

本次归档包含训练、仿真、评估源码和实际保存的训练配置；不上传原训练权重、完整轨迹、训练日志或重复数据。`results_summary.csv` 仅用于对应实验结果。没有重新执行完整训练。

原始启动脚本与配置中的服务器绝对路径保留。换机器运行时，需要先在工作副本中调整启动脚本的 `CODE_ROOT`、`PY` 以及实际使用的模型/数据路径；归档源码保持原样。旧 `reference_latest_complete_*.json` 是历史参考，正式参数以本版本 `run_configs/` 与下面指定的启动入口为准。

## 归档检查（2026-09-29）

全部 Python 文件与 shell 脚本通过语法检查。初次归档时恢复的旧测试仍包含 4 项过时断言：观测维度和辅助观测仍要求暴露已移除的信息，能量动作掩码仍要求为未来步预留能量。

本次仅将这 4 项测试同步为 2026-09-14 已有的修正版：确认观测维度固定为 14/14/19、辅助观测为空，以及动作能量筛选只检查当前步。训练算法、环境、模型、配置和历史结果均未修改。修正后的完整测试共 70 项，全部通过。原始历史测试保留在前一个 Git 提交中。此次检查没有重新执行完整训练。
