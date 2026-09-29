# 2026-09-22 完成的正式训练归档

对应 `paper_coupled_sum03_maskfix_grid2m_greedyLocal15_len022_uncapped_bw50_ref32_split1200_heightavoid_power091215_uav50_h47to53int_e8500_h160_192k_s42`，2026-09-22 06:17:51（UTC+8）完成。12 个学习方法各训练 192,000 步，另有 4 个基线；各方法 8 场景 × 3 回合 × 160 步。

**129 个原始源码文件均与本次训练启动时保存的 source_manifest.json 完全一致。** 原 `README.md` 保留为源码快照，其中“尚未启动”是启动前记录；当前归档状态与结果以本页为准。

配置：2 m/格，UAV 50 m，建筑 47–53 m，9/12/15 dBm 发射功率选择，50 MHz 总带宽，32 bit 源参考，8500 J 能量，种子 42。累计覆盖掩码修复已生效；贪心不确定性信息限制为 UAV 周围曼哈顿半径 15 格；local→global 全量重建，global 至少 15 步；未启用独立 NMSE 恶化重建。

| 主方法 | 终点平均 NMSE | 断联率 | 数据完成率 |
|---|---:|---:|---:|
| 量化 | 0.030191 | 4.79% | 99.86% |
| 非量化 | 0.042915 | 7.27% | 95.81% |

测试使用 final_model.pt。数据完成率是各回合完整包比特数/产生比特数的平均值。完整 16 方法对应关系见 results_summary.csv。

## 完整训练入口

`du_iibtd_based_fading_delta/run_paper_comparison_energy8500.sh`。脚本已使用相对位置确定 CODE_ROOT；换机器需调整固定 PY 解释器路径，并准备模型权重与数据。

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

全部 Python 文件通过语法检查，全部 shell 脚本通过 bash 语法检查。原有 131 项单元测试全部通过；129 文件 SHA-256 与训练启动清单一致。此次检查没有重新训练模型。
