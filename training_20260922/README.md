# 3% 累计改善切换与全量重建联动版本

本版本从 `dense_uncertainty_len022_20260917` 的冻结源码建立，旧源码、训练结果和阈值对照数据保留。修改前备份及校验清单见 `verification/backup_manifest.json`；本地副本见 `output/coupled_sum03_optimized_20260920/backup_manifest.json`。

## 正式机制

不确定性为完整集成方差张量 `U`。每次有效地图更新计算：

`r[k] = (||U[k-1]||_F - ||U[k]||_F) / (||U[k-1]||_F + 1e-12)`

在 local 中取最近两次更新的 **有符号改善率之和**。当 `r[k-1] + r[k] < 0.03` 时进入 global；等于 3% 不触发，边界只容忍浮点舍入。需要两次有效地图更新，不是两个时间步，也不是两个连续不达标窗口；负值保留。global→local、环境 reset 和完成切换时清空窗口。

切入 global 后，用当时已经完整回传且已提交的所有观测全量重建一次，然后同步重建图、不确定性图及下一次比较基准，再选择全局目标。重建不增加观测、更新轮次或回传记账。global 至少保持 15 个仿真步；保持结束且局部候选达到原条件时返回 local。正常增量重建继续。

独立不确定性恶化刷新和周期全量刷新已从本版本运行逻辑移除。初始化缺少集成成员时的全量拟合保留。触发条件只读取不确定性，不读取真值 NMSE；NMSE 继续用于原有奖励、critic 和评价。

配置明确保存：

- `hybrid_switch_metric = sum_relative_frobenius`
- `hybrid_uncertainty_window_updates = 2`
- `hybrid_uncertainty_improvement_threshold = 0.03`
- `reconstruction_refresh_mode = local_to_global`
- `hybrid_global_hold_intervals = 5`，`ensemble_refresh_interval = 3`

旧刷新/切换配置会明确报错，历史实验请使用冻结源码。验证旧策略时会显式迁移环境配置，记录为固定策略下的反事实测试。

## 训练入口

`du_iibtd_based_fading_delta/run_paper_comparison_energy8500.sh` 是本版本的统一启动脚本，沿用之前 12 组学习方法和 4 组基线的顺序，输出到新的 `paper_coupled_sum03_maskfix_grid2m_greedyLocal15_len022_uncapped_...uav50_h47to53int_...` 目录。旧的 8 个启动脚本已从本版本移除，以避免旧阈值及旧目录被误用；它们仍在旧版本和备份里。

量化和非量化共用 `shared/reconstruction.py` 的重建生命周期及 `uncertainty_refresh.py` 的联动决策；各方法原有策略与 UGV 控制方式保持一致。

2026-09-20 进一步清理了 11 个历史 Python 入口/分析脚本及 2 个旧配置快照。当前 12 组学习方法、4 组基线的入口和共享控制器保留；已移除文件仍完整保存在冻结旧版本及修改前备份中。清理范围、逐文件校验和验证结果见 `verification/cleanup_record.json`，本地对应记录位于 `output/coupled_sum03_optimized_20260920`。

## 性能优化

在固定高度、固定建筑地图上，预计算各格点执行一个 UAV 宏动作后的落点。落点仍逐格检查穿越路径，并在墙、地图边界处停下。有限动作步数内的可达性搜索使用这一张表，避免重复调用浮点动作模拟；目标到达半径用相同圆形结构元扩张。缓存由场景持有，场景重新初始化时清空，并限制缓存数量。

合并原本完全相同的量化/非量化重建类，删除被替代的独立/周期刷新分支；整合成员提交和资源释放，减少重复地图复制。模型迭代次数、集成规模、物理参数及训练样本数没有因提速而降低。

训练日志新增每轮 `rollout_seconds` 与 `optimization_seconds`；评价轨迹新增 `uncertainty_improvement_sum` 和 `ensemble_mode_switch_refresh_triggered`。原结果读取器需要的旧刷新诊断字段保留为不参与判断的兼容输出；独立刷新标记恒为 false。

## 场景与资源

50 MHz、每 band 原始负载 8 Mbit、32 bit 源参考、512 Mbit 队列、8500 J；发射功率 9/12/15 dBm；UAV 50 m，建筑按块取 47–53 m 整数，1200 平方米拆分阈值；LOS/NLOS 基础附加损耗 1.6/23 dB，遮挡长度修正 0.22 dB/m 无上限，阴影标准差 2/6 dB。规划使用标称链路，实际中断按接收 SNR。

验证过程、删除清单、性能实测及其局限见 `verification` 中的记录和本地 `output/coupled_sum03_optimized_20260920` 的中文修改报告。2026-09-20 的旧配置正式训练已完成，其结果保持原样。


## 2026-09-21 正式训练配置更新

直接更新最近一轮正式训练使用的本目录和原启动脚本。当前每格 2 m、UAV 50 m、UGV 天线 0 m，建筑按块均匀随机取 47～53 m 的整数高度（含两端，当前 UAV 高度 ±3 m），几何种子 42。100×100 网格对应 200×200 m；UAV/UGV 每步最多 4/5 格，即 8/10 m。

将此前已验证的累计覆盖 CNN 输入修复、贪心局部不确定性限制接入正式代码。贪心只查询 UAV 周围曼哈顿距离 ≤15 格的局部地图，对应 30 m；共用目标规划器仍保留。局部地图信息仍多于 PPO 的两个不确定性摘要。正式版本的重建后端独立存放，避免修改其他版本共用的后端。

量化、非量化、训练启动参数、评测网格参数和配置检查器同步。新训练输出目录区分 grid2m、uav50、h47to53，避免误用旧检查点。已完成的旧训练结果、旧保存配置和旧检查点不变。尚未启动本次新配置的正式训练，也没有以旧结果冒充新参数性能；独立 NMSE 恶化保护仍未合入本版。

修改前源码及本次验证见 verification/formal_grid2m_uav50_h47to53_20260921；本地验证在 output/formal_grid2m_uav50_h47to53_20260921。修改高度会影响飞行避障及空地链路，既有无线电场 NPZ 未重新生成。
