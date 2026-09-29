# UAV/UGV 协同无线电地图重建与 ODU-TD 训练

仓库保存 ODU-TD 训练代码、两套数据集，以及分别对应 2026 年 8 月底和 9 月 22 日结果的两套完整 UAV/UGV 实验代码。

## 目录

- [`training_20260827/`](training_20260827/README.md)：8 月 27 日完成的固定 1 dBm 版本，由完整备份和 5 个修改前文件恢复，20 个学习方法 + 6 个基线。
- [`training_20260922/`](training_20260922/EXPERIMENT.md)：9 月 22 日完成的正式版本，2 m/格、UAV 50 m、建筑 47–53 m、9/12/15 dBm，12 个学习方法 + 4 个基线；129 个原始源码文件均与训练时校验清单一致。

- `odu_td_training/DU_IIBTD_res_Sr_learn_nu/`：lab 原始训练源码。代码内部沿用 DU-IIBTD 命名；包括模型、训练入口和求解器接口，未修改原文件。
- `odu_td_training/configs/h04/`、`h05/`、`h06/`：2026-06-23 训练保存的配置和划分，对应核带宽 0.4、0.5、0.6。
- `data/RadioSeerDPM100PSD/`、`data/FARMOmniDPM100PSD_251/`：从 station 复制的完整数据目录，含 NPZ、清单、元数据和配套图像。

更早的旧代码保留在 Git 历史中。两个实验版本各自独立存放，附实际训练配置与简要结果对应表；不包含训练模型权重、完整轨迹或训练日志。

## 唯一上传的 ODU-TD 代码版本

本次仅选取 lab 上 `DU_IIBTD_res_Sr_learn_nu` 的模型训练版本。已在 lab 实际检查导入关系：`train.py` 中的 `from DU_IIBTD import ...` 指向同目录的 `DU_IIBTD.py`。训练源码、模型实现和接口文件均保持该版本原样。

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

下面命令使用 h04 保存的训练参数，并将旧服务器绝对路径改为当前仓库路径。共 150 个 mini-epoch，每轮 1000 个训练样本，batch size=2，每 5 轮验证；T=3、hidden=32、可学习 nu、全局 Sr 更新。观测点数 128–256，连续频段宽度 2–6。

```bash
python odu_td_training/DU_IIBTD_res_Sr_learn_nu/train.py \
  --dataset-root "$PWD/data/RadioSeerDPM100PSD" \
  --extra-dataset-root "$PWD/data/FARMOmniDPM100PSD_251" \
  --output-dir "$PWD/outputs/h04" --device cuda:0 \
  --seed 42 \
  --max-samples 0 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --num-workers 2 \
  --train-samples-per-epoch 1000 \
  --val-interval 5 \
  --val-max-samples 0 \
  --spatial-points-min 128 \
  --spatial-points-max 256 \
  --freq-mode sample_width_contiguous \
  --full-observation-ratio 0.5 \
  --freq-band-min 2 \
  --freq-band-max 6 \
  --min-sensors-for-update 6 \
  --epochs 150 \
  --batch-size 2 \
  --lr 0.0001 \
  --weight-decay 0.0001 \
  --grad-clip 10.0 \
  --sr-loss-weight 0.1 \
  --phi-loss-weight 0.1 \
  --obs-loss-weight 0.05 \
  --log-interval 100 \
  --T 3 \
  --hidden 32 \
  --nu 0.1 \
  --min-nu 1e-08 \
  --kernel-bandwidth 0.4 \
  --theta-chunk-size 4096 \
  --theta-ridge 1e-05 \
  --sr-update-mode global \
  --global-sr-update-weight 0.1
```

训练 h05 或 h06 时，将 `--kernel-bandwidth` 改为 `0.5` 或 `0.6`，并设置不同的 `--output-dir`。原配置和划分 CSV 中的 `/home/zsj/...` 路径保留作历史记录；运行时使用上述命令指定的路径。

训练源码目录中的原 `README` 是较早的说明；正式训练参数以 `configs/` 和本页命令为准。已核对全部 33,098 个数据文件的 SHA-256、12,032 个 NPZ 样本、三组训练配置的数据划分，并通过一轮 CPU 小规模训练、验证、测试和权重保存检查。该检查不代表重新完成了 150 轮正式训练。

## 来源

- lab：`/home/zsj/works/work1/code/archive_training_before_20260921/DU_IIBTD_res_Sr_learn_nu/`
- station 数据：`/home/zsj/works/work1/code/RadioSeerDPM100PSD/`、`/home/zsj/works/work1/code/FARMOmniDPM100PSD_251/`
- 替换前 GitHub main：`e215ac75afc5eb74792f02469d0f18d30528aeee`
