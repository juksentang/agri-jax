# Agri-JAX

**基于 JAX 的可微、批量并行田块尺度作物—土壤模型**——依据已发表方程独立实现，并以 DSSAT-CSM（BSD-3）和 RZWQM2 的输出验证。

[English](README.md) · [成果展示](https://juksentang.github.io/agri-jax/)

每个过程都是纯函数，因此一次 `vmap` 即可模拟 10⁵ 组参数的完整生长季，并用 `jax.grad` 求导。模型可直接读取农学研究者已有的 DSSAT 和 RZWQM 参数文件。

> **当前状态。** DSSAT-CSM v4.8.6 日步长模型（多层水桶模型土壤水分、SPAM 蒸散、根系吸水、CERES-Maize；关闭氮模块）已可自主运行完整生长季，并在 65 个生长季上与 `dscsm048` 对比。Richards 土壤水分求解器作为另一模块并行开发；它与 RZWQM2 风格过程的耦合组装仍在进行。`agrijax.calib.calibrate` 可一次调用拟合 CERES-Maize 品种系数，并将结果写回 DSSAT 的 `.CUL` 文件行。PyPI 包（`agrijax`）目前仅用于预留包名。

## 快速开始

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juksentang/agri-jax/blob/main/tutorial/agrijax_tutorial_zh.ipynb)

```bash
pip install "agrijax[plot] @ git+https://github.com/juksentang/agri-jax"        # Python >= 3.11；无需安装 DSSAT 程序
```

```python
import agrijax as aj

exp = aj.dssat.experiment("UFGA8201")                                           # 从公开文件读取 DSSAT 示例试验
season = exp.run(treatment=4)                                                   # 用 Agri-JAX 的 DSSAT-CSM v4.8.6 玉米模型模拟一个生长季
print(season.summary)                                                           # 吐丝期、成熟期、籽粒产量、地上部重量
aj.plot.season(season, observed=exp.observed(4))                                # 逐日 LAI、生物量、籽粒和土壤水分，并叠加观测值
season.write_dssat_out("out")                                                   # 按 DSSAT 格式写出 PlantGro.OUT、SoilWat.OUT、Summary.OUT

scen = exp.scenarios(treatment=4, years=range(1978, 1988), sowing_shift=[-14, 0, 14])
batch = scen.run({"G2": [800.0, 900.0, 1000.0]})                                # 一次批量调用，在全部 30 个情景下运行每组品种参数
g = exp.gradient(treatment=4, outputs=["HWAM"], params=["G2", "G3"])            # 各导数均附有可信度标签
salus = exp.run(treatment=4, soil_evaporation="salus")                          # 使用经过验证的替代过程
with aj.options(precision="float32", device="gpu"):                             # 按需启用；退出后恢复原有 JAX 设置
    quick = scen.run({"G2": [800.0, 900.0, 1000.0]})
res = aj.calibrate(exp, treatments=[4], holdout=[6])                            # 品种参数校准（联合 CMA-ES）
res.write_cul("out/MZCER048.CUL")

ref = exp.reference(treatment=4)                                                # 可选：原版 DSSAT 程序，首次使用时获取
season.compare_summary(ref)                                                     # 与 DSSAT 对比物候日期和产量
res.check_dssat()                                                               # 在 DSSAT 中运行校准后的品种参数行
```

Agri-JAX 从 DSSAT 自身的文件（FileX、`.SOL`、`.WTH`、`.CUL` / `.ECO` / `.SPE`）构建全部输入；这些文件按固定版本从 DSSAT 仓库一次性下载。只有对比调用（`exp.reference`、`check_dssat`）需要 DSSAT 程序。如果某个处理需要 Agri-JAX 尚未实现的 DSSAT 过程，程序会拒绝运行并说明原因（见“局限”）。由于模型在 float64 下验证，`aj.options(...)` 之外的调用会将 JAX 切换为 float64，并在当前进程后续运行中保持这一设置。Agri-JAX 写出的 `.OUT` 文件会在文件头中注明来源。[教程](tutorial/agrijax_tutorial_zh.ipynb)逐一介绍上述用法；校准细节见 [docs/calibration.md](docs/calibration.md)。

## 验证

下表各项均关闭氮模块、使用 float64，与 DSSAT-CSM v4.8.6.0（`dscsm048`）或 RZWQM2 4.6 输出对比；测量日期为 2026-09-23 至 2026-09-30。

| 检查项 | 结果 |
|---|---|
| 65 个生长季自主运行（模拟每日全部模块；58 个 DSSAT 玉米示例处理及 AmeriFlux CA-TPA 站点的 7 个生长季；共 11 138 天） | 59 个有籽粒生长季的产量：最大相对误差 1.5 %（BRPI0202 t04），中位相对误差 4.0e-5；6 个无籽粒的生长室生长季与 DSSAT 一样给出 0；65/65 个生长季的出苗、吐丝和成熟日期一致；生育阶段和叶片数逐日一致 |
| 同一批运行的逐日序列与 DSSAT 输出对比 | LAI RMSE ≤ 0.0033（示例处理）和 ≤ 0.0061（CA-TPA）m² m⁻²；生物量（CWAD）RMSE 分别 ≤ 生长季最大值的 0.34 % 和 0.075 %；剖面储水量 RMSE ≤ 0.33 mm；各层日末土壤含水量（水桶模型）与 DSSAT 值之差分别 ≤ 9.2e-5 和 ≤ 9.4e-4 cm³ cm⁻³。示例处理的差异在 DSSAT 输出精度以内；CA-TPA 生长季的逐日蒸散和土壤水分差异为该精度的 5–18 倍 |
| 同一批运行的逐日水量收支 | 每次运行的每一天，残差均 ≤ 6.6e-14 mm |
| XTRACT 根系吸水（58 次运行，10 050 天）、PETPT 潜在蒸散及土壤反照率（65 次运行），逐日使用参考模型自身的输入 | 误差分别在 REAL\*4 舍入单位的 1.03、4.7 和 4.5 倍以内；无一天超出测试限值 |
| 单独运行 CERES-Maize，以参考模型的土壤水分和蒸腾驱动（58 个处理） | 逐日 LAI 误差在 0.50 % 以内，生物量 0.63 %，产量 0.030 %；生育阶段逐日一致 |
| H100 float64（展开土层循环，GPU 默认设置；整卡及 1g.10gb 分区）与 CPU float64 对比，65 个生长季 | 产量相对差异 ≤ 1.4e-15，65/65 个生长季的阶段日期一致 |
| Richards 求解器，96×8 网格，CA-TPA 2015–2023，与 RZWQM2 对比；由 RZWQM2 的逐日入渗、蒸发和根系吸水驱动，每年从 RZWQM2 的剖面状态开始 | 各年储水量 RMSE 为 0.0152–0.0480 cm；24×3 网格在九年中的五年达到 0.05 cm 以内；逐日质量平衡误差 ≤ 6.3e-6 cm |
| Shuttleworth–Wallace 和 ASCE 潜在蒸散，CA-TPA 2015（一个站点年） | 与 RZWQM2 对比，潜在蒸发 RMSE 为 4.0e-4 mm/d，潜在蒸腾 RMSE 为 6.6e-5 mm/d；ASCE 与 `pyet` 的差异 < 1e-3 mm/d |

固定程序形状时，结果可逐位复现。在 CPU 上，相同形状和批次组成可逐字节复现验收报告；改变批次大小或组成可能改变最低有效位。在整张 H100 上，展开与保留循环的土层递推给出相同的阶段日期、土壤水分、径流、排水和产量，但其他输出存在最低有效位差异（65 个生长季中，17 个的全部输出逐位一致）。测试为 `tests/integration/test_day_dssat486_free.py` 和 `test_day_dssat486_free_gpu.py`；它们与插桩版 DSSAT 程序生成的表格对比，这些表格不随项目分发（原生输入测试 `tests/integration/test_dssat_free_inputs.py` 检查从 DSSAT 文件构建的输入是否与这些表格一致）。

## 速度

任务：将 65 个验证生长季复制至 10⁵ 个生长季（平均 171 天），使用 float64，并将十条逐日序列返回主机。Agri-JAX 对每个样本的 7 个参数施加 U(0.9, 1.1) 扰动；DSSAT 副本不作扰动。硬件：rorqual（Calcul Québec），2 × AMD EPYC 9654（192 核），一整张 NVIDIA H100 80GB HBM3（主机为 Xeon Gold 6448Y），JAX 0.10.2。“1 核”指此类节点上绑定的一个核心；“32 核”指此类节点分配的 32 个核心，并非 32 核工作站。每个单元格测量一次；同一节点不同进程的重复调用耗时最多相差 20 %。CPU 行测于 2026-09-28，H100 行测于 2026-09-29。

- **首次运行**（冷启动）：在编译缓存为空的新进程中，从 Python 启动计时至序列返回主机。**热启动**：第二个新进程读取 JAX 的持久编译缓存（设置 `jax_persistent_cache_min_compile_time_secs = 0`，否则小程序不会缓存）；仍需支付追踪和编译器中间表示生成的开销。**重复调用**：同一进程、输入已在设备上，取 3 次调用的中位数，将六个已编译程序（土层数和蒸发方法的每种组合一个）的耗时相加，再对两个进程取平均。
- **DSSAT-CSM**：从准备运行目录（节点本地内存盘；若用磁盘文件系统写逐日文件，耗时会更长）计时至最后一次 `dscsm048` 调用写完输出，按核心数分块并行。重复调用仅计运行部分，目录已准备完毕。

| 硬件 | Agri-JAX 首次运行 / 热启动 / 重复调用 | DSSAT-CSM 完整运行含逐日文件（仅汇总）/ 重复调用 | DSSAT ÷ Agri-JAX 首次运行 / 热启动 / 重复调用 |
|---|---|---|---|
| 1 核 | 99.1 s / 56.0 s / 48.9 s | 1426 s (745 s) / 1426 s | 14 / 25 / 29 |
| 32 核 | 20.8 s / 11.3 s / 2.1 s | 46.8 s (26.9 s) / 46.4 s | 2.3 / 4.1 / 22 |
| 192 核（整节点） | 20.5 s / 10.9 s / 0.66 s | 17.4 s (17.2 s) / 16.5 s | 0.85 / 1.6 / 25 |
| 整张 H100，展开土层循环（GPU 默认设置） | 77.5 s / 14.4 s / 0.424 s | 未运行 | |
| 整张 H100，`AGRI_JAX_DEPTH_UNROLL=0` | 28.7 s / 11.2 s / 1.31 s | 未运行 | |

两者执行的工作不同。DSSAT 写出完整逐日文件（每个生长季 303 kB 文本，10⁵ 个生长季共 30.3 GB），并运行整个 CSM，包括土壤温度和有机质。Agri-JAX 仅运行已验证日模型中的模块，返回十条逐日序列（LAI、CWAD、GWAD、stage、profile water、ES、EP、EOP、runoff、drainage；10⁵ 个生长季共 1.5 GB）。若与仅输出汇总结果的 DSSAT 对比，单核耗时比约减半（重复调用从 29 降至 15）；整节点耗时比基本不变。这些比值并非某个单独数值内核的加速比。当前代码的首次运行和热启动耗时增加了几秒，主要来自追踪和中间表示生成的开销（192 核节点：26.1 s / 13.6 s；H100 展开循环：80.5 s / 16.8 s；重复调用不变；测于 2026-09-30）。

| 任务 | 建议选择 | 测量依据 |
|---|---|---|
| ≤ 10³ 个生长季，仅运行一次，冷启动 | DSSAT-CSM | 单核运行 10³ 个生长季：DSSAT 14.3 s，Agri-JAX 首次运行 55.6 s（热启动 8.5 s）。一次性运行的耗时交叉点：单核首次运行约 4·10³ 个生长季，热启动约 6·10²；32 核分别约 4·10⁴ 和 2·10⁴；192 核热启动约 6·10⁴（首次运行的交叉点超出测量范围） |
| 10⁵ 个生长季，仅运行一次，少量核心 | Agri-JAX | 单核首次运行至热启动为 14–25 倍，32 核为 2.3–4.1 倍 |
| 10⁵ 个生长季，仅运行一次，整台 192 核节点 | 均可 | 首次运行 0.85 倍，热启动 1.6 倍 |
| 相同数组形状下多次调用（校准、敏感性分析、集合模拟） | Agri-JAX，CPU 节点或 GPU | 在 1–192 核上，重复调用 10⁵ 个生长季比 DSSAT 快 22–29 倍；整节点 0.66 s，H100 0.42 s。将参数作为实参传入，并将样本数填充到少数几种固定大小：新的形状会触发重新编译 |
| 一次 GPU 运行，或少量梯度步骤 | `AGRI_JAX_DEPTH_UNROLL=0` | 展开循环使每次前向调用快 3.1 倍，但首次运行增加约 45 s（10⁵ 个生长季；约 50 次重复调用后抵消开销），反向模式编译增加 251 s（10⁴ 个生长季，315.6 s 对 64.2 s；约 80 次梯度调用后抵消开销） |
| 仅前向运行，大批量 | float32 | H100 为 1.83–2.15 倍，CPU 节点为 1.15–1.23 倍。与 float64 相比，产量相对差异中位数为 3.3e-7，在 65 个验证生长季中的一个达到 1.5 %；10 000 个扰动样本中，9076 个有籽粒生长季有 13 个差异超过 0.1 %，最大 1.15 %。尚未测量梯度或校准 |

尚未测量，因此不提供耗时比：10⁶ 个及更多生长季、T4 或 Colab GPU、笔记本电脑。Agri-JAX 一侧的基准脚本为 `scripts/bench/d4_jax_scaling.py`。

## 校准

`calibrate(...)` 读取 DSSAT 玉米试验的处理和观测，通过批量联合 CMA-ES（默认方法）拟合六个品种系数（P1、P2、P5、G2、G3、PHINT）；按**写入** `.CUL` 行后的数值选择最佳候选（每个数占五个字符；舍入可能使日期移动一天），写出参数行，并可用它运行 `dscsm048`。适用范围、目标和方法见 [docs/calibration.md](docs/calibration.md)。

- **UFGA8201，品种 IB0035**（校准处理 4，留出处理 6；32 个 CPU 核，35.3 s，2026-09-30）：目标函数值 1.015 → 0.0100，留出处理 1.046 → 0.0239。DSSAT-CSM 使用写出的参数行后，吐丝和成熟日期与 Agri-JAX 一致，产量相对差异在 2.9e-4 以内。
- **写出后回验，22 次校准运行**（11 个处理、两种方法，参数行写入 `MZCER048.CUL` 的副本，2026-09-28）：22/22 次运行的日期与 Agri-JAX 一致，产量差异在 0.1 % 以内。
- **为何采用联合 CMA-ES**：从无噪声合成观测恢复已知品种参数（11 个问题 × 8 个起点 × 3 个随机种子），264 次拟合中有 225 次达到 1e-4 的目标函数值；对照的分阶段方案对生长系数使用梯度步骤，达到这一目标的为 212 次（该研究所用方案；`method="staged"` 本身不使用导数）。在 192 核节点上，两者分别耗时 135 s 和 210 s（2026-09-28）。分阶段方案每次拟合所需模型调用较少（中位数为 443 次前向调用和 51 次双方向导数调用，联合方案为 1008 次前向调用）。梯度步骤（`method="adam"`）仅用于梯度通过可信度报告的系数。
- **可识别性，UFGA8201**（2026-09-28）：仅用日期、产量、生物量和 LAI，G2 与 G3 位于损失函数的一条弯曲谷地上，拟合结果分布于 G2 355–982、G3 7.6–16.4。加入籽粒数（H#AM）后，结果更集中（G2 831–984、G3 7.5–9.4，接近公布值 924.3 和 8.17），留出处理的籽粒数误差也减小（处理 6，每个随机种子的最佳拟合，共三个种子）：−51…−49 % → −7…−6 %。代价体现在雨养处理 2：产量误差从 −5 % 扩大至 −41 %，模型低估了水分胁迫下的籽粒数。损失等高线是条件切片，并非后验分布，见成果展示页。

## 从设计上支持过程组合

过程是用 `@process(reads=..., writes=...)` 声明的纯函数 `(state, params, forcing) -> state`。过程作者遵循三条规则，无需编写 `scan`、`vmap` 或 `jit`：读取的内容全部通过参数传入，修改的内容全部返回；使用 `jnp.where` 分支，不对状态使用 Python `if`；不编写遍历土层、日期或样本的循环。AST 静态检查（AJ001–AJ012、AJ020–AJ021，包括对追踪值调用 NumPy 以及原地修改参数的检查）和启用 `AGRI_JAX_CHECK=1` 后对已声明写入的运行时检查共同落实这些规则。每个模型系数仅声明一次，并记录单位、含义和来源（参考模型及版本、文件和行号），默认均可校准。

端口带有单位，每个字段只有一个写入者，DSSAT 日模型由条目表定义，每个条目运行一个已注册过程。任意条目均可替换；替代过程须通过条目声明检查和符合性检验。运行 `python -m agrijax.testing.conformance --key 'soil_water/*'` 可执行整套符合性检查（元数据、静态检查、单位、通过扰动检查读取项、收支平衡、即时执行与 `jit` 和 `vmap` 的对比、float32、梯度与有限差分的对比）。一行即可将日模型的土壤蒸发过程从 Ritchie（`MESEV = R`）换为 SALUS：

```python
procs = day_processes(SLOT, soil_evaporation=SOIL_EVAPORATION_KEYS["S"])        # 两者均来自 agrijax.models.day_dssat486
```

两种方法均能复现 `dscsm048`。替换会改变模型结果：在 59 个有籽粒生长季中（2026-09-30），产量变化中位数为 −0.07 %，范围从 −27.7 %（BRPI0202 t02）至 +16.1 %（SIAZ9501 t03）；每次替换后的运行，水量收支闭合残差均 ≤ 5.9e-14 mm。

- [docs/swapping_a_process.md](docs/swapping_a_process.md)：替换过程及其通过的检查。
- [docs/tutorial_new_process.md](docs/tutorial_new_process.md)：无需数据，用八个步骤将自己的 NumPy 公式变为已注册、经过检查的过程。
- [docs/debugging.md](docs/debugging.md)：直接调用单个过程、`jax.disable_jit`、`jax.debug.print`、NaN 梯度，以及与参考运行逐日对比。

## 相关工具

PCSE（Python 中的 WOFOST）具有运行时变量写入权管理和逐模块强制状态测试，是与我们的写入权和重放检查最接近的先例。diffWOFOST 和 torchcrop（均基于 PyTorch）是可微作物模型，提供教程和梯度校准。AquaCrop-OSPy 是 AquaCrop 的 Python 实现，提供教程，并与原版保存的输出对比。Crop2ML 是组件交换标准，与框架内部契约相互补充；将 Agri-JAX 过程导出到该标准会是自然的后续方向。APSIM Next Generation 支持替换模块，并在持续集成中运行回归测试。DSSAT-CSM 本身也通过 MPI 并行运行网格模拟（Geosci. Model Dev. 14, 6541, 2021）。JAX-CanVeg 是基于 JAX 的可微冠层和陆面模型，并与其 MATLAB 原版比较速度。TrunX 是 3-PG 森林模型的 JAX 实现，与 r3PG 对比。我们尚未发现其他工具同时具备以下特征：可微、批量并行、多层土壤—作物模型及管理事件，以 DSSAT-CSM 输出验证（土壤水分和 PET 组件还以 RZWQM2 输出验证），并提供逐系数来源记录和面向过程作者的检查（调查时间为 2026 年 9 月）；如有遗漏，欢迎告知。

## 局限

- 仅支持玉米（CERES-Maize），关闭氮模块。如果氮使 DSSAT 产量变化超过 5 %，`calibrate` 会拒绝该处理。
- 对 Agri-JAX 已实现选项的处理，从 DSSAT 文件构建输入（65 次验证运行中的 50 次）；自动灌溉、带地表残留物的 CENTURY 有机质过程、耕作、暗管排水和地下水位均不支持，并会拒绝运行。`calibrate` 需要实测的氮效应，因此仅适用于 DSSAT v4.8.6 玉米示例处理。
- 暗管排水目前只有框架。Richards 求解器尚未在 RZWQM2 风格模型中与作物耦合；24×3 网格在 CA-TPA 九年中的四年未达到 0.05 cm 的标准，另八个 RZWQM2 情景中，96×8 网格仅在 82 个站点年中的 18 个达到 0.05 cm 以内；Shuttleworth–Wallace 仅在一个站点年上验证。
- 固定程序形状时可逐位复现（见“验证”）。GPU 上的 float32 与 CPU 并非逐位一致，且 float32 仅测量了前向运行。
- 梯度按适用范围标注：前向模型已经验证；部分参数支持梯度；穿过物候事件的梯度仍属实验性功能（使用 `method="adam"` 时，P1、P2、P5 和 PHINT 仍采用无导数方法）。
- 速度数据仅针对一个集群上的一项任务，每个单元格测量一次。

## 开发

```bash
git clone https://github.com/juksentang/agri-jax && cd agri-jax
uv sync --all-extras
uv run pytest -q tests/unit tests/integration
uv run ruff check . && uv run pyright
uv run python -m agrijax.core.lint src --strict
```

依赖数据的测试从 `--data-dir` 或 `AGRI_JAX_DATA` 读取数据；单元测试层无需数据。`AGRI_JAX_CHECK=1` 启用运行时写入检查。合并规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。

| 目录 | 内容 |
|---|---|
| `src/agrijax/core` | 状态、过程装饰器、运行时、事件、器官队列、单位、静态检查 |
| `src/agrijax/processes` | 土壤水分、PET、作物、冠层、资源仲裁 |
| `src/agrijax/models` | 组装后的模型（DSSAT 日模型、演示） |
| `src/agrijax/calib`、`sites` | 校准（`calibrate`、优化器、梯度可信度）及自主运行日模型的输入 |
| `src/agrijax/io`、`port` | RZWQM2、DSSAT、AmeriFlux 读写器；参考模型运行器和对比工具 |
| `tests`、`examples`、`scripts/bench` | 测试、可运行示例、基准脚本 |
| `docs` | 上述指南及成果展示页源文件 |

## 许可、来源、致谢与引用

Apache-2.0。CERES-Maize 和 DSSAT 土壤水量平衡依据开源 DSSAT-CSM（BSD-3）独立实现，其署名声明保留于 `THIRD_PARTY_NOTICES.md`。土壤水分和 PET 遵循已发表的 RZWQM2 方程（Ahuja et al., 2000; Farahani & Ahuja, 1996; Shuttleworth & Wallace, 1985）。项目不包含或再分发任何 RZWQM2 代码；RZWQM2 仅作为参考模型对比输出，并仅报告所得数值。

Richards 求解器的 Newton 提前停止实验（`scripts/bench/collab/`）由 Jiaqi Zhang 完成（pull request #1）。本研究的部分工作得到 [Calcul Québec](https://www.calculquebec.ca) 和 [Digital Research Alliance of Canada](https://alliancecan.ca) 的支持。全部 GPU 工作均在 Calcul Québec 运营的 rorqual 集群上运行。

引用本软件请参见列有作者信息的 `CITATION.cff`。设计说明和验证报告将以预印本形式发布。
