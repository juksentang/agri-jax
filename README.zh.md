# Agri-JAX

**用 JAX 写的田块尺度作物–土壤过程模型：可微分，可成批并行。**

[English](README.md) · [展示页](https://juksentang.github.io/agri-jax/zh_cn/)

> **项目状态（2026年9月）：早期实现阶段。** 核心运行时、文件读写器、基准模型运行器、潜在蒸散发模块、Richards 土壤水分求解器以及 CERES-Maize 作物模块均已就绪，且各自均通过了与基准参考模型的对比验证。下一步工作是将这些模块耦合为一个统一的水分-作物协同模型。当前 PyPI 包仅作为占位符保留名称。

## 致谢

本研究部分得益于 [Calcul Québec](https://www.calculquebec.ca) 与 [加拿大数字研究联盟（Digital Research Alliance of Canada）](https://alliancecan.ca) 提供的计算支持。所有 GPU 计算均在 Calcul Québec 运维的 rorqual 超算集群上完成。

## 项目简介

Agri-JAX 采用纯 JAX 函数构建了一套田块尺度的作物-土壤机理模型。其中作物生长模块基于开源项目 [DSSAT-CSM](https://github.com/DSSAT/dssat-csm-os) (BSD-3 协议) 中的 CERES-Maize 进行了独立重写；土壤水分动力学与潜在蒸散发则严格遵循 RZWQM2 已发表文献中的物理方程：结合了 Brooks–Corey 保水特性的 Richards 方程、Shuttleworth–Wallace 蒸散发模型以及暗管排水机制（Ahuja 等，2000；Farahani & Ahuja，1996；Shuttleworth & Wallace，1985）。

得益于所有模拟过程均为纯函数设计，用户仅需一次 `vmap` 调用，即可并发完成 $10^5$ 组参数配置下的多年份田块模拟；同时支持直接使用 `jax.grad` 进行自动微分，能够无缝接入基于梯度的参数校准、哈密顿蒙特卡洛（HMC）、集合数据同化或机理-机器学习（Process-ML）混合训练循环。所有模拟计算的正确性均已通过与 DSSAT-CSM 和 RZWQM2 基准参考模型的输出比对严格验证。

极致的运行速度并非本项目的单一追求。我们的目标是构建一个兼具精细分层土壤物理机制与农事管理事件的模型——它既能像轻量级可微作物模型一样，具备低廉的大规模批量运行开销与便捷的可微特性，又能直接兼容农学界已有的 DSSAT 和 RZWQM 参数文件资产。

## 实现进度

| 模块组件 | 实现状态 | 基准对比验证 |
|---|---|---|
| 核心基础：状态 PyTree、`@process` 装饰器、scan/vmap 运行时、“三守则”静态检查 | 已完成 | 运行时对比独立 Python 日循环（误差 $< 10^{-12}$）；自动微分梯度对比有限差分通过检验 |
| RZWQM2 与 DSSAT-CSM 文件读写器 | 已完成 | 15 个场景下实现字节级一致的往返读写（byte-identical round trips）；通过 DSSAT 样例输出校验 |
| 基准模型运行器（RZWQM2、DSSAT-CSM） | 已完成 | 全部 15 个 RZWQM2 场景及全部 DSSAT 玉米样例均成功运行 |
| Brooks–Corey 水力学特性 | 已完成 | 100+ 个土壤层的田间持水量与凋萎点通过基准验证；通过梯度检验 |
| Shuttleworth–Wallace、ASCE 及 Priestley–Taylor PET | 已完成 | ASCE 误差在 $5 \times 10^{-6}\text{ mm/d}$ 以内；单点位-年份下 Shuttleworth–Wallace 生育期对比 RZWQM2 的 RMSE 分别为 0.02 / 0.17 mm/d |
| 器官队列、事件表、AmeriFlux 数据加载器 | 已完成 | 满足守恒性质；通过 AmeriFlux CA-TPA 观测数据验证 |
| Richards 土壤水分求解器 | 已完成 | CA-TPA 2015 年逐日剖面蓄水量对比 RZWQM2：RMSE 为 0.026 cm（96×8 配置）/ 0.048 cm（24×3 配置）；单步质量守恒误差 $< 10^{-10}\text{ cm}$；梯度对比有限差分通过检验 |
| CERES-Maize 作物模块（暂未开启氮素模拟） | 已完成 | 58 组玉米处理对比 DSSAT-CSM v4.8.6：每日 LAI 相对偏差 $\le 0.50\%$，生物量偏差 $\le 0.63\%$，产量偏差 $\le 0.030\%$，生育期阶段逐日完全一致；原 DSSAT 中硬编码的常数已转化为可校准参数 |
| 水分-作物耦合模型、参数校准、不确定性量化 | 下一步计划 | — |

所有新模块的引入均严格遵循一条准则：必须先通过与独立基准模型的自动化、确定性对比测试。

## 研发背景与动机

| 应用任务 | 所需模拟次数 | 是否需要梯度 |
|---|---|---|
| 参数校准与不确定性量化（LHS、MCMC、HMC） | $10^4 \sim 10^6$ | 是（HMC 与变分推断需要） |
| 区域网格化大规模模拟（像元 $\times$ 年份 $\times$ 情景） | $10^6 \sim 10^8$ | 否 |
| 集合数据同化（EnKF、粒子滤波） | $10^3 \sim 10^4$ 个成员严格同步推进 | 否 |
| 机理-机器学习混合模型（Process–ML） | 每个训练步需要一个批次 | 是 |

传统的作物模型（如 DSSAT、RZWQM2、APSIM、STICS 及 WOFOST）大多采用 Fortran 或 C# 编写，专为单田块串行计算设计。其底层实现既无法原生支持张量化的批量并行计算，亦不具备可微性。业界现存的可微作物模型包括：面向作物生长的 diffWOFOST 和 torchcrop，以及面向冠层与地表过程的 JAX-CanVeg。Agri-JAX 的独特之处在于其微分耦合的广度与深度：它将分层土壤水分动力学、作物生长发育以及农事管理事件整体纳入可微框架，且能原生复用农学专家现有的 DSSAT 和 RZWQM 参数文件。评估这些工具的异同，最切实的维度是考察其在土壤分层机制、水流控制方程、农事管理过程、梯度计算机制及基准验证范围上的差异。

## 架构设计

在系统架构中，一个物理/生理过程（Process）是一个签名形如 `(state, params, forcing) -> state` 的纯函数，通过 `@process(reads=..., writes=...)` 进行声明。模块开发者需严格恪守三项守则，无需且严禁在过程内直接调用 `scan`、`vmap`、`jit` 或 `lax.cond`：

1. **显式数据流**：所有需要读取的变量均作为输入参数传入；所有被修改的状态均必须包含在返回值中。
2. **函数式分支**：涉及状态条件的分支逻辑统一使用 `jnp.where` 或 `jnp.select`，严禁在状态变量上使用 Python 原生 `if`。
3. **禁止显式循环**：严禁针对土层、时间步（日）或样本显式编写循环。时间推进与批处理统一交由底层运行时调度。

这三条守则在编译期通过 AST（抽象语法树）静态检查进行强制约束；在运行时还可通过设置 `AGRI_JAX_CHECK=1` 对声明的写入字段实施动态校验。这些规则主要用于规范过程模块的编写；而底层的核心数值计算核（例如三对角矩阵求解器）则直接采用 JAX 编写以释放最优计算性能。

一个完整的模型由“状态定义”与“过程有序列表”组合而成，其中的任一过程均可无缝插拔替换。例如，土壤水分模块将提供两种可互换的实现：DSSAT 风格的翻斗模型（tipping bucket）与 RZWQM 风格的 Richards 求解器，便于在两种不同水文机理下直接横向评测同一作物模块。此外，作物被定义为状态张量的一个独立维度，因此模拟间作仅是改变数组维度形状，无需分化出新的业务逻辑分支。叶片等器官状态由固定长度的队列维护，这为烟草等需要逐叶采收的作物预留了优雅的扩展空间。

## 初步性能表现（吞吐量）

以下性能数据测算自计算骨架原型（Computational Skeleton），而非完整模型。该骨架完整模拟了规划中单日计算的计算图拓扑：包含一个 37 节点隐式 Richards 求解步进（24 个子步进 $\times$ 3 次牛顿迭代）、三对角矩阵求解、Shuttleworth–Wallace 架构的 PET 计算以及 CERES 架构的作物计算步进，连续模拟 3287 天（约 9 年）。在单张 NVIDIA H100 上的测试表现如下：

| 计算配置 | $10^5$ 次 9 年期模拟耗时 | 吞吐量 |
|---|---|---|
| float64，24 子步进 $\times$ 3 次牛顿迭代 | 243 秒 | 411 次模拟/秒 |
| float32，24 $\times$ 3 | 101 秒 | 993 次模拟/秒 |
| float64，12 $\times$ 2 | 82 秒 | 1220 次模拟/秒 |

相比之下，RZWQM2 原生可执行程序在单核 CPU 上运行单次模拟需耗时 24 秒；若完成同样的 $10^5$ 次模拟需消耗 667 个核时（Core-hours），在超算集群上大约需要 10 小时挂钟时间（Wall-clock hours）。在此批处理规模下，GPU 算力已被充分占满，运行时间主要受每日隐式求解次数的支配。在保证与基准参考模型吻合且梯度可靠的前提下，计算步长与迭代次数究竟能精简到何种程度，将是第一篇学术论文重点探讨的核心问题。

## 研发路线图

1. **Richards 求解器与 CERES-Maize（已完成）**：在系统耦合前，各模块均已针对 RZWQM2 与 DSSAT-CSM 的输出完成了逐日的严格比对验证。
2. **单站点耦合模型**：以 AmeriFlux CA-TPA 玉米站点为基准，完成 $10^5$ 级样本性能评测，并在三个层级实施梯度检验：输出合理性、导数正确性、以及导数在统计推断中的适用性。
3. **参数校准与不确定性分析**：基于梯度的优化与 HMC 贝叶斯校准、Sobol 全局灵敏度分析、多站点联合校准及参数可识别性分析。
4. **功能拓展**：间作模拟、碳氮循环系统、机理-机器学习（Process-ML）混合建模，并为农业经济学者提供兼具 Stata 与 R 前端接口的分析报告套件。

## 本地开发

```bash
git clone https://github.com/juksentang/agri-jax && cd agri-jax
uv sync --all-extras
uv run pytest -q tests/unit tests/integration
uv run ruff check . && uv run pyright
uv run python -m agrijax.core.lint src --strict
```

依赖实际数据的测试可通过 `--data-dir` 选项或环境变量 `AGRI_JAX_DATA` 指定数据路径。单元测试层不需要外部数据支持。PyPI 发布包（`pip install agrijax`）在水分-作物耦合模型完全跑通前将持续保持占位状态。

| 目录 | 内容说明 |
|---|---|
| `src/agrijax/core` | 状态定义、过程装饰器、运行时、事件系统、器官队列、单位系统、代码检查器 |
| `src/agrijax/processes` | 土壤水分、潜在蒸散发（PET）、作物、冠层、资源分配与竞争裁决（arbitration） |
| `src/agrijax/models` | 组装完成的集成模型 |
| `src/agrijax/io` | RZWQM2、DSSAT、AmeriFlux 及 CA-TPA 数据读取器 |
| `src/agrijax/port` | 基准参考模型运行器及对比测试报告 |
| `docs/showcase` | Showcase 展示页面源码 |

## 开源协议与代码溯源

本项目采用 Apache-2.0 许可证。CERES-Maize 模块基于开源 DSSAT-CSM（BSD-3 协议）独立重写，并完整保留了原作者署名。DSSAT-CSM 的版权声明与许可证详见 `THIRD_PARTY_NOTICES.md`。土壤水分与潜在蒸散发模块完全基于 RZWQM2 公开发表的物理公式实现，未包含或二次分发任何原版 RZWQM2 代码。RZWQM2 仅用作基准比对参考：其输出数据仅在本地私有环境中运行比对，文档仅披露最终评估指标。与 DSSAT-CSM 的对比测试则是完全公开且可复现的。

## 引用信息

引用格式请参阅 `CITATION.cff`。系统架构设计说明（Design Note）与模型验证报告（Validation Report）将后续以预印本形式发布。
