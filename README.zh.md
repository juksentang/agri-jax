# Agri-JAX

**用 JAX 写的田块尺度作物–土壤过程模型：可微分，可成批并行。**

[English](README.md) · [展示页](https://juksentang.github.io/agri-jax/zh_cn/)

> **现状（2026 年 9 月）：早期实现。** 核心运行时、文件读写、参考模型运行器、潜在蒸散、Richards 土壤水分求解器和 CERES-Maize 作物模块都已完成，并各自对照参考模型验证。接下来要把水分和作物耦合成一个整体模型。PyPI 上的包目前只是占位。

## 致谢

本研究部分得到了[魁北克计算局（Calcul Québec）](https://www.calculquebec.ca)和[加拿大数字研究联盟（Digital Research Alliance of Canada）](https://alliancecan.ca)的支持。所有 GPU 计算都在魁北克计算局运营的 rorqual 集群上完成。

## 这是什么

Agri-JAX 把一套田块尺度的作物–土壤模型写成纯 JAX 函数。作物部分依据开源的 [DSSAT-CSM](https://github.com/DSSAT/dssat-csm-os)（BSD-3）独立实现 CERES-Maize；土壤水分和潜在蒸散按 RZWQM2 的公开文献实现，包括 Brooks–Corey 参数化的 Richards 方程、Shuttleworth–Wallace 蒸散和暗管排水（Ahuja 等，2000；Farahani 与 Ahuja，1996；Shuttleworth 与 Wallace，1985）。

每个过程都是纯函数，所以一次多年的田块模拟可以对 10⁵ 组参数用一个 `vmap` 同时算完，可以直接用 `jax.grad` 求导，也可以放进梯度标定、哈密顿蒙特卡洛、集合数据同化或者过程–机器学习混合训练里。结果是否正确，靠与 DSSAT-CSM 和 RZWQM2 参考模型的输出对照来检验。

快本身不是目的。我们想要的是一个有分层土壤物理、有田间管理事件的模型，批量跑起来和求导起来都像那些较简单的可微作物模型一样方便，同时还能直接读农学界手里现成的 DSSAT 和 RZWQM 参数文件。

## 实现进度

| 组件 | 是否实现 | 与参考的对照 |
|---|---|---|
| 核心：状态 pytree、`@process`、scan/vmap 运行时、三条规则的静态检查 | 是 | 运行时与独立的 Python 逐日循环一致（1e-12）；梯度与有限差分一致 |
| RZWQM2 和 DSSAT-CSM 文件读写 | 是 | 15 个场景读写往返逐字节一致；DSSAT 示例输出 |
| 参考模型运行器（RZWQM2、DSSAT-CSM） | 是 | 15 个 RZWQM2 场景和全部 DSSAT 玉米示例都能跑通 |
| Brooks–Corey 水力学 | 是 | 在 100 多个土层上复现参考模型的田间持水量和凋萎点；梯度检查 |
| Shuttleworth–Wallace、ASCE、Priestley–Taylor 潜在蒸散 | 是 | ASCE 误差 5e-6 mm/d；Shuttleworth–Wallace 生长季 RMSE 为 0.02 / 0.17 mm/d（对照 RZWQM2，一个站点年） |
| 器官队列、事件表、AmeriFlux 读取 | 是 | 守恒性质；AmeriFlux CA-TPA 数据 |
| Richards 土壤水分求解器 | 是 | CA-TPA 2015 逐日剖面储水 RMSE 0.026 cm（96×8）/ 0.048 cm（24×3），对照 RZWQM2；每步水量平衡误差 < 1e-10 cm；梯度与有限差分一致 |
| CERES-Maize 作物模块（不含氮） | 是 | 58 个玉米处理对照 DSSAT-CSM v4.8.6：逐日 LAI 误差 ≤ 0.50%，生物量 ≤ 0.63%，产量 ≤ 0.030%，生育阶段逐日一致；DSSAT 写死的系数已变为可标定参数 |
| 水分–作物耦合、标定、不确定性量化 | 下一步 | — |

已有模块没有通过与独立参考的自动化、确定性对照之前，不加新模块。

## 为什么要做

| 任务 | 需要的运行次数 | 是否需要梯度 |
|---|---|---|
| 标定与不确定性量化（LHS、MCMC、HMC） | 10⁴–10⁶ | 需要，HMC 和变分推断都要 |
| 区域格点模拟（像元 × 年份 × 情景） | 10⁶–10⁸ | 不需要 |
| 集合数据同化（EnKF、粒子滤波） | 10³–10⁴ 个成员同步推进 | 不需要 |
| 过程–机器学习混合模型 | 每个训练步一个批次 | 需要 |

DSSAT、RZWQM2、APSIM、STICS 和 WOFOST 都是顺序执行的 Fortran 或 C# 程序，一次只算一块田。它们的实现本身不支持数组式的批量执行，也不可微。可微的作物模型已经有了：作物生长方面有 diffWOFOST 和 torchcrop，冠层与陆面方面有 JAX-CanVeg。Agri-JAX 的区别在于它可微地耦合了哪些东西：分层土壤水分动态、作物生长和田间管理事件，而且 DSSAT 和 RZWQM 的参数文件原样可用。比较这些工具时，更有意义的是看土壤分层、流动方程、管理过程、梯度处理和验证范围。

## 设计

一个过程就是一个纯函数 `(state, params, forcing) -> state`，用 `@process(reads=..., writes=...)` 声明。写过程的人只需遵守三条规则，不用写 `scan`、`vmap`、`jit` 或 `lax.cond`：

1. 读到的东西都来自参数，改动的东西都放进返回值。
2. 分支用 `jnp.where` 或 `jnp.select`，不要对状态写 Python 的 `if`。
3. 不要对土层、日期或样本写循环。时间和批量交给运行时。

这三条由静态检查保证；设置 `AGRI_JAX_CHECK=1` 时，运行时还会核对声明的写入。规则约束的是写过程的人，三对角求解器这类数值内核仍直接用 JAX 编写。

一个模型就是一份状态定义加一串有序的过程，每个过程都可以替换。土壤水分会有两种可互换的写法：DSSAT 式的水桶模型和 RZWQM 式的 Richards 求解器，这样同一个作物模块可以在两种方案下比较。作物是状态的一个维度，间作只是数组形状变了，不需要另写代码。叶片等器官放在一个固定长度的队列里，给烟草这类逐叶采收的作物留好了位置。

## 初步吞吐量

下面的数字来自计算骨架，不是完整模型。骨架的逐日结构和计划中的模型相同：37 个节点的隐式 Richards 步，24 个子步、每步 3 次 Newton 迭代，三对角求解，Shuttleworth–Wallace 形式的蒸散，CERES 形式的作物步，共 3287 天。在一张 NVIDIA H100 上：

| 配置 | 10⁵ 次九年模拟 | 吞吐量 |
|---|---|---|
| float64，24 子步 × 3 次 Newton | 243 s | 411 次/s |
| float32，24 × 3 | 101 s | 993 次/s |
| float64，12 × 2 | 82 s | 1220 次/s |

RZWQM2 参考程序单核跑一次要 24 s。同样的 10⁵ 次需要 667 核时，在集群上大约要 10 个小时。这个批量下 GPU 已经跑满，耗时取决于每天要做多少次隐式求解。数值格式能削减到什么程度，同时还和参考模型一致、梯度仍然可信，是第一篇论文要回答的问题。

## 路线图

1. **Richards 求解器和 CERES-Maize（已完成）。** 两者都先与 RZWQM2 和 DSSAT-CSM 的输出逐日对照，再做耦合。
2. **单站点耦合模型。** 以 AmeriFlux CA-TPA 的玉米为对象，测 10⁵ 样本的耗时，并分三个层次检查梯度：输出合理，导数正确，导数可以用于推断。
3. **标定与不确定性。** 梯度标定和 HMC 标定、Sobol 敏感性分析、多站点联合标定和参数可识别性。
4. **扩展。** 间作、碳氮循环、过程–机器学习混合模型，以及面向农业经济学研究者的报告接口，配 Stata 和 R 前端。

## 开发

```bash
git clone https://github.com/juksentang/agri-jax && cd agri-jax
uv sync --all-extras
uv run pytest -q tests/unit tests/integration
uv run ruff check . && uv run pyright
uv run python -m agrijax.core.lint src --strict
```

依赖数据的测试从 `--data-dir` 或环境变量 `AGRI_JAX_DATA` 读取数据，单元测试不需要任何数据。耦合模型跑通之前，PyPI 上的 `agrijax` 一直保持占位版本。

| 目录 | 内容 |
|---|---|
| `src/agrijax/core` | 状态、过程装饰器、运行时、事件、器官队列、单位、静态检查 |
| `src/agrijax/processes` | 土壤水分、蒸散、作物、冠层、资源分配 |
| `src/agrijax/models` | 组装好的模型 |
| `src/agrijax/io` | RZWQM2、DSSAT、AmeriFlux 和 CA-TPA 的读取 |
| `src/agrijax/port` | 参考模型运行器和对照报告 |
| `docs/showcase` | 展示页源码 |

## 许可与来源

Apache-2.0。CERES-Maize 依据开源的 DSSAT-CSM（BSD-3）独立实现，并保留其署名。土壤水分和潜在蒸散按 RZWQM2 的公开文献实现，不包含也不分发任何 RZWQM2 代码。RZWQM2 只用作参考模型：与它的输出对照在内部进行，对外只报告数字。与 DSSAT-CSM 的对照是公开的，可以复现。

## 引用

见 `CITATION.cff`。设计说明和验证报告之后会以预印本形式发布。
