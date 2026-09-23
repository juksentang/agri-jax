# Agri-JAX — 计划零号

> 可微分、可批量的作物-土壤过程模型框架，面向参数标定、不确定性量化和机理-机器学习混合建模。
> GPU 是副产品，可微分和 batch 才是目的。

状态：构想阶段（2026-09-23）。本文件是第一版计划，所有内容都可以推翻。

**详细文档（2026-09-23 补充）**：

| 文档 | 内容 |
|---|---|
| [docs/zh_cn/02_architecture.md](docs/zh_cn/02_architecture.md) | 项目架构：目录、State/Params/Forcing、`@process`、Model、runtime、io、calib、report |
| [docs/zh_cn/03_development_plan.md](docs/zh_cn/03_development_plan.md) | 开发规划：第 0 周脚手架、PoC 逐周任务与里程碑、第二/三阶段、风险 |
| [docs/zh_cn/05_maintenance_pipeline.md](docs/zh_cn/05_maintenance_pipeline.md) | 维护管线：仓库/许可、工具链、lint、测试分层、CI、集群分工（rorqual 计算 / narval 旧数据）、发布 |
| [docs/zh_cn/06_open_source_ecosystem.md](docs/zh_cn/06_open_source_ecosystem.md) | 开源生态与选型：可直接用的库、设计参考、与 diffWOFOST 等的尺度区分 |
| [docs/zh_cn/08_throughput_comparison.md](docs/zh_cn/08_throughput_comparison.md) | 大规模批量耗时对比：参考模型（CPU）实测 vs H100 骨架实测 |

**对本文假设的更正**：RZWQM2 4.6 参考模型在本机 16 s 跑完 CA-TPA 九年；dssat-csm-os 4.8.5 源码与静态二进制在本机 `~/AFSoil`；CA-TPA 的 10 万次 LHS 已全部跑完；`agri-jax` 名字可用；GPU 作业去 rorqual H100 而非 narval。

---

## 1. 为什么做

现有作物模型（DSSAT、RZWQM2、APSIM、STICS、WOFOST）都是几十年积累的顺序代码。它们跑一块田很快，但下面四类工作需要成千上万份相同结构的模拟同时推进，而且越来越需要梯度：

| 用途 | 规模 | 需要梯度 |
|---|---|---|
| 参数标定 / 不确定性量化（LHS、MCMC、HMC） | 10⁴–10⁶ 次运行 | 是（HMC、变分推断） |
| 区域网格模拟（像元 × 年 × 情景） | 10⁶–10⁸ 次运行 | 否 |
| 集合数据同化（EnKF、粒子滤波） | 10³–10⁴ 成员同步推进 | 否 |
| 机理-ML 混合模型（模型嵌在训练循环里） | 每步一个 batch | 是 |

JAX 的 `scan`（时间）+ `vmap`（样本）+ `jit` + 自动微分正好对应这个形状。这条路线在水文里已经证明有效（differentiable modeling，Shen 等 2023，Nat. Rev. Earth Environ.），作物模型里还是空白。

**什么不是目的**：让农学家写 JAX；把单块田的模拟跑得更快。

## 2. 直接动机（我们自己的需求）

- RZWQM2 的 LHS 标定：每个站点 10 万次运行，narval 上 24–70 s/次，每站 700 GB 输出，排队和 I/O 是瓶颈，标定本身还是 GLUE 式的筛选。
- OpenET 项目：7 个通量塔站点（4 加拿大 + 3 美国）的 ET 序列和 4–6 个遥感 ET 模型结果，可作为验证集。
- 现成数据资产：15 个 RZWQM scenario、12 个已完成的 10 万样本 LHS 结果（`/project/def-zhiming/jsentang/RZWQM_sw_batch`，narval）。

## 3. 设计原则

### 3.1 农学家写过程，框架管其余

过程函数只有一种形状：

```python
@process
def leaf_growth(state, params, forcing):
    """CERES-Maize leaf expansion, Jones & Kiniry 1986 eq. 3.12"""
    stress = jnp.minimum(state.water_stress, state.n_stress)
    dlai = params.plai_rate * forcing.tt_today * stress
    dlai = jnp.where(state.stage == LEAF_GROWTH, dlai, 0.0)
    return state.replace(lai=state.lai + dlai)
```

给过程作者的规则只有三条：
1. 读的东西全在参数里，改的东西全在返回值里（纯函数）。
2. 分支用 `jnp.where` / `jnp.select`，不用 Python `if` 作用于状态量。
3. 不写循环；时间循环和样本循环由运行时负责。

`scan`、`vmap`、`jit`、`lax.cond` 对过程作者不可见。lint 规则在 CI 里拦截违规写法（traced 值上的 `if`、原地赋值、Python `for` 遍历土层）。

这三条比全局变量和 OOP 框架（APSIM NG、PCSE）的事件回调都简单：每个过程读什么、改什么，签名上一目了然。DSSAT 之所以难维护，根因是无接口、无测试、全局状态，不是语言；这三条规则正是补这三样。

### 3.2 三层接口

| 层 | 用户 | 内容 |
|---|---|---|
| 顶层 | 用模型的人 | `load_dssat_experiment()`、`calibrate(model, obs, method="hmc")`、`sensitivity()`、`ensemble()` |
| 中层 | 建模者 | 模型 = 状态定义 + 有序过程列表；可替换任一过程 |
| 底层 | 我们 | 运行时（scan/vmap/jit）、对照参考模型输出的验证工具 |

预期 90% 用户停在顶层和参数文件。

### 3.3 兼容现有参数文件（最强的采用钩子）

直接读 DSSAT 的 `.CUL / .ECO / .SPE / .SOL / .WTH` 和 RZWQM 的 `rzwqm.dat`，农学家几十年积累的品种参数一行不改就能用。

### 3.4 状态是带名字的 pytree

状态用 `NamedTuple` / `flax.struct` 定义，字段有名字和单位，`state.lai`、`state.sw[layer]`，不是数组下标。带 `n_crop` 维度（见第 5 节），单作只是 `n_crop = 1`。

### 3.5 精度与数值

- 验证阶段 `jax_enable_x64`，对照参考模型的双精度输出。
- 数据相关的迭代（Richards 方程收敛、根系吸水迭代）改为固定迭代数或固定步长隐式格式，接受与参考模型的小差异，差异写进验证报告。

## 4. 基础模型选择：DSSAT-CSM 作物模块 + RZWQM2 土壤/ET（按已发表公式独立实现）

| 候选 | 优点 | 问题 |
|---|---|---|
| 原版 DSSAT-CSM | 源码公开（GitHub `DSSAT/dssat-csm-os`），版本新，社区大，品种文件生态完整 | 土壤水是 tipping bucket，ET 选项简单，没有 Richards、大孔隙、SHAW |
| RZWQM2 | 土壤物理强（Richards + Green-Ampt、大孔隙、SHAW 能量平衡、S-W PET），氮磷碳循环完整，组里的标定流程都建在它上面 | 由 USDA-ARS 分发，只作为参考模型使用；内嵌的 DSSAT 作物模块是旧版 |

**决定**：
- 作物模块基于 **DSSAT-CSM 开源仓库**（BSD-3）独立实现（先 CERES-Maize），版本新且许可干净，保留 DSSAT-CSM 署名。
- 土壤水、PET、能量平衡按 **已发表的 RZWQM2 公式**（Ahuja 等 2000《Root Zone Water Quality Model》；Farahani & Ahuja 1996；Shuttleworth & Wallace 1985）独立实现。公式是公开的，实现是我们的。
- 土壤水模块做成可插拔：`tipping_bucket`（复现 DSSAT）和 `richards`（复现 RZWQM）两套，同一作物模块可以对照两种土壤物理，本身就是一个可发表的比较。
- 验证对象两个参考模型：DSSAT-CSM（作物部分）和 RZWQM2（土壤 + 作物整体，用现有 15 个 scenario），比较两者的输出。

## 5. 间作：DSSAT 不原生支持，这是设计机会而不是障碍

DSSAT-CSM 一块地一种作物，间作只有零散的实验性工作（玉米-豆间作的研究分支），没进主线。APSIM 支持间作（多作物共享土壤，冠层光截获按高度分层仲裁）。

Agri-JAX 从第一天就把作物当作状态的一个维度：
- 状态 `state.crop[n_crop]`，每种作物有自己的物候、生物量、根系分布；
- **光**：多层冠层截获，按各作物高度和 LAI 分层用 Beer 定律分配 PAR（APSIM 的做法）；
- **水和氮**：土壤各层是共享资源，各作物按根长密度和需求比例竞争（仲裁函数是一个可替换的过程）；
- 单作是 `n_crop = 1` 的特例，不需要单独代码路径。

`vmap` 下 `n_crop` 是静态形状，间作不增加编译复杂度。轮作（时间上的多作物）通过管理事件表实现，与间作正交。

## 6. PoC（第一阶段，目标 4 周）

范围：**RZWQM 水分平衡 + Shuttleworth-Wallace PET + CERES-Maize，单站 CA-TPA（玉米）**。

验收标准：
1. 同一组参数下，逐日 ET、LAI、各层土壤含水量、产量与 RZWQM2 参考模型输出对比，定量容差（初定：日 ET RMSE < 0.1 mm/d，产量差 < 2%）。
2. `vmap` 10 万组参数（现成的 `parameter.csv`）在一块 GPU 上的墙钟时间，对比 narval/rorqual 上 Fortran 的核时。
3. 对 6 个土壤水力参数的梯度数值稳定（有限差分核对），能跑通一次梯度标定或 NUTS。

不做：大豆/小麦、氮循环、SHAW、大孔隙流、管理事件之外的任何东西。

## 7. 验证方法：对照参考模型输出

每个过程按已发表的公式写成符合 3.1 规则的纯函数，再分三步验证：

1. **单元测试**：守恒、单调、边界、有限差分梯度；闭式关系（如 Brooks-Corey θ(h)、K(h)）核对到 1e-10。
2. **模块对照**：单独驱动一个模块（例如给定 PET 和根系吸水时的土壤水剖面），与参考模型输出逐日对比。
3. **整模型对照**：同一组参数下逐日对比 ET、LAI、各层土壤含水量、产量，容差写成数字（初定：日 ET RMSE < 0.1 mm/d，产量差 < 2%）；差异归因写进验证报告。

作物部分对照 DSSAT-CSM（公开、可复现），土壤水与 ET 对照 RZWQM2（私下运行，只报告数字）。

## 8. 路线图

| 阶段 | 产出 | 去处 |
|---|---|---|
| PoC（1 个月） | 第 6 节验收报告 | 内部决策 |
| 框架 + 首批模型（6 个月） | pip 包、文档、验证报告、参考模型对照工具 | GMD 或 Environmental Modelling & Software |
| 应用（之后） | 多站点联合梯度标定与参数可辨识性；机理-ML 混合模型在留出站点上的表现 | Nature Food / Nature Sustainability 级别取决于结果 |

## 9. 开放问题

- PoC 的 RZWQM2 对照：RZWQM2 不随本项目分发，对照在私下运行；公开的验证报告只放对比数字。需确认 USDA 对"按公开公式独立实现"的态度。
- CERES-Maize 以 DSSAT-CSM 哪个版本为准：最新主线，还是 RZWQM 内嵌的旧版（便于和现有 RZWQM 结果对齐）？倾向主线，旧版差异写进报告。
- 管理事件（播种、收获、施肥、灌溉）在 `scan` 里的表示：逐日展开的事件表是最简单的，但灌溉决策（自动灌溉）依赖状态，需要作为过程处理。
- 状态定义的稳定性：状态字段一旦公开就难改，PoC 阶段刻意不承诺。
- 名字：`agri-jax` 在 PyPI 是否可用，未查。

## 10. 目录规划（尚未创建）

```
Agri_JAX/
├── README.md              ← 本文件
├── docs/                  ← 设计文档、验证报告
├── agri_jax/
│   ├── core/              ← 状态定义、process 装饰器、运行时（scan/vmap）
│   ├── processes/         ← 过程函数库：soil_water/, pet/, crop/ceres_maize/, ...
│   ├── models/            ← 用过程拼出的模型：rzwqm_water_maize, dssat_maize, ...
│   ├── io/                ← DSSAT / RZWQM 参数与气象文件读取
│   ├── calib/             ← 标定、敏感性、UQ 顶层接口
│   └── report/            ← 面向农经的一页报告
├── tests/                 ← unit / integration / gpu / benchmark 四层
└── poc/                   ← 第 6 节 PoC 的脚本与结果
```
