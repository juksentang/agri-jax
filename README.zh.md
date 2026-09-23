# Agri-JAX — 计划零号

> 可微分、可批量的作物-土壤过程模型框架，面向参数标定、不确定性量化和机理-机器学习混合建模。
> GPU 是副产品，可微分和 batch 才是目的。

状态：构想阶段（2026-09-23）。本文件是第一版计划，所有内容都可以推翻。

**详细文档（2026-09-23 补充）**：

| 文档 | 内容 |
|---|---|
| [docs/zh_cn/02_architecture.md](docs/zh_cn/02_architecture.md) | 项目架构：目录、State/Params/Forcing、`@process`、Model、runtime、io、calib、port |
| [docs/zh_cn/03_development_plan.md](docs/zh_cn/03_development_plan.md) | 开发规划：第 0 周脚手架、PoC 逐周任务与里程碑、第二/三阶段、风险 |
| [docs/zh_cn/04_porting_and_diff_testing.md](docs/zh_cn/04_porting_and_diff_testing.md) | 移植方法学与差分测试手册：两个 oracle、七步流程、容差三级、整模型验证 |
| [docs/zh_cn/05_maintenance_pipeline.md](docs/zh_cn/05_maintenance_pipeline.md) | 维护管线：仓库/许可、工具链、lint、测试分层、CI、集群分工（rorqual GPU / narval Fortran）、发布 |
| [docs/zh_cn/06_open_source_ecosystem.md](docs/zh_cn/06_open_source_ecosystem.md) | 开源生态与选型：可直接用的库、设计参考、与 diffWOFOST 等的尺度区分 |
| [docs/zh_cn/08_throughput_comparison.md](docs/zh_cn/08_throughput_comparison.md) | 大规模批量耗时对比：Fortran 实测 vs H100 骨架实测 |

**对本文假设的更正**：RZWQM2 4.6 的 Fortran 源码在 narval `sharing/RZWQM_Linux_Ver45/`，二进制在本机 16 s 跑完 CA-TPA 九年；dssat-csm-os 4.8.5 源码与静态二进制在本机 `~/AFSoil`；CA-TPA 的 10 万次 LHS 已全部跑完；`agri-jax` 名字可用；GPU 作业去 rorqual H100 而非 narval。

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

**什么不是目的**：让农学家写 JAX；把单块田的模拟跑得更快；做一个通用 Fortran→JAX 编译器。

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

这三条比 Fortran 的 common 块和 OOP 框架（APSIM NG、PCSE）的事件回调都简单：每个过程读什么、改什么，签名上一目了然。DSSAT 之所以难维护，根因是无接口、无测试、全局状态，不是语言；这三条规则正是补这三样。

### 3.2 三层接口

| 层 | 用户 | 内容 |
|---|---|---|
| 顶层 | 用模型的人 | `load_dssat_experiment()`、`calibrate(model, obs, method="hmc")`、`sensitivity()`、`ensemble()` |
| 中层 | 建模者 | 模型 = 状态定义 + 有序过程列表；可替换任一过程 |
| 底层 | 我们 | 运行时（scan/vmap/jit）、差分测试、移植工具 |

预期 90% 用户停在顶层和参数文件。

### 3.3 兼容现有参数文件（最强的采用钩子）

直接读 DSSAT 的 `.CUL / .ECO / .SPE / .SOL / .WTH` 和 RZWQM 的 `rzwqm.dat`，农学家几十年积累的品种参数一行不改就能用。

### 3.4 状态是带名字的 pytree

状态用 `NamedTuple` / `flax.struct` 定义，字段有名字和单位，`state.lai`、`state.sw[layer]`，不是数组下标。带 `n_crop` 维度（见第 5 节），单作只是 `n_crop = 1`。

### 3.5 精度与数值

- 验证阶段 `jax_enable_x64`，对照 Fortran 双精度。
- 数据相关的迭代（Richards 方程收敛、根系吸水迭代）改为固定迭代数或固定步长隐式格式，接受与 Fortran 的小差异，差异写进验证报告。

## 4. 基础模型选择：DSSAT-CSM 作物模块 + RZWQM 土壤/ET（按公式重写）

| 候选 | 优点 | 问题 |
|---|---|---|
| 原版 DSSAT-CSM | 源码公开（GitHub `DSSAT/dssat-csm-os`），版本新，社区大，品种文件生态完整 | 土壤水是 tipping bucket，ET 选项简单，没有 Richards、大孔隙、SHAW |
| RZWQM2（组里的 Linux 版） | 土壤物理强（Richards + Green-Ampt、大孔隙、SHAW 能量平衡、S-W PET），氮磷碳循环完整，组里的标定流程都建在它上面 | 源码受限分发（USDA-ARS），内嵌的 DSSAT 作物模块是旧版；开源移植需许可 |

**决定**：
- 作物模块从 **DSSAT-CSM 开源仓库** 移植（先 CERES-Maize），版本新且许可干净。
- 土壤水、PET、能量平衡按 **RZWQM 的公式**（Ahuja 等 2000《Root Zone Water Quality Model》及相关论文）重写，不翻译组里那份源码。公式是公开的，实现是我们的。
- 土壤水模块做成可插拔：`tipping_bucket`（复现 DSSAT）和 `richards`（复现 RZWQM）两套，同一作物模块可以对照两种土壤物理，本身就是一个可发表的比较。
- 验证对象两个：DSSAT-CSM 二进制（作物部分）和组里的 RZWQM 二进制（土壤 + 作物整体，用现有 15 个 scenario）。

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
1. 同一组参数下，逐日 ET、LAI、各层土壤含水量、产量与 RZWQM 二进制对比，定量容差（初定：日 ET RMSE < 0.1 mm/d，产量差 < 2%）。
2. `vmap` 10 万组参数（现成的 `parameter.csv`）在一块 GPU 上的墙钟时间，对比 narval/rorqual 上 Fortran 的核时。
3. 对 6 个土壤水力参数的梯度数值稳定（有限差分核对），能跑通一次梯度标定或 NUTS。

不做：大豆/小麦、氮循环、SHAW、大孔隙流、管理事件之外的任何东西。

## 7. 移植方法学：半自动翻译 + 差分测试

不做通用 Fortran→JAX 编译器（`goto`、common 块、数据相关的 `do while` 无法自动变成可 vmap 的纯函数）。做的是可复用的流水线：

1. 用 Fortran 前端（LFortran / fparser）拆子程序，生成调用图和每个子程序的读/写变量清单。
2. 逐子程序用 LLM 翻译为符合 3.1 规则的过程函数。
3. **差分测试**：在 Fortran 二进制里插桩，真实模拟中 dump 每个子程序的输入/输出，作为对应 JAX 函数的单元测试。翻译错误定位到子程序级。
4. 自底向上组装，最后整模型逐日对比。

这套流水线本身可复用到其他 Fortran 作物模型（STICS、WOFOST、老版 APSIM），是方法学贡献的一部分。

## 8. 路线图

| 阶段 | 产出 | 去处 |
|---|---|---|
| PoC（1 个月） | 第 6 节验收报告 | 内部决策 |
| 框架 + 首批模型（6 个月） | pip 包、文档、验证报告、差分测试工具 | GMD 或 Environmental Modelling & Software |
| 应用（之后） | 多站点联合梯度标定与参数可辨识性；机理-ML 混合模型在留出站点上的表现 | Nature Food / Nature Sustainability 级别取决于结果 |

## 9. 开放问题

- PoC 的 Fortran 对照：RZWQM 二进制不能开源，但用它做私有验证没有问题；公开的验证报告只放对比数字。需确认 USDA 对"按公式重写"的态度。
- CERES-Maize 从 DSSAT-CSM 哪个版本移植：最新主线，还是 RZWQM 内嵌的旧版（便于和现有 RZWQM 结果对齐）？倾向主线，旧版差异写进报告。
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
│   └── port/              ← Fortran 拆解、差分测试插桩与比对
├── tests/
│   └── diff/              ← 对 Fortran dump 的差分测试
└── poc/                   ← 第 6 节 PoC 的脚本与结果
```
