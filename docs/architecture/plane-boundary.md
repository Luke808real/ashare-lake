# 与下游策略系统的 Plane 边界

ashare-lake 在 `ARCHITECTURE_CONVERGENCE_V01` 协议里是 **Data Plane**：它回答
"市场事实上发生了什么"，不回答 "这个 setup 好不好" 或 "该不该优先看"。

## 本仓拥有

- 数据采集、source adapter、source fallback、落湖与增量编排；
- corporate action、历史 ST、停牌、交易日历、raw OHLCV、分钟线、复权事实；
- provenance、data quality、data availability、PIT 数据边界。

## 本仓明确不拥有（禁止进入本仓的语义）

- B1/B2 逻辑、R9 score、`setup_stage`、SECOND_LAUNCH 成功标签；
- watchlist ranking、TradePlan、入场/失效价、支撑/压力语义。

即：**Data Plane ≠ Strategy Plane**。本仓只对事实和口径负责；策略判断属于
Runtime 仓库（`a-share-limit-pullback`），策略真相属于 Brain 仓库
（`a-share-strategy-brain`）。

## 面向 Runtime 的边界约定

- Runtime 通过 canonical 查询契约消费本仓（`ashare_lake.query` 与对应
  adapter），不直接依赖本仓内部表结构或文件布局；
- 单条 canonical 行保持单一 provider lineage，禁止跨 provider 拼接字段；
- 缺数、冲突、覆盖不全、UNKNOWN 状态必须显式暴露并 fail-closed，本仓不静默
  回退；
- 本仓 schema 只增不改；破坏性变更需要版本 bump 与迁移说明，并经过
  Runtime 侧适配器兼容验证。

## 当前授权状态（2026-08-13）

Runtime 侧 ASL 集成仍为技术合入：`ST_READY=NO`、`PROVENANCE_GAP=OPEN`、
`PRODUCTION_CUTOVER=NO_GO`。本边界文档只定义职责协议，不构成生产切换、
provider 删除或任何策略授权。

相关文档：[设计原则](design-principles.md) · [架构总览](overview.md) ·
[决策记录](../adr/)
