# LEGO Lean Production Simulation - 参数与流程文档

## 目录
1. [整体生产流程](#整体生产流程)
2. [阶段参数详解](#阶段参数详解)
3. [缓冲区参数](#缓冲区参数)
4. [团队参数](#团队参数)
5. [关键变量](#关键变量)
6. [事件类型与逻辑](#事件类型与逻辑)
7. [KPI指标](#kpi指标)
8. [全局参数](#全局参数)

---

## 整体生产流程

### 流程图
```
S1 (Type Sorting) 
  ↓
B (Warehouse B)
  ↓
S2 (Set Sorting) 
  ↓ (概率路由)
  ├─→ C1 (40%) → S3 (Axis Assembly) → D1 ─┐
  ├─→ C2 (40%) → S4 (Chassis Assembly) → D2 ┼→ S5 (Final Assembly) → E
  └─→ C3 (20%) ────────────────────────────┘
```

### 流程说明
1. **S1 (Type Sorting)**: 源阶段，无输入缓冲区，直接生产并推送到 B
2. **B (Warehouse B)**: 中间缓冲区，存储 S1 的输出
3. **S2 (Set Sorting)**: 从 B 拉取，按概率路由到 C1 (40%)、C2 (40%)、C3 (20%)
4. **S3 (Axis Assembly)**: 从 C1 拉取，输出到 D1
5. **S4 (Chassis Assembly)**: 从 C2 拉取，输出到 D2
6. **S5 (Final Assembly)**: 多输入阶段，需要同时从 D1、D2、C3 拉取，输出到 E
7. **E (Finished Gliders)**: 最终产品缓冲区

### 生产策略
- **Push 模式**: 订单在 t=0 时一次性释放到源阶段（S1），系统按物料可用性推进
- **无 CONWIP**: 不使用全局 WIP 上限控制
- **多输入同步**: S5 需要等待所有输入（D1、D2、C3）都可用才能开始

---

## 阶段参数详解

### S1: Type Sorting (类型分拣)

| 参数 | 值 | 说明 |
|------|-----|------|
| `stage_id` | `"S1"` | 阶段标识符 |
| `name` | `"Type Sorting"` | 阶段名称 |
| `team_id` | `"T1"` | 所属团队 |
| `input_buffers` | `[]` | **源阶段，无输入** |
| `output_buffer` | `"B"` | 输出到缓冲区 B |
| `base_process_time_sec` | `2.5` | **每工人每件处理时间（秒）** |
| `workers` | `2` | **工人数量** |
| `time_distribution` | `{"type": "triangular", "p1": 2.0, "p2": 2.5, "p3": 4.0}` | 时间分布：三角分布（最小值2.0，众数2.5，最大值4.0） |
| `transport_time_sec` | `0.2` | 运输时间（秒） |
| `defect_rate` | `0.00` | 缺陷率（0%） |
| `rework_stage_id` | `None` | 无返工阶段 |

**实际处理时间计算**:
- 基础时间 = `base_process_time_sec / workers` = 2.5 / 2 = 1.25 秒/件
- 总时间 = 处理时间（从分布采样）+ 运输时间（0.2秒）
- 示例：采样值 2.5 秒 → 实际 = 2.5 / 2 + 0.2 = 1.45 秒/件

---

### S2: Set Sorting (套装分拣)

| 参数 | 值 | 说明 |
|------|-----|------|
| `stage_id` | `"S2"` | 阶段标识符 |
| `name` | `"Set Sorting"` | 阶段名称 |
| `team_id` | `"T2"` | 所属团队 |
| `input_buffers` | `["B"]` | 从缓冲区 B 拉取 |
| `output_rules` | `[{"buffer_id": "C1", "p": 0.40}, {"buffer_id": "C2", "p": 0.40}, {"buffer_id": "C3", "p": 0.20}]` | **概率路由规则** |
| `base_process_time_sec` | `3.0` | 每工人每件处理时间（秒） |
| `workers` | `2` | 工人数量 |
| `time_distribution` | `{"type": "triangular", "p1": 2.0, "p2": 3.0, "p3": 5.0}` | 三角分布 |
| `transport_time_sec` | `0.3` | 运输时间（秒） |
| `defect_rate` | `0.01` | 缺陷率（1%） |
| `rework_stage_id` | `"S2"` | 返工回到自身 |

**路由模式**:
- **随机模式** (`routing_mode="random"`): 按概率随机选择输出缓冲区
- **确定性模式** (`routing_mode="deterministic"`): 使用轮询算法平衡输出，避免下游饥饿

**实际处理时间**:
- 基础时间 = 3.0 / 2 = 1.5 秒/件
- 总时间 = 采样值 / 2 + 0.3 秒

---

### S3: Axis Assembly (轴装配)

| 参数 | 值 | 说明 |
|------|-----|------|
| `stage_id` | `"S3"` | 阶段标识符 |
| `name` | `"Axis Assembly"` | 阶段名称 |
| `team_id` | `"T3"` | 所属团队 |
| `input_buffers` | `["C1"]` | 从缓冲区 C1 拉取 |
| `output_buffer` | `"D1"` | 输出到缓冲区 D1 |
| `base_process_time_sec` | `4.0` | 每工人每件处理时间（秒） |
| `workers` | `2` | 工人数量 |
| `time_distribution` | `{"type": "triangular", "p1": 3.0, "p2": 4.0, "p3": 6.0}` | 三角分布 |
| `transport_time_sec` | `0.4` | 运输时间（秒） |
| `defect_rate` | `0.02` | 缺陷率（2%） |
| `rework_stage_id` | `None` | 无返工（缺陷品报废） |

**实际处理时间**:
- 基础时间 = 4.0 / 2 = 2.0 秒/件
- 总时间 = 采样值 / 2 + 0.4 秒

---

### S4: Chassis Assembly (底盘装配)

| 参数 | 值 | 说明 |
|------|-----|------|
| `stage_id` | `"S4"` | 阶段标识符 |
| `name` | `"Chassis Assembly"` | 阶段名称 |
| `team_id` | `"T4"` | 所属团队 |
| `input_buffers` | `["C2"]` | 从缓冲区 C2 拉取 |
| `output_buffer` | `"D2"` | 输出到缓冲区 D2 |
| `base_process_time_sec` | `4.0` | 每工人每件处理时间（秒） |
| `workers` | `2` | 工人数量 |
| `time_distribution` | `{"type": "triangular", "p1": 3.0, "p2": 4.0, "p3": 6.0}` | 三角分布 |
| `transport_time_sec` | `0.4` | 运输时间（秒） |
| `defect_rate` | `0.02` | 缺陷率（2%） |
| `rework_stage_id` | `None` | 无返工（缺陷品报废） |

**实际处理时间**:
- 基础时间 = 4.0 / 2 = 2.0 秒/件
- 总时间 = 采样值 / 2 + 0.4 秒

---

### S5: Final Assembly (最终装配)

| 参数 | 值 | 说明 |
|------|-----|------|
| `stage_id` | `"S5"` | 阶段标识符 |
| `name` | `"Final Assembly"` | 阶段名称 |
| `team_id` | `"T5"` | 所属团队 |
| `input_buffers` | `["D1", "D2", "C3"]` | **多输入：需要同时从 D1、D2、C3 拉取** |
| `output_buffer` | `"E"` | 输出到最终缓冲区 E |
| `base_process_time_sec` | `6.0` | 每工人每件处理时间（秒） |
| `workers` | `3` | 工人数量 |
| `time_distribution` | `{"type": "triangular", "p1": 5.0, "p2": 6.0, "p3": 9.0}` | 三角分布 |
| `transport_time_sec` | `0.5` | 运输时间（秒） |
| `defect_rate` | `0.03` | 缺陷率（3%） |
| `rework_stage_id` | `None` | 无返工（缺陷品报废） |

**多输入同步逻辑**:
- S5 只有在 **所有输入缓冲区都有库存** 时才能开始处理
- 如果任一输入不足，阶段进入**饥饿状态**（starvation），等待所有输入就绪

**实际处理时间**:
- 基础时间 = 6.0 / 3 = 2.0 秒/件
- 总时间 = 采样值 / 3 + 0.5 秒

---

## 缓冲区参数

| 缓冲区ID | 名称 | 容量 | 初始库存 | 说明 |
|---------|------|------|---------|------|
| `B` | Warehouse B (post-TypeSorting) | 999 | 30 | S1 的输出缓冲区 |
| `C1` | C1 (Axis Parts) | 999 | 0 | S2 的输出（40%），S3 的输入 |
| `C2` | C2 (Chassis Parts) | 999 | 0 | S2 的输出（40%），S4 的输入 |
| `C3` | C3 (Final Assembly Only Parts) | 999 | 0 | S2 的输出（20%），S5 的输入 |
| `D1` | D1 (Axis Subassembly) | 999 | 0 | S3 的输出，S5 的输入 |
| `D2` | D2 (Chassis Subassembly) | 999 | 0 | S4 的输出，S5 的输入 |
| `E` | E (Finished Gliders) | 999 | 0 | **最终产品缓冲区**（finished_buffer_ids） |

**缓冲区操作**:
- `can_pull(qty)`: 检查是否有足够库存可拉取
- `pull(qty)`: 从缓冲区拉取指定数量
- `can_push(qty)`: 检查是否有足够容量可推送
- `push(qty)`: 向缓冲区推送指定数量
- 容量为 `None` 表示无限容量

---

## 团队参数

| 团队ID | 名称 | 规模 | 班次 | 说明 |
|--------|------|------|------|------|
| `T1` | Type Sorting Team | 2 | day | 负责 S1 |
| `T2` | Set Sorting Team | 2 | day | 负责 S2 |
| `T3` | Axis Team | 2 | day | 负责 S3 |
| `T4` | Chassis Team | 2 | day | 负责 S4 |
| `T5` | Final Assembly Team | 3 | day | 负责 S5 |

**团队利用率跟踪**:
- `busy_time`: 累计忙碌时间（模拟时间单位）
- `last_busy_start`: 当前忙碌开始时间
- 利用率 = `busy_time / sim_time`

**注意**: 团队规模（`size`）仅用于可见性，实际处理速度由阶段的 `workers` 参数控制。

---

## 关键变量

### 仿真状态变量

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `self.t` | `float` | **当前模拟时间（秒）** |
| `self._queue` | `List[Event]` | **事件队列**（优先队列，按时间排序） |
| `self._evt_seq` | `int` | 事件序列号（用于打破时间平局） |

### 缓冲区状态

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `buffer.current` | `int` | **当前库存数量** |
| `buffer.capacity` | `Optional[int]` | 容量限制（None = 无限） |

### 阶段状态

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `stage.busy` | `bool` | **阶段是否忙碌**（True = 正在处理，False = 空闲） |
| `stage.workers` | `int` | **工人数量**（影响处理速度） |

### WIP 与订单跟踪

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `self.current_wip` | `int` | **当前在制品数量**（已释放但未完成的订单） |
| `self.started` | `int` | 已释放的订单总数 |
| `self.finished` | `int` | **已完成的产品数量**（推送到 finished_buffer_ids 的数量） |
| `self.source_stage_orders` | `Dict[str, int]` | **源阶段的待处理订单数**（key=stage_id, value=订单数） |

### KPI 累积变量

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `self.lead_times` | `List[float]` | 每个完成产品的提前期（秒） |
| `self.wip_time_area` | `float` | WIP×时间的积分（用于计算平均 WIP） |
| `self.last_wip_time` | `float` | 上次 WIP 更新时间 |

### 阶段级计数器

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `self.stage_completed_counts` | `Dict[str, int]` | **每个阶段完成的次数** |
| `self.starvation_counts` | `Dict[str, int]` | **每个阶段饥饿的次数**（等待输入） |
| `self.blocking_counts` | `Dict[str, int]` | **每个阶段阻塞的次数**（输出缓冲区满） |

### 路由状态（确定性模式）

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `self._served_counts` | `Dict[str, Dict[str, int]]` | 每个阶段对每个输出缓冲区的服务次数 |
| `self._rr_index` | `Dict[str, int]` | 每个阶段的轮询索引 |

### 时间线采样

| 变量名 | 类型 | 说明 |
|--------|------|------|
| `self.timeline` | `List[Dict[str, Any]]` | **时间序列快照**（每个采样点的 WIP、完成数、吞吐量、缓冲区水平） |
| `self._sample_dt` | `float` | 采样间隔（秒，默认 5.0） |
| `self._next_sample_t` | `float` | 下次采样时间 |

---

## 事件类型与逻辑

### 事件类型

| 事件类型 | 说明 | 触发时机 |
|---------|------|---------|
| `try_start` | 尝试在阶段开始处理 | 订单释放、阶段完成、下游唤醒 |
| `complete` | 阶段完成处理 | 处理时间到期 |
| `time_advance` | 时间推进（内部） | 每次事件处理时自动触发 |

### 事件处理流程

#### 1. `try_start` 事件处理 (`_on_try_start`)

**逻辑步骤**:
1. 检查阶段是否忙碌 → 如果忙碌，延迟 0.001 秒后重试
2. **源阶段检查**: 如果无输入缓冲区，检查 `source_stage_orders[stage_id] > 0`，否则返回
3. **多输入检查**: 对于每个 `input_buffers`，尝试从缓冲区拉取 1 单位
   - 如果任一输入不足 → 回滚已拉取的输入，增加 `starvation_counts`，延迟 0.5 秒后重试
4. **开始处理**:
   - 标记 `stage.busy = True`
   - 启动团队利用率跟踪 (`team.start_busy(t)`)
   - 计算处理时间: `base_process_time_sec / max(1, workers)` → 从分布采样 → 加上运输时间
   - 检查随机中断（缺失积木）→ 如果发生，增加惩罚时间
   - 调度 `complete` 事件在 `t + 处理时间` 触发

#### 2. `complete` 事件处理 (`_on_complete`)

**逻辑步骤**:
1. 释放团队 (`team.stop_busy(t)`)
2. **缺陷处理**:
   - 增加self.stage_defect_counts，
   - 如果发生缺陷（概率 = `defect_rate`）:
     - 如果有 `rework_stage_id` → 触发返工阶段的 `try_start`
     - 否则 → 报废（`current_wip -= 1`）
   - 如果无缺陷 → 继续输出
4. **输出路由**:
   - 如果 `output_buffer` 存在 → 使用该缓冲区
   - 如果 `output_rules` 存在:
     - **随机模式**: 按概率随机选择
     - **确定性模式**: 使用平衡算法（选择服务次数/概率比值最小的输出）
5. **推送输出**:
   - 尝试推送到选定的输出缓冲区
   - 如果缓冲区满 → 增加 `blocking_counts`，延迟 0.5 秒后重试 `complete`
   - 如果成功:
     - 如果输出缓冲区在 `finished_buffers` 中 → `finished += 1`, `current_wip -= 1`, 记录提前期
     - 增加 `stage_completed_counts[stage_id]`
     - 更新路由服务计数（确定性模式）
6. **唤醒下游**:
   - 标记 `stage.busy = False`
   - **源阶段**: 如果 `source_stage_orders[stage_id] > 0`，立即触发 `try_start`
   - **非源阶段**: 立即触发 `try_start`（尝试处理下一件）
   - **下游阶段**: 对于消费选定输出缓冲区的阶段，如果它们**不忙碌**，触发它们的 `try_start`

#### 3. 班次约束

- 如果事件时间不在班次窗口内 → 推迟到下一个班次开始时间
- 班次检查: `_is_in_shift(t)` 和 `_advance_to_next_shift_start(t)`

---

## KPI指标

### 核心 KPI

| KPI | 计算公式 | 说明 |
|-----|---------|------|
| `throughput_per_sec` | `finished / sim_time` | **吞吐量**（件/秒） |
| `lead_time_avg_sec` | `sum(lead_times) / len(lead_times)` | **平均提前期**（秒） |
| `wip_avg_units` | `wip_time_area / sim_time` | **平均在制品**（件） |
| `utilization_per_team` | `team.busy_time / sim_time` | **团队利用率**（每个团队） |
| `finished_units` | `self.finished` | **完成产品数** |
| `started_units` | `self.started` | **已释放订单数** |
| `service_level` | `self.finished / self.started` | **服务等级** |
| `defect_rate_per_stage` | `self.stage_defect_counts / self.stage_completed_counts + self.stage_defect_counts` | **各阶段次品率** |
| `total_defect_rate` | `total_defects / total_processed` | **生产线次品率**|

### 阶段级 KPI

| KPI | 说明 |
|-----|------|
| `stage_completed_counts` | **每个阶段完成的次数**（字典：`{stage_id: count}`） |
| `starvation_counts` | **每个阶段饥饿的次数**（等待输入材料） |
| `blocking_counts` | **每个阶段阻塞的次数**（输出缓冲区满，无法推送） |
| `defect_defect_counts` | **每个阶段次品的次数** 

### 时间序列数据 (`timeline`)

每个采样点包含:
- `t`: 时间戳（秒）
- `wip`: 当前在制品数量
- `finished`: 累计完成数
- `throughput_per_min`: 每分钟吞吐量（自上次采样）
- `B`, `C1`, `C2`, `C3`, `D1`, `D2`, `E`: 各缓冲区的当前库存

---

## 全局参数

### 仿真参数 (`parameters`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `target_takt_sec` | `10.0` | 目标节拍时间（秒） |
| `timeline_sample_dt_sec` | `5.0` | 时间线采样间隔（秒） |
| `finished_buffer_ids` | `["E"]` | **完成产品缓冲区列表**（推送到这些缓冲区的产品计入 `finished`） |
| `routing_mode` | `"random"` | **路由模式**（`"random"` 或 `"deterministic"`） |

### 班次安排 (`shift_schedule`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `shift_id` | `"day"` | 班次标识符 |
| `start_minute` | `480` | 开始时间（一天中的分钟数，8:00 = 480） |
| `end_minute` | `960` | 结束时间（一天中的分钟数，16:00 = 960） |

### 随机中断 (`random_events`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `missing_brick_prob` | `0.10` | **缺失积木概率**（每个操作） |
| `missing_brick_penalty_sec` | `2.0` | **缺失积木惩罚时间**（秒，额外处理时间） |

---

## 时间分布类型

支持的时间分布类型（`time_distribution.type`）:

| 类型 | 参数 | 说明 |
|------|------|------|
| `constant` | - | 恒定时间（`base_process_time_sec / workers`） |
| `triangular` | `p1=low, p2=mode, p3=high` | 三角分布 |
| `normal` | `p1=μ, p2=σ` | 正态分布 |
| `lognormal` | `p1=μ, p2=σ` | 对数正态分布 |
| `uniform` | `p1=a, p2=b` | 均匀分布 |
| `exponential` | - | 指数分布（λ = 1 / base_time） |

**注意**: 所有分布的基础值都是 `base_process_time_sec / workers`（每工人时间）。

---

## 处理时间计算公式总结

对于阶段 *i*，实际处理时间 *S<sub>i</sub>* 的计算：

```
S_i = T_i + Δ_i + Z_i
```

其中:
- **T<sub>i</sub>**: 从分布 `D_i` 采样的处理时间，基础值为 `τ_i / w_i`
  - `τ_i` = `base_process_time_sec`（每工人每件时间）
  - `w_i` = `workers`（工人数量）
- **Δ<sub>i</sub>**: `transport_time_sec`（运输时间，不受工人数影响）
- **Z<sub>i</sub>**: 随机中断惩罚（概率 = `missing_brick_prob`，惩罚 = `missing_brick_penalty_sec`）

**示例**:
- S1: `base_process_time_sec=2.5`, `workers=2`, `transport_time_sec=0.2`
  - 基础处理时间 = 2.5 / 2 = 1.25 秒/件
  - 从三角分布采样（例如 2.5 秒）→ 实际 = 2.5 / 2 = 1.25 秒
  - 总时间 = 1.25 + 0.2 = 1.45 秒/件

---

## 关键设计决策

1. **Push 模式**: 订单在 t=0 时一次性释放，系统按物料可用性推进
2. **工人参数**: 处理时间与工人数成反比（`time_per_unit = base_time / workers`）
3. **多输入同步**: S5 需要等待所有输入（D1、D2、C3）都可用
4. **确定性路由**: 可选模式，使用平衡算法避免下游饥饿
5. **源阶段订单跟踪**: 使用 `source_stage_orders` 跟踪待处理订单，避免无限重试
6. **完成产品识别**: 推送到 `finished_buffer_ids` 的产品计入 `finished`
7. **班次约束**: 事件自动推迟到下一个班次开始时间

---

## 使用建议

1. **调整工人数量**: 通过 `workers` 参数平衡产能和成本
2. **监控饥饿/阻塞**: 使用 `starvation_counts` 和 `blocking_counts` 识别瓶颈
3. **路由平衡**: 在 S2 使用确定性路由模式避免 C1/C2/C3 不平衡
4. **时间分布**: 使用三角分布模拟实际变异性
5. **初始库存**: 设置 B 的初始库存（如 30）可以快速启动下游流程

---

*文档版本: v1.0*  
*最后更新: 2024*

