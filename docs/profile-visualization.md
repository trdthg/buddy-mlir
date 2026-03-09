# Profile 可视化设计说明

## 1. 目标

这套可视化的目标不是“把所有算子画出来”，而是同时满足三件事：

- 保持原始数据依赖关系正确，不引入伪环
- 能按源码模块层级阅读大模型
- 在需要时仍然能看到真实输入、真实输出和热点信息

当前主路径产物有两类：

- flat profile 图：看原始 DAG 和热点
- hierarchy SVG：看模块层级和子图接口

当前的 hierarchy SVG 是主视图。

## 2. 数据来源

可视化只依赖两份输入数据：

1. `graph.json`
2. `profile.json`

它们分别提供：

- `graph.json`
  - `nodes[*].id`
  - `nodes[*].name`
  - `nodes[*].kind`
  - `nodes[*].shape`
  - `nodes[*].dtype`
  - `nodes[*].parents`
  - `nodes[*].profile_name`
  - `nodes[*].provenance`
  - `edges[*].source_id`
  - `edges[*].target_id`
  - `source_context`
- `profile.json`
  - `records[*].op_name`
  - `records[*].avg_ms`
  - `records[*].percentage`
  - `total_avg_ms`

其中最关键的约束是：

- 图边只认 `source_id/target_id`
- 绝不再按节点名字推边

这是为了避免重名节点把不同 layer 的边串错。

## 3. 总体流水线

当前流水线分成四层：

1. `graph_report.py`
   - 合并 graph 与 profile
   - 计算 `FLOPs / bytes / intensity`
   - 解析 provenance
2. `hierarchy_model.py`
   - 从 provenance 建立 path-based hierarchy tree
3. `hierarchy_scene.py`
   - 把 hierarchy tree 投影成某一个 scene 的显示图
   - 处理边界节点、输入输出节点、core 节点、防伪环
4. `hierarchy_svg.py`
   - 把 scene 渲染成 DOT/SVG
   - 处理节点标签、颜色、交互元数据

CLI 入口在：

- `tools/buddy_tools/profile_viz/visualize_profile.py`

## 4. 我们对节点的定义

这里有两个层次的“节点”。

### 4.1 原始节点

原始节点就是 `graph.json` 里的 Buddy graph node。

典型字段：

- `id`
- `name`
- `kind`
- `shape`
- `dtype`
- `provenance`
- `profile`
- `estimates`

这是唯一有真实数据依赖语义的节点。

### 4.2 显示节点

显示节点是 scene 投影之后得到的节点，只存在于可视化层。

当前有几类：

1. `module`
   - 某个源码模块路径对应的模块节点
   - 例如 `self_attn · Qwen2Attention`

2. `module-core`
   - 某个模块路径下、没有落到更深子模块上的直属原始节点集合
   - 例如 `self_attn core`

3. `node`
   - 直接显示的原始算子节点
   - 例如 `iota`

4. `external-io`
   - 当前 scene 外部、但和当前 scene 有连接关系的显示节点
   - 包括：
     - 根场景的 `输入` / `输出` summary
     - deeper scene 里的显式输入节点
     - deeper scene 里的显式输出节点
     - mixed boundary 节点

## 5. 如何对节点分组

当前 hierarchy 主分组只按 **源码模块路径** 做，不按 class 做。

class 只用于标签显示，不用于主分组。

### 5.1 分组依据

分组主要依赖 `provenance.nn_module_stack`。

对每个原始节点，会先提取：

- `block_path`
- `submodule_path`
- `leaf_path`
- `source_class`

### 5.2 分组规则

规则按优先级如下：

1. `PlaceholderOp`
   - 归到根层的 `输入`

2. `OutputOp`
   - 归到根层的 `输出`

3. 有 provenance 且能解析出模块路径的普通节点
   - 进入对应 path 的模块树

4. 一个节点如果只属于某个模块，但不属于更深子模块
   - 归到该模块的 `core`

5. 没有可用模块路径的节点
   - 归到 `未归类`

### 5.3 一个 path 只能出现一次

这是一个硬约束：

- 同一个 `source_path`
- 不能一边作为容器节点出现
- 一边又作为 sibling leaf 再出现一次

这是为了防止之前那种：

- `self_attn`
- `self_attn · Qwen2Attention`

同时并列显示，最终导致投影伪环。

## 6. Scene 投影规则

Hierarchy tree 不是直接画出来的。  
真正画出来的是某一个 `focus_group_key` 对应的 scene。

### 6.1 根场景

根场景是总览。

当前根场景默认显示：

- `输入`
- 顶层模块
- `未归类`
- `输出`

其中：

- 根层的 `输入` / `输出` 仍然是 summary 节点
- 不把所有 placeholder 全摊开

这是为了让第一层保持规整。

### 6.2 普通模块场景

普通模块场景显示三类内容：

1. 直接子模块
2. 当前模块直属 raw 节点
3. 当前模块的显式输入边界节点、显式输出边界节点

这里“直属 raw 节点”的意思是：

- 属于当前模块
- 但不属于任何更深的子模块

### 6.3 I/O 详情场景

根层的 `输入` / `输出` 可以点进去。

在 I/O 详情场景里：

- 会显示真实 raw input / output 节点
- 会显示它们和顶层 peer group 的连接

## 7. 连线逻辑

### 7.1 原始边

原始边来自 Buddy graph 的数据依赖：

- `source_id -> target_id`

这是唯一可信的边关系。

### 7.2 Scene 边的投影

在 scene 里，原始边会被重新映射到显示节点之间。

投影规则：

1. `inside -> inside`
   - 变成 scene 内普通边

2. `outside -> inside`
   - 变成输入边界节点到内部节点的边

3. `inside -> outside`
   - 变成内部节点到输出边界节点的边

### 7.3 防伪环规则

scene 必须保持 DAG。

当前防伪环主要靠三条规则：

1. 不再按名字解边
2. display node membership 不重叠
3. 如果某个 `module-core` 聚合后引入伪环
   - 自动把该 `core` 展开回 raw 节点

这也是为什么现在图虽然节点有时会多一点，但整体更规整。

## 8. 输入节点、输出节点、边界节点

### 8.1 输入节点

当前有两种：

1. 根层 summary `输入`
2. deeper scene 里的显式输入边界节点

显式输入边界节点表示：

- 当前 scene 外部某个真实 raw 节点
- 它把结果送进了当前 scene

### 8.2 输出节点

同理，输出节点表示：

- 当前 scene 内部某个结果被送到 scene 外部

当前 deeper scene 中，输出侧是显式拆开的，不是一个总的 `输出` port。

### 8.3 mixed boundary 节点

如果同一个外部 raw 节点同时：

- 给当前 scene 提供输入
- 又接收当前 scene 的输出

就会显示成一个 mixed boundary 节点，而不是重复画两份。

## 9. 节点标签规则

### 9.1 普通 raw 节点

当前会显示：

- 名字
- 角色标记（如果是输入/输出/本地输入源）
- `kind`
- 输入形状
- 输出形状
- profile 信息
- estimate 信息

例如：

- `iota`
- `输入节点`
- `IotaOp`
- `输出形状=[1024]`

### 9.2 输入边界节点

当前会显示：

- `输入节点`
- `连接=N`
- `来自=...`
- `连接到=...`
- `输出形状=...`

### 9.3 输出边界节点

当前会显示：

- `输出节点`
- `连接=N`
- `来自=...`
- `流向=...`
- `输入形状=...`

### 9.4 模块节点

模块节点显示：

- 模块标签
- class 名
- 节点数
- 输入形状摘要
- 输出形状摘要
- 时间 / FLOPs / bytes / intensity

## 10. 颜色设计

当前颜色设计刻意把“角色”和“来源”分开。

### 10.1 节点填充色

主要表达“角色”：

- 根层 `输入`：蓝系
- 根层 `输出`：橙系
- 普通模块：浅蓝底
- raw 普通节点：浅灰底
- `未归类`：浅黄底

### 10.2 节点边框色

主要表达“来源分组”。

#### 输入侧

1. 顶层外部输入 `PlaceholderOp`
   - 整体统一蓝色

2. 由计算节点产生、再作为当前 scene 输入的边界节点
   - 按“来源模块”分色

#### 输出侧

一个子图里的所有输出边界节点：

- 统一使用当前子图自己的颜色

也就是说：

- 同一个 scene 里，无论输出多少个边界节点
- 输出颜色都是一致的

### 10.3 本地输入源节点

如果某个 raw 节点在当前 scene 中：

- 没有任何上游
- 但有下游

就把它视为 scene 内部的本地输入源。

这类节点：

- 会被打上 `输入节点`
- 会获得稳定颜色
- 它发出的边也会使用同一种颜色

例如：

- `iota`

### 10.4 连线颜色

当前连线颜色规则：

1. 输入边界相关的边
   - 跟对应输入来源颜色一致

2. 输出边界相关的边
   - 跟当前 scene 的输出颜色一致

3. 模块块自身发出的边
   - 优先按 `edge_color_key` 着色

4. 其它普通内部边
   - 保持灰色

### 10.5 为什么不用固定 8 色桶

之前用离散 8 色调色板时，不同来源容易碰撞成同色。

现在改成：

- 对 `color_key` 做 hash
- 生成连续色相
- 并避开外部输入蓝、输出橙附近色域

这样：

- 不同模块的输出边更不容易撞色
- 例如 `rotary_emb` 和 `model core` 就不会再碰巧同色

## 11. 关键中间数据结构

### 11.1 GraphReport

表示 `graph.json + profile.json` 合并后的图。

核心内容：

- enriched raw nodes
- raw edges
- profile
- estimates
- provenance

### 11.2 GroupCollection

表示 path-based hierarchy tree。

核心内容：

- `groups`
- `groups_by_key`
- `children_by_group`
- `root_keys`

### 11.3 Scene

表示某一层具体要画的子图。

核心内容：

- `scene_id`
- `title`
- `subtitle`
- `display_nodes`
- `edges`
- `parent_scene_id`

## 12. 当前刻意放弃的方案

以下方案试过，但不再作为主路径：

1. 前端浏览器主导渲染
   - 交互强
   - 但边布局质量不如 Graphviz
   - 最终放弃

2. 主路径按 class 分组
   - 类名适合作为标签
   - 但不适合作为 hierarchy 主键

3. 主路径 repeated/template folding
   - 对某些模型能压缩
   - 但容易引入额外复杂性和伪环
   - 当前主路径放弃

4. flat HTML 报告
   - 维护成本高
   - 已删除

## 13. 当前边界和后续方向

当前这版已经能满足：

- 模块层级清晰
- 子图输入输出明确
- scene 无伪环
- 颜色能表达来源分组

但仍有几个边界：

1. provenance 本身可能不完整
   - 个别节点仍可能进入 `未归类`

2. group 节点上的 shape 仍然是摘要，不是逐输入逐输出精确枚举

3. 颜色语义目前主要服务可读性，不是严格的数据语义编码系统

如果后面继续演化，优先级应该是：

1. 继续修 provenance 传播
2. 提升 placeholder / argument 的语义命名
3. 保持 hierarchy scene 的 DAG 性质不被破坏

## 14. 当前结论

这套设计的核心选择是：

- 主分组按 path，不按 class
- 图边只按 id，不按名字
- 根层适度折叠，deeper scene 显式输入输出
- 输入颜色按来源分组，输出颜色按子图统一
- scene 内本地源节点显式标成 `输入节点`
- 渲染权交给 Graphviz，交互保持轻量

这就是当前 profile hierarchy SVG 的正式设计基线。
