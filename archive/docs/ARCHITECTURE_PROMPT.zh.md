# 给架构图 Agent 的完整 Prompt

请为开源研究项目 **FlyArm：MaleCNS-Gated Franka Control** 绘制一张可编辑、可审计的技术架构图。读者是具身智能、机器人控制和计算神经科学研究人员。图的核心不是“果蝇风格的装饰”，而是清楚回答：真实 MaleCNS 数据在哪里进入控制器，观测与动作如何被强制经过果蝇图，Franka 物理任务如何运行，以及当前证据到底支持什么。

## 事实边界

当前实现使用 MaleCNS v1.0 官方连接组导出的确定性子图：256 个真实 Body ID、4,678 条真实有向突触边。它不是整个果蝇脑的生理仿真，也不声称这些神经元天然会控制机械臂。项目研究假设是：固定的真实拓扑是否能成为控制先验；这仍未被证实。

机械臂必须画成 **Franka Emika Panda**。模型来自版本锁定且经校验的 Google DeepMind MuJoCo Menagerie；任务在 MuJoCo 中运行。4 cm 自由刚体立方体只通过重力、摩擦和双指接触运动，无 weld/equality attachment，仿真 step 不直接改物体 qpos。

最新实测必须标成负基线：脚本 teacher 在 24/24 held-out episodes 完成抓取、抬升、搬运、释放和稳定放置；受限 MaleCNS 学习策略抓取 6/24、抬升 0/24、放置 0/24。因此只能说任务与管线可行，不能说果蝇网络已经学会 pick-and-place，也不能说真实拓扑优于对照。

## 推荐版式

用横向五区结构：

1. Official data + provenance
2. Restricted neural interface
3. Learning
4. Live closed loop
5. Causal evaluation + evidence boundary

第 4 区最大，展示实时闭环。底部增加一条“Next experiments”虚线带。输出可编辑 SVG 或 draw.io，加一张 PNG 预览。浅色科学仪器风：近白背景、白色面板、冷灰蓝细边框、深海军蓝正文；输入神经元用琥珀色、内部神经元用珊瑚色、输出神经元用青色。不要使用暗色主题，不要让装饰性果蝇图片压过接口和信息流。

## 1. Official data + provenance

左侧画官方数据源：**Janelia / Cambridge MRC LMB / Google Research — MaleCNS v1.0**。输入包含 annotations、connectome weights、neurotransmitter predictions、somaLocation。

进入校验模块：固定 GCS generation、SHA256、schema、int64 Body ID。再进入确定性子图选择，输出：

- 256 measured neurons
- 4,678 directed synaptic edges
- graph SHA256 `7a5018c5481d14307e1efec305f4f448648545e5feda7caf9297dab518f4da5d`

边方向是 presynaptic → postsynaptic。contacts 决定强度；ACh 为正，GABA/glutamate 为负，其他为零，这是工程简化，不是完整受体模型。somaLocation 只用于可视化的真实解剖坐标。

## 2. Restricted neural interface

这一部分必须画得最明确，避免误解成“把果蝇图贴在普通控制器旁边”。

观测只能写入图中 23 个真实 `ascending_neuron` 节点；动作只能从 43 个真实 `descending_neuron` 节点读出。两组 Body ID 固定、互斥、与图 fingerprint 绑定。实测有 42/43 个输出能从输入沿真实有向路径到达。

画出严格数据流：

`37-D observation → trainable 37→23 encoder → 23 ascending nodes → frozen signed MaleCNS recurrent graph → 43 descending nodes → trainable 43→4 readout → 4-D action`

不得画 observation→action、encoder→readout 或普通 hidden-state shortcut。只有 encoder 和 readout 可训练，内部图权重冻结。需要注明：ascending/descending 是用于机械臂接口的工程代理，不代表果蝇神经元的天然 Panda 功能。

37 维特权状态包括：Panda q/qdot 14 维；末端、立方体、目标 XYZ 共 9 维；cube−ee 与 goal−cube 共 6 维；夹爪开度 1 维；立方体线速度 3 维；左右接触 2 维；历史 grasp/lift flag 2 维。不得画入 teacher stage 或 teacher action。

## 3. Learning

脚本 teacher 通过与策略完全相同的四维动作接口产生真实 MuJoCo 轨迹：approach、descend、contact-gated close、lift、transport、lower、open、retreat。teacher 只用于数据和独立 sanity baseline，不在 learned evaluation 中在线纠偏。

训练流程：完整 episode 划分 train/validation/test → 只用训练集统计归一化 → phase-balanced behavior cloning → 可选 DAgger。DAgger 中策略独立访问状态，teacher 只为 policy-visited states 生成离线标签，再聚合重训；held-out rollout 不调用 teacher。

当前配置：96 train episodes、16 validation、24 held-out test；两轮 DAgger，每轮 24 policy rollouts；单训练 seed。必须标注这是初始基线，不是充分统计比较。

## 4. Live Franka closed loop

画出真正在线循环：

`MuJoCo Panda + free cube → 37-D state → restricted MaleCNS policy → [Δx, Δy, Δz, gripper] → damped-least-squares IK + original finger actuator → joint-position servos → MuJoCo dynamics/contact → new state`

控制 20 Hz；物理步长 2 ms，每个动作推进 25 个 physics steps。XYZ 是有界 Cartesian increment；末端姿态固定；gripper 通过 Menagerie 原始执行器开合。动作更新 actuator control，不直接写关节或物体姿态。

成功必须同时满足：曾有双指接触；曾抬升至少 6 cm；最终 XY 目标误差 <3 cm；夹爪打开且脱离接触；物体落在桌面且速度 <0.06 m/s；稳定条件持续 10 个控制步。

实时 UI 与该循环相连，而非 replay：WebSocket 传递 MuJoCo 状态；Reset/Run/Pause/Step 是真实 server command；cube/goal 可拖动并重置 episode；神经元可点击查看 Body ID、type、superclass 和 live activation；网络与 Franka 视图都可 orbit/zoom。Three.js Franka 是 live MuJoCo body positions 的轻量渲染，物理仍在 MuJoCo 中。

## 5. Causal evaluation + evidence boundary

并列画四个同数据、同种子、同动作接口的学习策略：

1. Restricted measured MaleCNS graph
2. Restricted directed-degree-preserving shuffled graph
3. Exact trainable-parameter-matched MLP
4. Near-parameter-matched GRU

再画两类 sanity/causal check：teacher 与 zero-action；以及 measured policy 的 post-training edges-silenced、state-reset-every-step。edges silenced 保持 encoder/readout 不变，只清零递归边；不是重新训练的 disconnected baseline。

证据面板要把三个命题分开：

- **Task feasibility: supported** — teacher 24/24 place，zero 0/24。
- **Graph-mediated successful control: not supported** — intact measured policy 0/24 lift、0/24 place。
- **Measured topology advantage: not supported** — measured graph没有优于 shuffle/MLP/GRU。

实测表格：MaleCNS grasp/lift/place = 6/24, 0/24, 0/24；shuffle = 8/24, 0/24, 0/24；MLP = 19/24, 19/24, 1/24；GRU = 20/24, 9/24, 0/24。不要把运动画面、单次抓到物体或 teacher 轨迹升级为果蝇网络成功证据。

## Next experiments（全部虚线）

第一优先级是解决 contact/lift/release phase transition learning，同时保持受限接口，不添加 teacher stage/action。随后做多 seed 复验、数据量曲线、重新训练的 disconnected baseline、扰动/OOD 目标、RGB 感知、更大图或高效神经动力学后端。硬件 Franka 必须另做限幅、碰撞、急停和独立安全审查，不能从当前仿真图直接连到真实机械臂。

## 开源复用标注

明确标出实际复用：MaleCNS 官方数据（CC BY 4.0）、Google DeepMind MuJoCo Menagerie Panda（Apache 2.0）、MuJoCo、Gymnasium、PyTorch、FastAPI、React、Three.js。FlyArm 自建部分是数据校验/子图与接口绑定、受限策略、物理任务、训练评测协议、实时 server 和研究 UI。不要把上游资产归为 FlyArm 自研，也不要把未来候选库画成已集成。
