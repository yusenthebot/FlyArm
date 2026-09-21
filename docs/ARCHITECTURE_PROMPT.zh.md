# 给架构图 Agent 的完整 Prompt

请为开源研究项目 **FlyArm：Connectome-Informed Arm Control** 绘制一张可编辑的技术架构图。读者是具身智能、机器人控制和计算神经科学研究人员。希望一眼看清「真实果蝇连接组如何成为可学习机械臂控制器的一部分」，同时准确区分已实现 MVP 与后续研究路线。不要虚构成功率、性能优势、全脑仿真或生物学结论。

## 研究问题与当前边界

研究假设：在输入/输出适配器可训练、内部连接图固定的条件下，真实果蝇连接拓扑是否能在样本效率、闭环稳定性或扰动鲁棒性方面提供有效先验？这是待验证假设，不是已有结论。

第一阶段实际实现：MaleCNS v1.0 的确定性 256 节点子图，4,678 条有向边；连续 leaky-tanh 隐状态网络；MuJoCo 中的 Franka Panda；固定末端姿态、夹爪张开、局部三维到达。观察使用仿真特权状态，包括真实目标相对位置。不是 RGB-only，也不是抓取，更不是整个果蝇脑的生理精确复现。

## 版式

横向技术架构图，分为四个区域：A 数据与图准备；B 训练；C 在线控制闭环；D 公平评测。在线闭环应占最大面积。底部留一条虚线 Future Work 区域。请输出可编辑 SVG 或 draw.io 文件和 PNG 预览，保留清晰的图例。

实线框/实线箭头表示当前 MVP；虚线表示尚未实现的扩展。颜色区分外部开源资产、冻结的图计算、可训练模块、控制/物理仿真、评测。不要仅依赖颜色表达信息。

## A. 数据与图准备

外部数据源：**Janelia / Google / Cambridge / MRC — MaleCNS v1.0**。三个输入框：body annotations、connectome weights、neurotransmitter predictions。

三者流入「固定 GCS generation + SHA256 / schema 校验」，再到「保留原始 int64 神经元 ID；剔除自环和少于 3 contacts 的边；从 DNa02、DNg13、DNge104、DNp01 类型种子按邻接 contact 总量确定性选子图」。明确输入类型是解剖种子，不意味着这些细胞天然控制 Panda。

输出真实子图与可复现的数据/选择清单。边从 presynaptic 指向 postsynaptic；contacts 决定强度；ACh 为 +1，GABA / glutamate 为 -1，其余为 0；这是简化符号假设，不含受体模型。按每个目标节点的传入绝对强度归一化。将 256 nodes / 4,678 edges 标成此次实例，而非固定的项目上限。

这里分支出「directed degree-preserving edge swaps」生成随机化对照图：精确保留每节点入度、出度、源节点符号和归一化前出强度；不保持传入加权强度、空间局部性或 motifs。真实图与随机图都经过相同归一化。

## B. 训练

「解析 Cartesian P-controller teacher」通过与策略完全相同的动作接口产生真实 MuJoCo 轨迹。按完整 episode 划分 train / validation / test，不按单帧随机打散；观察归一化只拟合训练集。

「Sequence behavior cloning + truncated BPTT」只更新输入编码器和输出读出层；图的邻接矩阵是 frozen buffer。验证集 imitation MSE 选择 checkpoint；测试集只报告性能、不选模型。每 episode 重置隐藏状态。

训练轨迹箭头进入策略训练模块；梯度箭头仅指向 encoder / readout，不指向固定图。教师只在采集和独立 teacher baseline 中出现，**不得画成学习策略评测时的补救/纠偏控制器**。

## C. 在线控制闭环（主图）

从「MuJoCo + Google DeepMind Menagerie Panda」输出 20 维观测：q(7)、qdot(7)、末端 xyz(3)、target − ee(3)。流向「训练集统计归一化」→「Trainable Linear Encoder, 20 → N」→「Frozen Signed Connectome Recurrent Core」→「Trainable Linear Readout + tanh, N → 3」→「Bounded Cartesian displacement Δxyz」→「Damped Least-Squares IK + fixed orientation」→「joint-position servos / joint command limits」→「MuJoCo dynamics」，再由仿真状态反馈到观测形成真正闭环。

图核公式可简写：h[t] = 0.5 h[t−1] + 0.5 tanh(W_in x[t] + 0.8 A h[t−1])。A 固定且方向为 A[post, pre]。输出 a ∈ [-1, 1]^3，位移为 0.02 a 米，每个轴限幅；不是向量范数 2 cm 限幅。无绕过图状态直接到输出的支路，但输入适配器可以学习强行为映射，因此需做图消融。

控制周期 20 Hz，物理步长 2 ms，每次动作推进 25 次 MuJoCo dynamics。动作更新的是 actuator commands，不能画成直接改关节位置来播放动画。成功条件：末端距离目标小于 2 cm 并连续保持 5 个控制步。这里的限幅只是仿真实验约束，不要标为硬件认证安全控制器。

## D. 公平评测

画成共享数据/任务/动作接口下的并列策略对照：

1. 真实连接组固定图策略。
2. 逐节点入出度保持的随机化图策略。
3. MLP：输入和输出层尺寸相同，训练参数精确匹配。
4. GRU：调整隐藏维数，使训练参数近似匹配；实际参数必须报告，不写完全一致。

补充独立「teacher」和「zero-action」环境 sanity baselines；真实图训练后的「edges silenced」消融保持已训练 encoder/readout 不变，只清零邻接矩阵，以检查闭环行为是否依赖递归边。这不是重新训练的 disconnected baseline，应明确标注。

共享测试 episode seeds，3 个训练/图随机种子。指标包括持续达标成功率、末端误差、动作变化、推理延迟、训练参数、训练用量、1 cm xyz 观测噪声测试。请不要把单次演示或初期小样本均值画成统计显著优势。

## Future Work（全部虚线）

样本量曲线 / 更多种子 → 扰动与 OOD 目标评测 → 更大子图、LIF 或全脑高效后端 → RGB 感知和操作任务 → 受限硬件验证。FlyGM / FLYNN / Shiu 等列为论文方法参考，Stable-Baselines3 为可能的后续 RL 复用，MLX 仿真器为可能的 Apple Silicon 后端；不能画成当前已经直接集成。

## 开源复用标注与风格

实际复用：MaleCNS 官方数据（CC BY 4.0）、Google DeepMind Menagerie 的 Panda 模型（Apache 2.0）、MuJoCo、Gymnasium、PyTorch。FlyArm 自建部分：数据适配/校验、子图选择与随机化、策略桥接、最小 reaching 环境、训练与评测协议。不要把上游成果归为 FlyArm 自研。

风格：白底、克制的学术配色、正文可读、对齐整齐、箭头少交叉。中英混排允许；模块短标题英文，关键研究边界中文。避免装饰性果蝇脑图片压过接口和信息流。
