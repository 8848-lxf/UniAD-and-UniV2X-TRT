# 4090 UniAD-tiny TensorRT 修复记录

本文档只记录 RTX 4090 上 UniAD-tiny 实权重部署链路的已知问题、修复方法和可复核证据。时间戳统一使用运行进程的 PDT（UTC-07:00），不使用系统的 Asia/Shanghai 时区。

---

## 2026-08-13T17:05:00-07:00 (PDT) - 量化边界与 occupancy 全零问题复核

### 已知问题

- 旧说明把异常点写成“两个 Mul 算子保留 FP16”，容易被理解为对两个 Mul 节点做算子级排除。官方配置实际只全局排除 `MatMul`，并未默认排除 `Mul`。
- 168-feed 原始 INT8 QDQ 图的 `seg_out` 全为 0，导致 collision optimizer 没有障碍栅格可查询，raw 与 optimized planning 轨迹逐字节一致。

### 根因与修复

- 异常对象是 occupancy 末端同一个 `Mul` 的两个 activation Q/DQ 输入边，不是两个 Mul 算子。
- 两路激活的量化 scale 乘积使反量化后的最大可表达值低于后续 `Greater(..., 0.1)` 阈值，因此该分支确定性输出全 0；这不是正常量化误差，也不是碰撞优化“偶尔不触发”。
- 最终混合精度图只旁路这两个末端 activation Q/DQ 边，保留对应计算为 FP16；其余显式 QDQ 覆盖和全局 `MatMul` 排除规则不变。
- runtime 增加逐帧 collision optimizer 审计和可选 `seg_out` bit-pack dump，能够区分“没有检测到风险”与“occupancy 分支失效”。

### 验收规则

- `seg_out` 不能全 0。
- 必须记录 raw/optimized 轨迹是否改变、优化触发次数和碰撞事件，而不能仅凭最终 avg. Col 推断后处理是否工作。

---

## 2026-08-13T18:10:00-07:00 (PDT) - 时序协议和 planning MSE 统计口径修复

### 已知问题

- 旧评估混用了两种递归协议：官方 literal 脚本只在全 validation 的 `sample_id == 0` 初始化外部 track/BEV/timestamp/ego-pose；scene 切换主要以 `use_prev_bev=0` 屏蔽旧 BEV。另一套实现则在每个 scene 首帧清空全部外部状态。
- 使用一种协议生成 PyTorch reference、另一种协议运行 TensorRT，会把时序输入差异错误计入 FP32 engine 误差。
- `planning MSE` 名称有歧义：NVIDIA 文档正文描述的是平均点 L2，但官方表中的 FP32 数值更接近平方距离量纲。

### 修复

- metadata、校准准备、PyTorch audit、TensorRT runtime 和评估工具均增加显式 `official_literal` / `scene_reset` 协议参数及 manifest。
- 比较工具读取 sidecar manifest；协议不一致时直接失败，不再产生看似合法的 planning 差值。
- PyTorch audit 逐帧保存 raw planning、packed occupancy、scene 边界及 frame provenance。
- planning 评估同时输出三个不混淆的量：
  - `mean_point_l2_m = mean(sqrt(dx^2 + dy^2))`
  - `coordinate_mse_m2 = mean([dx^2, dy^2])`
  - `mean_squared_point_distance_m2 = mean(dx^2 + dy^2)`

### 200 帧同协议回归

| 协议 | FP32 mean point L2 | coordinate MSE | mean(dx²+dy²) |
| --- | ---: | ---: | ---: |
| official literal | 0.00333184 m | 1.34448e-5 m² | 2.68896e-5 m² |
| scene reset | 0.00336922 m | 1.37592e-5 m² | 2.75184e-5 m² |

固定 track capacity 1150 与 1600 的前 200 帧输出逐字节一致，静态 padding 不是该误差的根因。

---

## 2026-08-13T19:35:00-07:00 (PDT) - C++ JPEG 预处理一致性修复

### 根因

- C++ runtime 使用 STB 解码 JPEG，而 PyTorch/mmcv 使用 OpenCV/libjpeg-turbo。
- 即使 resize、颜色通道、归一化参数均一致，JPEG IDCT/舍入差异仍会进入 BEV 时序递归并逐帧放大。
- 修复前首帧六相机归一化输入与 PyTorch 的最大绝对误差为 `0.0348585`，等价原始像素最大相差 `2`；前 200 帧 FP32 occupancy IoU 约为 `95.3%`。

### 修复

- `pre_process.hpp` 增加 libjpeg decoder，实现与 OpenCV/libjpeg-turbo 一致的 JPEG 解码；CMake 增加 `UNIAD_USE_LIBJPEG_DECODER` 与 `UNIAD_JPEG_ROOT`。
- 构建只使用 Conda 隔离工具链和 Conda 缓存的 libjpeg-turbo：`/home/lixingfeng/anaconda3/pkgs/libjpeg-turbo-2.0.0-h9bf148f_0`，没有调用系统 CUDA 或系统 G++。
- 增加 C++ 预处理张量 dump 和逐元素比较脚本，作为构建后的强制 parity gate。

### 修复后 200 帧证据

- 首帧六相机归一化输入：max abs `2.3841858e-7`，MAE `1.131e-8`，RMSE `3.270e-8`。
- FP32 raw planning 相对同协议 PyTorch：mean point L2 `0.000453936 m`，coordinate MSE `3.22812e-7 m²`，`mean(dx²+dy²) = 6.45624e-7 m²`。
- FP32 occupancy：IoU `97.0315%`，precision `98.3376%`，recall `98.6496%`；五个 horizon IoU 为 `97.7509/97.7173/97.3131/96.8776/95.9126%`。
- collision-optimized planning：avg. L2 `0.742615 m`，box Col `0.750000%`。短序列碰撞率波动较大，只用于回归，不作为全 validation 结论。
- corrected runtime：`/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt109_20260813/runtime_protocol_cvjpeg_fix_20260813/uniad`。

### 当前状态

- FP32、FP16、修正版 INT8 与同协议 PyTorch 的 6018 帧全量复评已完成；正式结果见后续 `2026-08-13T21:47:23-07:00 (PDT)` 章节。
- 旧 STB runtime 的结果保留为根因对照，不再作为最终验收结果。

---

## 2026-08-13T21:47:23-07:00 (PDT) - 6018 帧最终回归与 planning MSE 定义修正

### 评估协议

- checkpoint：`tiny_imgx0.25_e2e_ep20.pth`，不是随机 ONNX，也不是 base 权重。
- validation：6018 个 TensorRT 帧；PyTorch audit：6018 帧；150 个 scene。
- external state：`official_literal`。只有整个 validation 的第 0 帧初始化外部 track/BEV/timestamp/pose；scene 边界使用 `use_prev_bev=0`，不清空外部 track。
- runtime：TensorRT 10.9.0 编译和执行，固定 track capacity 1600，单 TensorRT execution context，collision optimization 开启。
- 输出目录：`/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt109_20260813/cvjpeg_official_literal_full6018_20260813/`。

### 官方示例原表

| Model / framework | Precision | Latency | FPS | avg. L2 | avg. Col | planning MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| UniAD-tiny / PyTorch 1.12 | FP32 | 843.5172 ms | 1.18 | 0.9986 m | 0.27% | 0 |
| UniAD-tiny / TensorRT 10.7 | FP32 | 64.0469 ms | 15.61 | 0.9986 m | 0.27% | 9.2417e-7 |
| UniAD-tiny / TensorRT 10.7 | FP16 | 49.7559 ms | 20.10 | 1.0021 m | 0.26% | 0.0458 |
| UniAD-tiny / TensorRT 10.7 | INT8(EQ)+FP16 | 39.3125 ms | 25.44 | 1.0029 m | 0.27% | 0.0502 |

官方文档把 planning MSE 文字描述为平均点 L2，但 FP32 的 `9.2417e-7` 与本地逐帧 `mean(dx²+dy²)` 的数量级一致，而不是平均点 L2 的数量级。因此本报告的主 `planning_mse` 定义为 `mean(dx²+dy²)`，同时始终保留平均点 L2 和逐坐标 MSE；评估 JSON schema 已升级为 2，避免三个量被混称。

### 本机 6018 帧结果

PyTorch reference 使用相同 official-literal 时序协议，输出为 raw `outs_planning`，不含 Python `use_col_optim` 后处理。TensorRT 的 avg. L2/Col 使用同一份 validation GT；planning MSE 使用 TensorRT raw trajectory 与该 PyTorch raw reference 的逐帧对齐结果。

| Backend | Precision | avg. L2 (GT) | avg. Col (box, GT) | planning MSE `mean(dx²+dy²)` | mean point L2 vs PyTorch | model mean / p50 / p99 (ms) | inference mean / p50 / p99 (ms) | E2E mean / p50 / p99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch audit | FP32 | 0.737383 | 0.556663% | 0 | 0 | n/a | n/a | n/a |
| TensorRT | FP32 | 0.776288 | 0.252022% | 6.432e-7 | 0.000425 | 18.585 / 18.372 / 23.137 | 22.895 / 22.648 / 29.049 | 79.152 / 78.027 / 100.532 |
| TensorRT | FP16 | 0.837544 | 0.459732% | 0.061875 | 0.032670 | 14.052 / 13.735 / 19.893 | 18.586 / 18.252 / 25.452 | 76.227 / 75.127 / 96.694 |
| TensorRT | INT8(EQ)+FP16 | 0.784751 | 0.263100% | 0.080659 | 0.097120 | 14.585 / 14.288 / 20.301 | 19.058 / 18.775 / 25.904 | 76.390 / 75.393 / 95.920 |

这里的 TensorRT avg. Col 是修复 raw trajectory 后处理后重新计算的结果；它不再把 raw `outs_planning` 误当成最终 `use_col_optim=True` 轨迹。FP32 和 INT8 的 box collision 已回到官方 `0.27%` 附近。FP16 的 occupancy 误差较大，导致碰撞优化过度，仍是需要单独修复的低精度分支问题。

### occupancy 对齐

occupancy 指标是相对同一 PyTorch 输入的 TRT occupancy 二值一致性，不是 nuScenes GT 检测精度：

| Engine | IoU | precision | recall | byte-exact frames / 6018 | horizon IoU (0.5--2.5 s) |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 | 96.897% | 98.332% | 98.516% | 1300 | 97.333 / 97.178 / 96.999 / 96.722 / 96.303% |
| FP16 | 71.279% | 84.481% | 82.018% | 1005 | 76.938 / 71.111 / 70.195 / 69.118 / 69.684% |
| INT8(EQ)+FP16 | 62.833% | 70.832% | 84.765% | 884 | 68.760 / 64.849 / 62.037 / 60.251 / 59.213% |

因此当前 FP32 planning MSE 已经不是“口径错误”造成的；FP32 的 raw trajectory 误差已经达到 `6.432e-7`，接近官方 FP32 数值。FP16/INT8 的差异则与 occupancy/时序状态低精度误差直接相关。静态 1600 padding 不是主因，前 200 帧 1150 与 1600 输出逐字节一致。

### 时延趋势结论

- FP32 -> FP16 的 model enqueue p50 从 `18.372` 降至 `13.735 ms`，E2E p50 从 `78.027` 降至 `75.127 ms`。
- 当前受保护末端 occupancy activation Q/DQ 的 INT8 engine 并没有比 FP16 更快：model enqueue p50 为 `14.288 ms`，E2E p50 为 `75.393 ms`。这是 TensorRT 10.9 在 RTX 4090 上的 tactic/混合精度结果，不是 CUDA 回退到 CPU；inference-call 仍由同一 CUDA context 同步完成。
- 因此“FP32 > FP16 > INT8”的官方延迟单调性在本机只满足 FP32 -> FP16，INT8 相对 FP16 略慢。强行移除末端 FP16 保护会重新触发 occupancy 全零，不能作为可接受加速结果。

### 量化和后处理审计

- `collision_optimization_audit` 的 `frames_modified`：FP32 `2733`、FP16 `3425`、INT8 `3047`，三档都实际执行了轨迹修正；本轮不再存在“raw 与 optimized 逐字节一致是因为 occupancy 分支全零”的假阳性。
- FP32/FP16/INT8 的正 occupancy 栅格数分别为 `4,695,134 / 4,549,747 / 5,608,223`；INT8 不再是全零分支。
- 168-feed INT8 图中异常的是末端 `Mul` 的 activation Q/DQ 边尺度，不是两个 `Mul` 算子被量化。官方全局排除的是 `MatMul`；`Mul` 没有默认排除。最终图只旁路两个末端 activation Q/DQ，保留其计算为 FP16。

### 构建链和接续修复

- 构建脚本不再引用不存在的 `repro/package/uniad-trt`；默认使用受版本控制的 `UniAD/runtime/inference_app/enqueueV3`，可用 `UNIAD_APP_ROOT` 覆盖。
- CMake 增加显式 `UNIAD_STB_ROOT`、`UNIAD_CUOSD_ROOT`、`UNIAD_BEVFORMER_TRT_ROOT`、`UNIAD_JPEG_ROOT`，运行脚本默认从 `UniAD/repro/UniAD_deploy` 启动以解析 `data/...` 相对图片路径。
- 独立 CMake 验证构建通过：Conda `nvcc 11.8.89`、Conda `g++ 11.2.0`、TensorRT 10.9；构建产物位于 `/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt109_20260813/cmake_repro_verify_20260813`。
- 两帧新构建 runtime 与全量 runtime 的 raw/optimized planning 输出一致；没有调用系统 CUDA 或系统 G++。

### 当前剩余问题

1. FP16/INT8 occupancy 与 PyTorch 的 IoU 仍不足以声称完整任务精度等价；需要对 occupancy head 的 activation scale、threshold 和状态回灌做逐层/逐 scene 校准。
2. 官方文档的 planning MSE 生成脚本未公开，正文的“平均点 L2”与 FP32 数值矛盾。本地 JSON 同时保存平方点 MSE、平均点 L2、coordinate MSE，最终验收应明确采用哪一列。
3. 本轮 TensorRT runtime 只输出 planning、occupancy 和解码框审计，尚未从 engine 输出重建完整 Python detection/tracking/map 指标，因此没有虚构 mAP/AMOTA/IoU。
4. `UniAD-and-UniV2X-TRT` 主仓库中的 `UniAD/repro/UniAD_deploy/data` 与 `third_party` 是用户已有软链接，保持未跟踪状态；不要删除或提交大数据目录。

---

## 2026-08-14T08:43:36-07:00 (PDT) - FP16 occupancy 修复路线收口

### 6018 帧 recurrent-state 候选

中断前已完成 `fp16_all_state_fp32` 推理，本轮补齐同协议 occupancy、raw planning 和 optimized planning 评估。结果没有达到修复门槛：

| Engine | occupancy IoU | raw avg. L2 / box Col | optimized avg. L2 / box Col | model p50 |
| --- | ---: | ---: | ---: | ---: |
| 普通 FP16 | 71.278528% | 0.729297 m / 0.548355% | 0.837544 m / 0.459732% | 13.735280 ms |
| FP16 + recurrent BEV/track FP32 | 71.686034% | 0.735608 m / 0.553894% | 0.842598 m / 0.462501% | 12.148512 ms |

新候选只提高 `0.407506` 个 IoU 百分点，optimized L2 和 Col 均未改善，因此标记为 rejected，不作为 FP16 正式 engine。

### scene 内误差定位

| scene offset | 帧数 | 普通 FP16 IoU | recurrent-state FP32 IoU |
| --- | ---: | ---: | ---: |
| 0 | 150 | 79.749421% | 92.783810% |
| 1-4 | 600 | 75.195962% | 75.295846% |
| 5-9 | 750 | 69.906218% | 69.902859% |
| 10-19 | 1500 | 69.645572% | 69.975025% |
| 20+ | 3018 | 71.313178% | 71.572604% |

state 保护只显著改善 scene 首帧，从下一帧开始收益消失。主要误差由每帧重新执行的共享 FP16 feature/occupancy consumer 路径产生，不是单纯的跨 scene 或长时序累积。

### 200 帧混合精度门禁

- 同一 Python builder、插件和 timing-cache 的零约束 FP16 control IoU 为 `91.598030%`，证明 builder/tactic 控制组与旧 `trtexec` FP16 的 `91.5879%` 一致。
- 仅锁定末端 Conv 为 FP32 后 IoU 为 `90.484706%`；decoder Conv 深度 `2/4/8` 分别为 `90.538580/90.445100/91.345556%`。
- occupancy+planning 独有子图 FP32 为 `90.574686%`。
- recurrent state + LayerNorm + occupancy/planning 独有子图的闭环组合达到 `92.017620%`，但仅比 state-only 高 `0.009589` 点，model enqueue p50 从 `11.984384` 增至 `14.986192 ms`。扩大保护范围没有工程收益，未启动第二轮 6018 帧。

builder 新增两项默认关闭的审计参数：`--fp32-lineage-convolutions` 会同时锁定卷积 compute/output 并记录卷积深度；`--force-fp32-exclusive-output-lineage` 按输出所有权区分 shared/exclusive ancestry。此前只设置 output dtype、在 Conv 处停止回溯的 lineage 不能表述“卷积已用 FP32 计算”。

### 阈值与空间过滤否证

- runtime 增加可选 `seg_score_out` 输出和 `UNIAD_DUMP_OCCUPANCY_SCORES=1`，用于 dump `ReduceMax(pred_ins_sigmoid)` 的 FP32 表示。
- `score > 0.1` 在 200/200 帧逐位复现 engine `seg_out`。全局最佳阈值约 `0.0990`，IoU 仅由 `91.598030%` 变为 `91.611279%`；五个 horizon 独立调阈值的最大收益也小于 `0.1` 点。
- 3x3 opening、closing、median 的 IoU 分别降到 `88.936263/84.492192/89.174446%`。错误不是阈值整体偏置，也不是可去除的孤立噪点。

### 最终部署门禁

- 普通 FP16 只保留为 raw planning 数值和时延诊断，不接受为 collision-optimized 完整部署。
- 完整 accuracy reference 使用 TensorRT FP32；需要 reduced precision 时使用已验收的显式 INT8 QDQ + occupancy 末端 FP16 保护图，同时保留其 occupancy parity 未完全对齐的限制。
- 不再以继续扩大单引擎 FP32 lineage 的方式声称“修复 FP16”；该路线已经接近 FP32 计算范围，却没有恢复 occupancy。
- 轻量证据：`UniAD/evidence/trained_tiny_fp16_occupancy_audit_20260814/summary.json`。大 engine、timing cache、score dump、packed mask 和 report 仅保存在 `/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt109_20260814`。

---

## 2026-08-14T11:22:12-07:00 (PDT) - FP16 全量候选与阈值审计

### 混合精度候选的 6018 帧结果

将 200 帧最优候选提升到 6018 帧：LayerNorm、递归 BEV 和浮点 track 输出 lineage 保持 FP32，其余仍允许 FP16。所有帧均有限，但没有通过完整部署验收：

| 指标 | 普通 FP16 | state/LayerNorm FP32 候选 |
| --- | ---: | ---: |
| occupancy IoU | 71.278528% | 71.697673% |
| raw avg. L2 / box Col | 0.729297 m / 0.548355% | 0.735522 m / 0.553894% |
| optimized avg. L2 / box Col | 0.837544 m / 0.459732% | 0.842812 m / 0.462501% |
| `mean(dx²+dy²)` vs matching PyTorch | 0.061875 | 0.006976 |
| mean point L2 vs matching PyTorch | 0.032670 m | 0.015541 m |
| model mean / p50 / p99 | 14.052 / 13.735 / 19.893 ms | 15.791 / 15.789 / 17.703 ms |

该候选证明轨迹时序数值可通过混合精度明显改善，但 occupancy、最终 L2/Col 和速度均未改善。因此 FP16 异常不是一个可由递归状态保护统一解决的问题。

### 连续分数和碰撞优化阈值

- 新增 `UniAD/repro/scripts/sweep_fp16_occupancy_thresholds.py`，读取 `seg_score_out` 连续值和 raw trajectory，复用与 C++ 一致的逐点 BFGS collision optimizer，并在完整 GT 上计算 L2/point Col/box Col。
- 200 帧回归中，阈值 `0.1` 的离线结果与 C++ runtime 完全一致，排除了扫描器统计口径错误。
- 使用正式标准插件重新导出 6018 帧连续分数。默认 `0.1` 为 `L2 0.837544 / box Col 0.459732%`；验证集 GT 上的最佳单阈值 `0.55` 为 `0.745060 / 0.354492%`。
- `0.55` 仍明显差于本机 FP32 的 `0.252022%`，而且它是用 validation GT 选出的，会改变训练/导出定义的 `score > 0.1` 语义。按 PyTorch occupancy 一致性选择时最优仍约为 `0.099`，IoU 只增加 `0.013249` 个百分点。

因此不把 `0.55` 写入正式 runtime，也不把 validation 调参结果包装成 FP16 修复。最终门禁保持：FP32 是 accuracy reference；需要低精度完整路径时使用已验收的显式 INT8 QDQ + occupancy 末端 FP16 保护图，并保留其 occupancy parity 限制。

---

## 2026-08-14T12:01:23-07:00 (PDT) - Track profile 纠正与 FP16 score 分布复核

### 1150/1600 profile 纠正

- NVIDIA 示例使用动态 profile `MIN/OPT/MAX=901/901/1150`。`1600` 是早期诊断阶段为了把每帧 track 输入固定为同一形状、避免动态 shape/Myelin 配置更新而留出的保守上界，不是官方设置；现存记录没有支持“必须取 1600”的数据依据。
- 对 official-literal PyTorch audit 的 6018 帧重新统计，实际 track count 为 `min=901`、`max=1138`、`mean=926.580924`，超过 `1150` 的帧数为 `0`。固定到 1600 时平均每帧多填充 `673.419076` 行，因此该上界没有必要。
- 既有 200 帧 A/B 已证明固定容量 1150 与 1600 的 FP32 输出逐字节一致；它不是 FP16 occupancy 异常根因。历史 6018 帧表仍明确标注其使用 1600，不回写或伪装实验协议。
- 受版本控制的 `run_build_engines.sh` 和 `run_trt_evaluation.sh` 默认值均已是官方 `1150`。后续正式重建和评估不得再用 1600，除非新的输入清单证明存在 `track_count > 1150`，并在报告中单独声明偏离官方协议。

### FP16 连续 score 分布

对标准插件 FP16 engine 的 6018 帧 `seg_score_out`（`ReduceMax` 后、进入 `Greater(..., 0.1)` 前的连续值）重新审计：

- 共 `75,225,000` 个 score；PyTorch 二值参考正栅格 `4,686,362`，FP16 正栅格 `4,551,249`。
- false positive 为 `706,802`，其 score 中位数为 `0.229126`；false negative 为 `841,915`，有限 score 中位数为 `0.004032`。这些值大多远离 `0.1`，不是简单的 FP16 阈值舍入。
- 全部 score 落在 `0.1 +/- 0.001/0.005/0.01/0.05` 内的比例分别只有 `0.0220%/0.1114%/0.2243%/1.2405%`。这与全量最优 parity 阈值约 `0.099` 只能增加 `0.013249` 个 IoU 百分点一致。
- 发现 `31,700` 个 NaN，集中在 frame `1948/1953/4825` 三帧；其中 `2,854` 个覆盖 PyTorch 正栅格，只占全部 false negative 的 `0.338989%`。NaN 必须作为 FP16 数值缺陷保留，但不足以解释整体 occupancy IoU 和 Col 差异。
- 五个 horizon 的 IoU 为 `76.9358/71.1144/70.2011/69.1212/69.6997%`，误差随未来时域扩散；这不是单一末端阈值或形态学噪点。

### 逐路径混合精度定位结论

- 200 帧 FP16 control IoU 为 `91.598030%`；插件 FP32 accumulate 为 `91.606901%`，MSDA FP32 为 `91.458717%`，均排除单个自定义插件作为主因。
- 末端 Conv FP32、occupancy decoder Conv 深度 2/4/8 的 IoU 分别为 `90.484706/90.538580/90.445100/91.345556%`，排除末端卷积舍入作为主因。
- recurrent state lineage FP32 为 `92.008031%`；再叠加 occupancy/planning 独有 lineage FP32 仅为 `92.017620%`。状态保护主要改善 scene 首帧，不能修复 scene 内后续帧。
- 因而当前根因边界位于每帧共享的 image/BEV feature producer 与 occupancy consumer 的组合数值传播，而不是 occupancy 最后一层、阈值、单个插件或单纯递归 state。继续把大范围 lineage 强制 FP32 已接近 FP32 engine 的计算范围且没有恢复 parity，不构成可验收 FP16 修复。

---

---

## 2026-08-14T15:29:17-07:00 (PDT) - FP16 precision-boundary candidates

本轮继续使用隔离的 `modelopt_uniad_dl4agx`、Conda CUDA 11.8、TensorRT 10.9.0.34、native plugin 和 `official_literal` 协议；未修改权重/数据集。

| 候选 | 状态 | 200 帧 occupancy IoU | raw avg. L2 | raw box Col |
| --- | --- | ---: | ---: | ---: |
| full output-lineage FP32（既有候选） | 可构建 | `92.3829%` | `0.721249 m` | `1.166667%` |
| PluginV2 precision 强制 FP32 | TensorRT 10.9 构建失败 | - | - | - |
| 大范围 layout precision | Myelin SSA/IR 构建失败 | - | - | - |
| 仅 `Reshape_1957/1983/1997` FP32 | 可构建但退化 | `90.3709%` | `0.738667 m` | `1.083333%` |
| 官方 profile `901/901/1150` plain FP16 | 可构建 | `91.2459%` | `0.737405 m` | `1.166667%` |

- Plugin 强制方案在 `MultiScaleDeformableAttnTRT_1998` 报 `No supported formats`，不是 OOM；大范围 Shuffle/Resize/Select 方案报 Myelin SSA verifier 错误。
- 三个入口 Shuffle 的局部保护和官方 `MAX=1150` profile 都没有改善 occupancy/Col，故不提升到 6018 帧。
- `seg_score_out` debug alias 被融合到带 Half 内部结果的 ForeignNode。TensorRT API 确认该 binding 与 `seg_out` 都声明为 `LINEAR`，因此不是 C++ wrapper 漏处理非线性 stride；但 alias 的 `score > 0.1` 与最终 `seg_out` 空间位置不逐元素一致，说明 Myelin 中间 alias 不能作为最终 `Greater` 的语义等价输出。正式比较只使用原生 `seg_out.packbits`；ONNX 阈值仍为 `0.1`，不修改模型语义。

结论：FP16 剩余差异位于共享 image/BEV feature producer 到 occupancy consumer 的融合数值传播，尚未找到在 TensorRT 10.9 上既可构建又能恢复 occupancy parity 的单引擎修复。FP16 仍未通过正式 Col 验收；FP32 继续作为 accuracy reference。

---

## 2026-08-14T21:01:37-07:00 (PDT) - direct logits 与时序放大定位

- Builder 新增默认关闭的 `--mark-output-direct-alias SOURCE=ALIAS`，直接将原始中间 tensor 设为 network output。FP16/FP32 direct-score engines 的 `score > 0.1` 均与各自 `seg_out` 在 200 帧、250 万栅格上逐 bit 一致。
- 有效阈值 sweep 的最佳点 `0.107` 只把 FP16 对 PyTorch occupancy IoU 从 `91.149326%` 提高到 `91.188137%`。
- FP16 对 FP32 的有效 score MAE 为 `0.001581457`，abs error p99/p99.9/max 为 `0.04099321/0.24271484/0.87691808`；occupancy IoU `91.849044%`，共 `5,676` flips，五个 horizon 为 `958/992/1143/1221/1362`。
- scene 首帧平均只有 `1.0` flip，非首帧为 `29.2268`。禁用外部 temporal state 后 flips 降到 `749`、IoU 提高到 `95.263091%`，证明时序传播是主要放大器，但该模式不是正式评估协议。
- 仅保护 `prev_bev` 七个入口层后仍有 `5,659` flips，对 FP32 IoU `91.854624%`，没有改善。根因继续收窄到更深的 shared temporal encoder/BEV feature 传播，而非 `prev_bev` 入口 reshape/rotate。

本轮修复了 logits 审计路径，没有修复正式 FP16 engine 精度；所有候选均未通过 200 帧门禁，因此不重复 6018 帧评估。

---

## 2026-08-14T21:29:22-07:00 (PDT) - FP16 中间特征二分与 tactic 否证

### 新增的默认关闭审计接口

- enqueueV3 runtime 增加 `UNIAD_DUMP_BEV_EMBED`，可逐帧写出真正回灌到下一帧的 `bev_embed`。
- 增加通用 `audit_out` 槽和 `UNIAD_DUMP_AUDIT_OUTPUT`，与 Builder 的 direct output alias 配套，只在诊断 engine 中导出固定形状中间 tensor。正式 engine 不包含 `audit_out`，环境变量默认关闭，不改变正式输出或时序回灌。
- 新增 `UniAD/repro/scripts/compare_tensor_audit.py`，在比较前校验 manifest、字节数、shape、帧数和 temporal protocol，避免再次使用语义不可靠的 Myelin Identity alias。

### 时序入口候选

仅保护 11 个 `prev_track_intances*` 直接消费 Mul 后结果退化：对 PyTorch occupancy IoU `90.629807%`，对 TRT FP32 `91.314118%`，flips `6,030`，raw planning `0.738576 m / 1.083333% box Col`。结合上一轮 prev_bev 入口的 `91.854624% / 5,659 flips`，可以排除两个外部状态入口的 reshape/乘法单点。

### FP16/FP32 中间张量对照

| 边界 | 帧数 | shape/frame | MAE mean | relative RMSE mean | cosine mean/min |
| --- | ---: | --- | ---: | ---: | ---: |
| recurrent `bev_embed` | 200 | `2500x1x256` | `0.00621349` | `1.489651%` | `0.999335 / 0.908129` |
| dense decoder `future_states.3` | 40 | `1x5x256x50x50` | `0.04092132` | `3.259410%` | `0.999070 / 0.986162` |

`future_states.3` 五个未来 horizon 的 MAE 为 `0.033794/0.035245/0.038626/0.042905/0.054038`，呈单调增长；相同 40 帧中 FP16/FP32 最终 occupancy IoU 为 `95.729326%`、flips `1,148`。因此误差在 `bev_embed` 之后进入 dense future feature 路径时已经约翻倍，并继续随未来时域放大。该结果把根因收窄到 dense future feature 与 occupancy query 的组合传播，但尚未证明唯一异常 layer。

### optimization level A/B

现有正式和混合精度候选均为 builder optimization level 3；历史文件名 `native_opt906` 中的 `opt906` 是 profile opt shape，不是优化级别。本轮新建 level-0 纯 FP16 engine：

- 对 PyTorch occupancy IoU `90.116329%`；
- 对 TRT FP32 occupancy IoU `90.790114%`，`6,432` flips；
- raw/optimized planning 均为 `0.738535 m / 1.083333% box Col`。

该候选比 level-3 标准 FP16 更差，排除 level-3 特定 tactic/Myelin 优化强度是主因。所有过程继续使用隔离的 `modelopt_uniad_dl4agx`、Conda CUDA 11.8、TensorRT `10.9.0.34`、官方 profile `901/901/1150` 和 official-literal 协议；固定 1150 只用于已经验证等价的诊断加速。

最终状态不变：审计工具已修复且误差边界进一步收窄，但没有候选通过 200 帧门禁，因此不重新跑 6018 帧，也不宣称 FP16 occupancy/Col 已修复。正式 accuracy reference 仍为 FP32。

---

## 2026-08-14T21:44:55-07:00 (PDT) - Dense decoder 完整约束候选

### decoder 入口直接审计

在 FP16/FP32 direct-output engine 上导出 `input.1743`，使用相同 200 帧、official-literal 和固定 1150 诊断协议：

| tensor | shape/frame | MAE mean | relative RMSE mean | cosine mean/min |
| --- | --- | ---: | ---: | ---: |
| `bev_embed` | `2500x1x256` | `0.00621349` | `1.489651%` | `0.999335/0.908129` |
| `input.1743` | `1x256x13x13` | `0.00648140` | `0.870129%` | `0.999598/0.938283` |
| `future_states.3` | `1x5x256x50x50` | `0.04092132` | `3.259410%` | `0.999070/0.986162` |

`input.1743` 诊断引擎的最终 occupancy IoU 为 `91.606570%`、`5,831` flips。入口相对误差低于 BEV 输出，而 decoder 出口误差显著增加，故主要放大位于 dense future decoder 内。

### 严格 FP32 decoder 候选

旧 `fp16_occ_shared96_fp32` 的 `max_convolutions=0` 导致 Conv/Resize 只进入 lineage 报告而未被设为 FP32，其 `90.393891%` IoU 不能用于证明“完整 decoder FP32”。本轮重新构建：

- root tensor：`future_states.3`；
- lineage depth：`48`；
- convolution depth：`5`；
- layout precision：开启；
- precision constraint：严格 `OBEY`；
- 实际 FP32：21 个 Conv、相关 Resize/Shuffle、合计 376 个数值层；
- engine SHA256：`baffbef44575601b55a957828da3fc7d957c8c8111df5a6d9779288205a61af4`。

200 帧结果仍退化：对 PyTorch occupancy IoU `90.753124%`，对 TRT FP32 `91.448952%`、`5,934` flips；raw/optimized 均为 `0.738131 m / 1.083333% box Col`。model enqueue mean/p50/p99 为 `11.495/11.311/19.507 ms`。

因此“可观测误差在 decoder 放大”不等于“把 decoder 强制 FP32 就能恢复”；TensorRT 融合边界变化会改变周边数值路径，严格候选也未通过门禁。当前没有可验收的单引擎 FP16 修复，不进行 6018 帧长跑，正式 FP16 全量表保持原值和未验收标记。

---

## 2026-08-14T23:11:14-07:00 (PDT) - 原始 IPOPT 后处理与 3x3 交叉消融

### 为什么存在本地 BFGS

- NVIDIA 公布的 tiny Python 配置设置 `use_col_optim=True`，标准 Python 测试使用 CasADi/IPOPT 的 `CollisionNonlinearOptimizer`。
- ONNX 导出使用的 `PlanningHeadSingleModeTRT.forward_trt` 只返回 raw trajectory；其 collision optimization 代码被注释。NVIDIA C++ sample 的 `post_process.hpp` 也只有 `TODO: collision optimization`，README 明确说明 sample 未实现 collision correction。
- 因此本地 C++ BFGS 是为了补上部署后处理而实现的同目标函数近似，不是 NVIDIA C++ sample 原有实现。此前将其结果直接称为官方等价 `use_col_optim=True` 不够严谨。

### 完整核查协议

- 输入：同一次 `official_literal` 6018 帧 FP32/FP16/INT8 raw trajectory 与 `seg_out.packbits`。
- 环境：已有且未修改的 `torch112`，CasADi `3.5.6`/IPOPT，8 个 CPU worker，不占 GPU。
- 组合：3 档 raw × 3 档 occupancy，共 9 组；每组执行当前 float-grid BFGS、相同 float-grid IPOPT、公开 Python literal IPOPT。
- 完成：`54,162` 组合帧任务、`162,486` 方法结果，IPOPT 失败 `0`。
- 公开 Python literal 语义包含一个容易遗漏的行为：`torch.nonzero()` 产生 `int64`，网格中心坐标写回时向零截断。实测 `torch112` 保持该行为，因此另设 float-grid IPOPT 只隔离求解器差异。

| 后处理 | FP32 avg. L2 / box Col | FP16 avg. L2 / box Col | INT8 avg. L2 / box Col |
| --- | ---: | ---: | ---: |
| 本地 BFGS，浮点坐标 | `0.776288 / 0.252022%` | `0.837544 / 0.459732%` | `0.784751 / 0.263100%` |
| IPOPT，同一浮点坐标 | `0.776184 / 0.254791%` | `0.837380 / 0.459732%` | `0.784621 / 0.265869%` |
| 公开 Python literal，整型坐标 + IPOPT | `0.827332 / 0.229866%` | `0.913302 / 0.437576%` | `0.838548 / 0.218788%` |

相同浮点坐标下，IPOPT 相对 BFGS 的 box Col 变化仅为 FP32 `+0.002769`、FP16 `0`、INT8 `+0.002769` 个百分点。BFGS 不是 FP16 Col 异常根因。公开 Python 整型坐标行为会明显改变绝对轨迹和指标，但 FP16 仍显著高于另外两档。

### 原始 Python IPOPT 交叉消融

box Col，行为 raw trajectory，列为 occupancy：

| Raw \ occupancy | FP32 | FP16 | INT8 |
| --- | ---: | ---: | ---: |
| FP32 | `0.229866%` | `0.434807%` | `0.216019%` |
| FP16 | `0.227096%` | `0.437576%` | `0.213249%` |
| INT8 | `0.238174%` | `0.432037%` | `0.218788%` |

固定 occupancy 后，raw precision 对 box Col 影响很小；固定 raw 后，FP16 occupancy 在三行都将 box Col 提高到 `0.432-0.438%`，而 FP32/INT8 occupancy 保持 `0.213-0.238%`。这直接证明异常来自 FP16 occupancy 在规划轨迹邻域的空间分布，不是 raw planning 或求解器。

全图 IoU 不能预测该结果：当前后处理只筛选每个 raw 轨迹点 5 m 内的正栅格。float-grid 候选点为 FP32 `464,970`、FP16 `636,074`、INT8 `513,075`；FP16 虽然全图正栅格少于 INT8，却在碰撞敏感走廊内产生更多候选，因此修改帧更多、Col 更差。

证据：

- 复现脚本：`UniAD-and-UniV2X-TRT/UniAD/repro/scripts/audit_collision_optimizer_parity.py`
- 紧凑结果：`UniAD-and-UniV2X-TRT/UniAD/evidence/trained_tiny_collision_optimizer_parity_full6018/summary.json`
- 完整逐帧结果：`/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/collision_optimizer_parity_20260814/full6018`

## 空目录说明

`/home/lixingfeng/UniAD-examine/UniAD-and-UniV2X-TRT` 与实际工作区 `/home/lixingfeng/UniAD_examine/UniAD-and-UniV2X-TRT` 不是同一路径。前者目录名使用了连字符，后者使用下划线。前者创建时间为 `2026-08-12T20:56:35-07:00`，目前只包含空的 `UniAD/evidence` 和 `UniV2X/evidence` 目录，没有 `.git`、普通文件或软链接。根据路径拼写、内容和时间戳，它是一次证据目录初始化时的路径拼写错误留下的空壳，不是第二个项目，也不参与当前部署。未获得明确删除指令前保留它。

---

## 2026-08-15T08:55:04-07:00 (PDT) - FP16 MatMul/Mul 全量因果消融

四组使用同一 ONNX、标准插件、TensorRT 10.9、`official_literal` 和 collision optimizer，仅改变严格 FP32 operator constraint：baseline、557 个 MatrixMultiply、269 个 FLOAT Mul、两者联合。269 个 FLOAT Mul 已覆盖 ONNX 中所有可 FP16 化的 Mul；其余 64 个为 INT32/INT64 shape/index Mul。

`901/901/1150` profile 在第 1225 帧产生 1152--1153 个 track，四组均按容量门禁退出，因此完整评估另用相同 `901/901/1600` profile 和 fixed 1600，不截断递归状态。

| Engine | occupancy IoU | raw L2 | planning MSE | optimized L2 | optimized Col | modified frames | model p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 baseline | `71.329463%` | `0.729232` | `0.0618751` | `0.837594` | `0.440346%` | `3425` | `11.247 ms` |
| MatMul-FP32 | `71.337650%` | `0.728676` | `0.0624877` | `0.835565` | `0.429268%` | `3417` | `13.324 ms` |
| Mul-FP32 | `71.257946%` | `0.728846` | `0.0623445` | `0.835578` | `0.437576%` | `3414` | `12.116 ms` |
| MatMul+Mul-FP32 | `71.305393%` | `0.728650` | `0.0624150` | `0.835803` | `0.426498%` | `3420` | `13.367 ms` |

完整 6018 帧证明：MatMul-only IoU 仅 `+0.00819` 点，Mul-only 和联合反而降低；最佳 Col 只改善 `0.01385` 点，仍显著高于 INT8 约 `0.2631%`，planning MSE 也未改善。因此未排除 MatMul/Mul 的 Half 计算不是 FP16 Col 异常主因。更符合证据的根因仍是共享 BEV 到 dense-future occupancy 路径的融合误差传播和未来时域放大；TensorRT 构建时同时给出了 LayerNorm Reduce/Pow FP16 溢出风险提示。

本轮四组并行运行，延迟是相对诊断口径；精度指标为同协议完整验证集结果。紧凑证据位于 `UniAD-and-UniV2X-TRT/UniAD/evidence/trained_tiny_fp16_matmul_mul_ab_full6018/summary_6018.json`，大文件位于 `/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/fp16_matmul_mul_ab_20260815_max1600`。

---

## 2026-08-15T23:28:46-07:00 (PDT) - seg_out 反向逐层定位

本轮使用同一 ONNX、插件和 TensorRT 10.9.0.34，构建 FP32/FP16 direct-output 探针，输入为同一批 40 帧记录 feed，关闭递归 temporal state 和 collision optimizer。runtime 改为保存动态输出的 `shapes_per_frame`，所以 `track_scores`、cross-attention 等输入相关 shape 分叉不会被错误地当成 engine 失败。

从 `seg_out` 反向得到的首个可观测连续误差边界在 tracking classification score 分支：

| Tensor | FP16 相对 FP32 relative-RMSE | 说明 |
| --- | ---: | --- |
| `onnx::Sigmoid_10611`（pre-Sigmoid logits） | `1.916788%` | 首个显著连续差异边界，最大绝对差 `5.026286` |
| `onnx::ReduceMax_10612`（Sigmoid 后） | `7.269288%` | 放大分数差异 |
| `scores.1` | `4.713701%` | 进入 active-index 条件 |

`scores.1` 的 `0.35/0.4` 条件在第 19、21、28 帧发生翻转，例如第 19 帧 `0.35: 0.3400408 -> 0.3544922`、`0.40: 0.4021551 -> 0.3833008`。其后 `track_scores` shape 在第 19/21/28 帧分别出现 `[2]->[1]`、`[3]->[2]`、`[2]->[1]`，再导致 cross-attention 输入长度和后续 `seg_out` 路径分叉。第 21 帧第一个可比的连续后续输出 `input.1791` RRMS 为 `58.016%`。

因此当前首个因果边界是“FP16 tracking score 数值差异 + 离散 active-track 筛选”，不是 occupancy 末端 `Mul`。occupancy gate `onnx::Less_23765` 的 0.3 mask 在 shape 对齐帧没有翻转；`future_states.3` 到 occupancy 末端的误差是后续放大或低能量张量的相对统计。tracking score 分支内部仍含 MatMul/Gemm，尚未把该分支逐个算子保护；后续修复应优先对 tracking classification score/active-index 路径做选择性 FP32 候选，并重新验证动态 shape、occupancy 和 Col，不应继续全局保护所有 MatMul/Mul。

复核脚本：`UniAD/repro/scripts/run_fp16_seg_reverse_layer_audit.sh`、`UniAD/repro/scripts/summarize_fp16_seg_reverse_layer_audit.py`。轻量证据：`UniAD/evidence/trained_tiny_fp16_seg_reverse_layer_audit_20260815/summary.json`。本轮为 40 帧归因实验，不覆盖已发布的正式 6018 帧 FP16 结果。
