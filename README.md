# 氧化机制表征学习与动力学分析软件 V1.0

增强功能详见 [V1.0 增强功能使用说明](docs/增强功能使用说明_V1.0.md)。

安装完整功能：`python -m pip install -r requirements-full.txt`，启动：`python server.py`。

本地完整版本，包含六通道 UMAP、数值聚类与簇解释、预测样本定位、候选对空间与 GLM 差异比较、Optuna TPE 优化，以及数据输入、固定划分、参考编码、机制预训练、冻结迁移、预测检索、候选筛选及 GLM 机制分析。所有程序文件使用功能名称，不依赖 Fig.6 / Fig.7 等图号脚本，也不需要叠加旧版补丁。

## 1. 安装与启动

建议 Python 3.10–3.12。在解压后的软件根目录打开终端：

```bash
python -m pip install -r requirements-full.txt
python server.py
```

浏览器打开 `http://127.0.0.1:8765`。Windows 安装依赖后可双击 `start_windows.bat`；macOS/Linux 可执行 `bash start.sh`。端口冲突时使用 `python server.py --port 8766`，浏览器访问对应端口。

- `requirements.txt`：界面、数据处理、PCA、NumPy 神经网络训练及 NPZ 模型推理。
- `requirements-model.txt`：额外提供 PyTorch、Transformers，用于 SciBERT 编码和旧 `.pt` 模型推理。
- `requirements-llm.txt`：GLM 机制文字分析依赖。
- `requirements-analysis.txt`：UMAP 和图表导出。
- `requirements-optimization.txt`：Optuna TPE 优化。
- `requirements-clustering.txt`：可选 HDBSCAN，默认 KMeans 无需此文件。
- `requirements-full.txt`：汇总除可选 HDBSCAN 外的上述依赖。

首次安装依赖需要联网。安装完成后，本地数值计算无需联网；调用 GLM 时需要连接智谱服务。直接双击 `web/index.html` 无法使用软件，应通过 Python 服务打开页面。

本版所有文件独立完整，建议解压到新目录运行。关闭之前的 Python 服务，避免浏览器连到旧版本。页面左下角应显示 **V1.0.0**；必要时按 Ctrl+F5。

## 2. 使用原来的模型与配置

已有模型可以继续使用，不需要为升级重新训练。在新软件根目录执行：

```bash
python tools/import_previous_workspace.py "D:/你的旧软件目录"
```

该命令把旧 `config/workspace.json` 中的数据、模型和参考文件位置转为绝对路径，保留原文件；并导入非敏感 LLM 配置，必要时将 .env 路径指向旧目录。旧数据和旧权重不会被修改。然后启动或重启新软件服务。

也可以在「模型与路径配置」页手动填写旧文件的绝对路径。旧 PyTorch 模型需要 `trial_summary.json` 或准确的预训练架构、`transfer_config.json`、`best_model_state.pt`、`scaler.json` 及对应的参考 CSV/划分。

如果已有标准化器仅为自己训练得到的可信 `shared_scaler.joblib`，可转换一次：

```bash
python tools/scaler_converter.py D:/旧模型/shared_scaler.joblib D:/旧模型/scaler.json --trust-local-file
```

路径含空格时加双引号。真实模型文件名请按上述名称保留。

## 3. 从 JSON 和 Excel 运行完整训练

在「训练任务」页可以选择文件，或直接输入本机文件路径。文件选择会把文件保存至本机软件目录，不向外部服务上传。若相同目标文件已存在，旧文件备份至 `outputs/uploads/`。

自动模式默认位置：

| 内容 | 默认位置 |
| --- | --- |
| 联合抽取结果 | `data/private/joint_six_channel.json` |
| 材料与动力学数据 | `data/private/oxidation_dataset.xlsx` |
| 本地 SciBERT 模型与 tokenizer | `models/scibert/` |

**JSON 和 Excel 是研究输入；新建参考文本向量还需要本地 SciBERT 文件。** SciBERT 不包含在软件包内，不能仅凭抽取 JSON 生成其权重。如果你已有原始文本向量，也可在页面指定原始向量目录，跳过 SciBERT 编码。

操作步骤：

1. 选择「总数据自动划分」，上传或填写 JSON 与总数据 Excel。
2. 填写本地 SciBERT 目录，保存配置。
3. 点击「一键运行全部阶段」。
4. 查看任务阶段、实际日志和数据数量。
5. 全部训练成功后，程序自动激活本次模型；转到「联合预测与检索」，选择「本地研究模型」开始分析。

也支持分阶段执行「准备与固定划分」→「参考表示＋机制预训练」→「动力学迁移与评估」。准备后的运行沿用保存的材料文件和划分，不因界面后续改变数据路径而重新分割；修改输入数据后应重新准备。SciBERT 或原始向量目录可以在预训练前更新。

## 4. 自动与手动划分

### 自动模式

默认按文献来源 `Ref_source_num` 分组，以固定种子 42 划分为约 60% 训练、20% 验证、20% 测试。一个来源的所有记录进入同一集合。来源大小不同，记录比例可能与 60/20/20 略有差异；页面显示实际数量。分组分配仅依据组别、数量与种子，不读取动力学标签挑选划分。

也可选择按记录随机划分；若来源跨集合，数据报告会标记来源重叠。分组策略至少需要三个来源。软件保存一份完整四元标识 `split.csv`，此后预训练、迁移、检索和解释都依据这份划分。

### 手动模式

将文件放在下列位置，或填写其他本机路径：

| 文件 | 默认位置 |
| --- | --- |
| 训练集 | `data/manual/train.xlsx` |
| 验证集（可选） | `data/manual/val.xlsx` |
| 测试集 | `data/manual/test.xlsx` |

验证文件存在时采用 train/val/test；不存在或路径留空时采用 train/test。手动文件既可分别包含 Sheet3/Sheet1，也可只有一张完整材料表。联合 JSON 仍用于为这些材料记录提供机制监督。

软件不会重分手动数据；检查跨集合重复四元标识及 sample_id。文献来源跨集合时提示并记录，但保留你的手动划分。若不希望使用旧验证文件，应将验证路径留空，避免软件按仍存在的文件启用验证。

## 5. 数据格式

Excel 优先读取：

- `Sheet3`：材料和氧化条件特征。
- `Sheet1`：材料和氧化条件特征，以及 `lgkp` 动力学标签。
- 没有这些名称时，读取第一张工作表；手动单表文件采用这一规则。

两个表中相同四元标识的特征必须一致；有效动力学标签会合并。没有动力学标签的记录仍可用于机制学习。迁移训练需要训练和测试集合有有效 lgkp；若设置了验证集，也需要有效验证 lgkp。

四元标识为：

```text
Ref_source_num, sample_id, alloy_id, condition_id
```

Excel 缺少 condition_id 时，只有在 JSON 的其余三个标识能够唯一匹配条件时才补充；多条件歧义会明确报错。

29 个特征为：

```text
solu_temp, solu_time, middle_temp, middle_time, aging_temp, aging_time,
Co, Al, W, Ni, Cr, Mo, Fe, Nb, C, Hf, Si, Ta, Ti, Y, V, B, Zr, Mn, Sc, La, Re,
Temperature, E_Time
```

成分用 at.% 且总和为 100±0.2；`x_Co` 等元素别名也可识别。温度为 °C，时间为 h。未添加元素、未执行步骤要显式写 0。特征缺失或无效会报错，不自动把未知信息填成零。预测 CSV 模板在「数据管理」页可下载。

联合 JSON 支持对象、对象列表、`target_samples` 列表或平铺样本。每个样本的 `channels` 使用六个通道名称，包含 `tags`、`text`、`status`、`evidence_basis`。未报告或空文本记录不进入机制损失及有效检索参考。

### 已有原始文本向量（可选）

页面「已有原始文本向量目录」下需要 `ch1_raw.csv` 至 `ch6_raw.csv`，各文件具有完整四元标识和连续的 `raw_0, raw_1, ...` 列。它们应为同一固定文本编码器的原始输出，不可把已拟合 PCA 的坐标当作原始 SciBERT 向量使用。填入该目录时跳过文本模型编码，但仍在当前训练参考上重新拟合 PCA。

## 6. 固定训练参数与模型实现

| 项目 | 默认值 |
| --- | --- |
| 输入特征 | 29 |
| 共享隐藏层 | 128 / 128 / 64 |
| 机制通道 | 六个独立线性头 |
| 请求机制维度 | 64 |
| Dropout | 0.1 |
| Batch size | 32 |
| 预训练 | 300 轮，Adam，学习率 0.001 |
| 动力学头 | 64 / 32 / 1 |
| 动力学训练 | 1000 轮，Adam，学习率 0.001 |
| Weight decay | 0.00001 |
| 有验证集时早停耐心 | 50 |
| 动力学迁移 | 冻结编码器、BatchNorm 统计及六通道头 |

本版训练模块使用 NumPy 实现共享 MLP、BatchNorm、ReLU、Dropout、独立机制头和 Adam；这是实际神经网络数值训练，不是解析演示映射。机制损失为各有效通道平均的余弦损失，缺失机制逐通道屏蔽；动力学损失为 MSE。

架构延续已有研究设计，但随机初始化、数据划分与训练集 PCA 的实现不同，**重新训练不会自动复现论文中已定模型的指标或权重**。已有 PyTorch 权重通过兼容预测引擎加载。

请求 64 维时，如果最少有效训练参考数不足，所有通道采用共同可拟合维度：不超过 64、最少有效训练数减 1、原始文本维度。参考报告记录实际维度。每个通道至少三条有效训练机制。

数值标准化器只拟合训练材料；PCA 只拟合各通道有效训练机制。固定 SciBERT 可以编码验证和测试文本，但不在这些文本上更新模型或拟合 PCA。

有验证集：预训练按验证余弦指标选模型；动力学按验证 RMSE 选模型；测试结果在选定权重之后计算。仅 train/test：按固定轮数保存最后模型，不依据测试集选轮数或早停。

页面提供路径配置和固定参数说明；高级参数存于 `config/training.json`，固定训练采用以上参数；可在训练页额外执行 Optuna TPE，具体搜索配置和验证集要求见增强功能说明。

## 7. 训练产物与自动激活

每次准备建立独立目录 `outputs/training_时间_编号/`：

| 子目录 | 内容 |
| --- | --- |
| `dataset/` | train/val/test.xlsx、master.csv、split.csv、数据报告 |
| `references/` | 六通道 CSV、原始向量缓存、训练 PCA、参考报告 |
| `pretrained/` | best_pretrain_model.npz、pretrain_config.json、scaler.json、划分、曲线及指标 |
| `kinetics/` | best_model_state.npz、transfer_config.json、曲线、预测及分集合指标 |

本版新模型使用 `.npz` 权重，加载采用 `allow_pickle=False`。完成动力学训练并检查必要产物后，更新 `config/workspace.json`，无需手动移动文件或转换标准化器。上一份预测配置备份为该运行的 `previous_workspace.json`。

任务状态和实际日志保存在 `outputs/jobs/`，重启后历史记录仍可查看。服务关闭时仍在运行的旧任务会显示状态未知，应先确认旧进程情况再开始新任务；本版未提供训练暂停或恢复功能。

## 8. 预测、Top-5 与机制分析

在联合预测页选择「本地研究模型」后运行预测，可直接点击「进入 LLM 机制分析」。当前预测的 run_id 和样本列表直接传给分析页，不依赖历史扫描成功才能显示样本。历史读取错误会显示出来，不静默跳过。

输出下载路径统一使用 `/`，支持 Windows。网页和接口禁用缓存，并为 JS/CSS 添加 V1.0 版本标识。若仍无样本，检查：

1. 页面是否为 V1.0.0、是否启用了 Top-5。
2. 新版 Python 服务是否在当前浏览器端口运行。
3. 页面中的读取错误和 `outputs/run_.../result.json` 是否包含 retrieval。
4. 当前预测的编号是否与分析页选中的编号一致。

参考范围可选训练或训练＋验证；按照真正的三划分标识排除测试。若旧共享划分只有 train/val 而没有 test，阻止把它直接视为训练＋验证，以免将旧迁移测试记录作为验证参考。同源排除按 `Ref_source_num` 执行，自身四元标识也被排除。

GLM 配置沿用：

```python
ChatZhipuAI(temperature=0, model="glm-4-plus", api_key=api_key)
```

把 `.env.example` 复制为 `.env` 并填写 `ZHIPUAI_API_KEY`，或沿用已有 `API_KEY` / `ZHIPU_API_KEY`。网页只显示密钥是否检测到，不显示值。环境变量优先，修改密钥后重启服务。

机制分析先生成六个带参考编号的通道解释，再根据经引用结构检查的解释生成整体总结。展示参考共识、分歧及未明确支持的细节，支持点击引用追溯原参考。每个样本正常调用两次模型；仅点击生成时发送所选样本及 Top-5，不发送全部原始数据或权重。

引用检查不是科学事实审核，解释仍需人工核对。分析报告、JSON 和提示词记录保存到原预测运行下。服务重启后分析任务的内存状态不保留，已导出的文件保留。合成演示结果仍有明确标识。

## 9. 候选筛选与演示模式

候选筛选使用动力学容差和参考距离百分位校准的平衡机制分数，沿用已有候选筛选逻辑，距离校准只使用训练参考。本版没有接入原论文全部输入支持度及附加研究筛选约束。

默认合成演示可在没有数据和权重时展示预测、Top-5、候选配对等页面。它使用合成解析输出，与实际训练模型区分，不能用于研究结论。训练任务运行的是实际输入文件和神经网络计算，不模拟训练曲线。

## 10. 检查与限制

```bash
python tools/inspect_environment.py
python -m unittest discover -s tests -v
```

若安装了 Node.js，可额外执行实际分析页脚本的样本衔接检查：

```bash
node tests/test_analysis_handoff.cjs
```

本次已用合成数值夹具实际完成自动及手动划分、训练集 PCA、六通道预训练、冻结迁移、模型激活、真实权重预测、Top-5 和分析样本预览；检查了梯度与有限差分、测试标签不影响选定权重，以及两集合固定轮数规则。

SciBERT 真模型编码与 GLM 真实服务未在本次环境运行：本次没有实际本地 SciBERT 权重、PyTorch 或 API 密钥，训练闭环使用预计算数值夹具，GLM 使用响应替身。本增强版的数值与界面检查范围见 docs/验证报告.md。软件不包含 EXE 安装器、用户研究数据、论文权重或文献全文。
