# 圆环谐振器最终 MLP 训练交付

本目录包含当前正式使用的三层 PyTorch MLP、对应训练数据、最终模型文件，以及生成新增训练标签所需的 COMSOL 控制和严格模态识别代码。

## 目录

- `01_训练数据/`：7 个训练 CSV。
- `02_训练代码/`：最终 MLP 的特征工程、训练和预测代码。
- `03_最终模型/`：已训练完成的最终 MLP、配套预处理器和指标。
- `04_训练输出/`：重新训练及预测的输出位置。
- `05_仿真与数据追加/`：COMSOL 控制、位移向量导出、严格模态识别和有效数据追加。

## 环境

推荐 Python 3.11/3.12。安装依赖：

```powershell
python -m pip install -r requirements.txt
```

## 直接训练

双击根目录的 `START_TRAINING.bat`（推荐）或 `训练最终模型.cmd`。启动后窗口会立即显示 Python 检查、数据读取和每 10 个 epoch 的训练进度。

也可以运行：

```powershell
python train.py
```

默认设置：

- 数据：`01_训练数据/training_q_*.csv`
- 有效性：`valid_for_Q=1/true`
- Q 过滤：`selected_Q >= 1e6`
- 样本数：19,320
- 输入：半径、平均环宽、1–8 阶 Fourier 系数及派生几何特征
- 输出：`log10(Q)`、`selected_freq_MHz`
- 网络：`59 -> 256 -> 128 -> 64 -> 2`
- 最大 epoch：350，带学习率调度和早停
- 随机种子：20260626

训练结果固定写入：

`04_训练输出/final_mlp/`

训练过程同时记录在 `04_训练输出/final_mlp/training.log`。如果窗口没有显示或意外关闭，可直接查看该日志。

主要结果文件：

- `deep_model_summary.csv`：测试集指标。
- `torch_mlp.pt`：模型权重。
- `preprocessing.joblib`：特征顺序、缺失值填补和标准化器。
- `metadata.json`：数据与训练配置。
- `torch_mlp_history.csv`：训练/验证损失。
- `test_predictions.csv`：测试集真实值和预测值。

## 使用已有最终模型预测

双击 `运行预测示例.cmd`，或运行：

```powershell
python predict.py --input example_input.csv
```

预测输出固定写入 `04_训练输出/predictions.csv`。输入 CSV 至少包含 `a_ring_um`、`h_ring_um` 以及实际使用的 `Ck_um/Sk_um`；缺少的 Fourier 系数自动补零。

## 已交付模型指标

最终模型使用 19,320 个样本，训练/验证/测试划分为 12,673/2,783/3,864。测试集指标：

- `logQ_MAE = 0.0351545`
- `logQ_R2 = 0.9784610`
- `Q_p90_rel_err = 0.180473`
- `freq_MAE_MHz = 0.00453164`

重新训练结果允许因 PyTorch、CPU 和线程调度不同产生小幅差异。

## 添加新的仿真数据

1. 将新 COMSOL 模型放入 `05_仿真与数据追加/models/`，按六阶或八阶的既定文件名替换。
2. 在 `05_仿真与数据追加/config/project_config.json` 检查 COMSOL 可执行文件地址。
3. 编辑 `input/designs_width6_example.csv`，再双击 `RUN_NEW_DATA.bat`。
4. 程序会完成仿真、位移向量模态识别，并把有效数据追加至 `01_训练数据/training_q_user_added.csv`。

详细文件名、模态判据和输出位置见 `05_仿真与数据追加/README.md`。追加后双击 `START_TRAINING.bat` 重新训练。

## 自动寻找新设计点

- 双击 `GENERATE_NEW_DESIGNS.bat`：只寻找下一批设计，不运行 COMSOL。结果写入 `05_仿真与数据追加/input/designs_width6_auto.csv`。
- 检查生成的 CSV 后双击 `RUN_GENERATED_DATA.bat`：仿真这批已检查的点、追加有效数据并重训模型，不会重新换一批点。
- 双击 `AUTO_FIND_AND_RUN.bat`：完成一次“找点 → COMSOL → 模态识别 → 追加有效数据 → 重训模型”闭环。

默认每批选择 6 点：4 个用于补充设计空间覆盖，2 个为代理模型预测的高 Q 点。候选必须满足环宽 `6.5–13.5 μm`，并与已有训练点保持距离。设置位于 `05_仿真与数据追加/config/design_search_config.json`。

程序优先使用 `04_训练输出/final_mlp/` 中最近重训的模型；没有重训模型时使用 `03_最终模型/`。搜索详情写入 `04_训练输出/candidate_search/`。
