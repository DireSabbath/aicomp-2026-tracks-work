# 仿真、模态识别与新增数据

## 自动寻找新点

返回交付包根目录：

- `GENERATE_NEW_DESIGNS.bat`：根据现有训练数据和当前 MLP 生成 6 个新设计点，不运行 COMSOL。
- `RUN_GENERATED_DATA.bat`：运行已经生成并人工检查过的 auto CSV，不会重新生成另一批点。
- `AUTO_FIND_AND_RUN.bat`：自动找点、运行 COMSOL、完成模态识别、追加有效数据并重训 MLP。

默认结果为 `input/designs_width6_auto.csv`，策略是 4 个覆盖点加 2 个预测高 Q 点。点数、阶数、候选池大小和覆盖比例在 `config/design_search_config.json` 中修改。

## 1. 放入 COMSOL 模型

- 六阶模型：`models/plain_width6_model_manual10.mph`
- 八阶模型：`models/plain_width8_model_manual10.mph`

新模型放入 `models/`，并改成上面的对应文件名。若要保留其他文件名，请修改 `python/simulate_and_score.py` 中的 `runner_paths()`。

模型默认使用研究标签 `std3`，请求 10 个特征频率；模态识别会检查实际存在的解序号 1–20。

## 2. 检查 COMSOL 地址

打开 `config/project_config.json`，确认：

- `batch_exe`：`comsolbatch.exe` 的完整地址，用于运行仿真。
- `compile_exe`：`comsolcompile.exe` 的完整地址，仅重新编译 Java 或更换 COMSOL 版本时使用。

当前配置示例：`D:\comsol\COMSOL63\Multiphysics\bin\win64\`。

## 3. 运行新数据

在 `input/designs_width6_example.csv` 中每行填写一个设计，然后双击 `RUN_NEW_DATA.bat`。六阶输入需要 `a_ring_um,h_ring_um,C1_um,S1_um,...,C6_um,S6_um`。

命令行运行方式：

```powershell
python python/simulate_and_score.py --order 6 --input input/designs_width6_example.csv --label batch_001 --append
```

八阶数据把 `--order 6` 改为 `--order 8`，输入系数增加至 `C8_um,S8_um`。

自动生成的设计已写入 `input/designs_width6_auto.csv`。若只想仿真这批点，可运行：

```powershell
python python/simulate_and_score.py --order 6 --input input/designs_width6_auto.csv --label auto_manual --append
```

## 4. 模态识别

程序对每个可用模态导出 64 个采样点的三维位移向量，并与 `reference/strict_mac_initial_vectors.csv` 中的参考模态计算严格 MAC。有效条件为：

- strict MAC ≥ 0.90
- 径向位移同号比例 ≥ 0.95
- 呼吸均匀性 ≥ 0.95

频率只作为训练目标，不用于删除样本。结果位于 `output/<label>/`：

- `mode_scores.csv`：所有模态的识别分数。
- `summary.csv`：每个设计的最佳模态。
- `training_valid.csv`：通过判据的有效数据。

使用 `--append` 时，有效数据自动追加到 `../01_训练数据/training_q_user_added.csv`。完成后双击根目录 `START_TRAINING.bat` 重新训练。
