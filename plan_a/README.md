# 既有圆环仿真的核对页

这一页只读 `圆环谐振器代理模型训练交付_20260929` 里已经写好的 CSV。不重新训练，不跑 COMSOL。

在仓库根目录运行：

```bash
python3 -m plan_a.render
```

写出 `plan_a/output/index.html`。浏览器直接打开，不需要网络。

品质因数和频率是交付包里那次仿真的结果，不是本队今年的实测。表里没有温度。两套权重的误差分别来自 `03_最终模型/deep_model_summary.csv` 和 `04_训练输出/final_mlp/deep_model_summary.csv`。

赛题对照、公式和资源说明在仓库根目录：`赛题解读与分析清单.md`、`数学公式.md`、`资源清单.md`。按赛题大纲写成的作品方案是 `技术方案.md`，PDF 是 `技术方案.pdf`。
