# 输出位置

执行根目录的 `训练最终模型.cmd` 后，全部训练产物写入：

`04_训练输出/final_mlp/`

其中包括模型权重 `torch_mlp.pt`、预处理器 `preprocessing.joblib`、配置 `metadata.json`、指标 `deep_model_summary.csv`、训练曲线 `torch_mlp_history.csv` 和测试集预测 `test_predictions.csv`。

执行 `运行预测示例.cmd` 后，结果写入：

`04_训练输出/predictions.csv`
