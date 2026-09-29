from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "05_仿真与数据追加" / "config" / "design_search_config.json"


def run(command: list[str], title: str) -> None:
    print(f"\n=== {title} ===", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="执行一次自动找点、COMSOL 验证、数据追加和模型重训闭环")
    parser.add_argument("--skip-training", action="store_true", help="完成仿真和数据追加后不重新训练 MLP")
    parser.add_argument("--compile", action="store_true", help="运行 COMSOL 前重新编译 Java runner")
    parser.add_argument("--use-existing", action="store_true", help="不重新找点，直接运行已生成并检查过的 auto CSV")
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    order = int(config["order"])
    design_file = ROOT / "05_仿真与数据追加" / "input" / f"designs_width{order}_auto.csv"
    label = f"auto_{datetime.now():%Y%m%d_%H%M%S}"

    if args.use_existing:
        if not design_file.exists():
            raise FileNotFoundError(f"尚未生成设计文件：{design_file}。请先运行 GENERATE_NEW_DESIGNS.bat。")
        print(f"\n使用已检查的设计文件：{design_file}", flush=True)
    else:
        run([sys.executable, str(ROOT / "generate_new_designs.py")], "1/3 自动寻找下一批设计")
    simulation = [
        sys.executable,
        str(ROOT / "05_仿真与数据追加" / "python" / "simulate_and_score.py"),
        "--order", str(order),
        "--input", str(design_file),
        "--label", label,
        "--append",
    ]
    if args.compile:
        simulation.append("--compile")
    run(simulation, "COMSOL 仿真、模态识别和有效数据追加")

    if args.skip_training:
        print("\n已跳过重训。需要更新代理模型时运行 START_TRAINING.bat。")
    else:
        run([sys.executable, str(ROOT / "train.py")], "使用新增有效数据重新训练 MLP")
    print(f"\n本轮输出：{ROOT / '05_仿真与数据追加' / 'output' / label}")


if __name__ == "__main__":
    main()
