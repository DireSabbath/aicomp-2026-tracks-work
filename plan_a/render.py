"""把既有圆环交付包里的 CSV 画成一份离线 HTML。不重新训练，也不跑仿真。"""

from __future__ import annotations

import csv
import html
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "圆环谐振器代理模型训练交付_20260929"
DATA_DIR = PACKAGE / "01_训练数据"
Q_MIN = 1_000_000.0
GEO_FIELDS = ["a_ring_um", "h_ring_um"] + [
    f"{prefix}{order}_um" for order in range(1, 9) for prefix in ("C", "S")
]


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _f(row: dict, key: str, default: float = 0.0) -> float:
    raw = row.get(key, "")
    if raw is None or raw == "":
        return default
    return float(raw)


def _hist(values: list[float], start: float, stop: float, step: float) -> list[tuple[float, int]]:
    count = int(round((stop - start) / step))
    bins = [0] * count
    for value in values:
        index = int((value - start) / step)
        if index < 0 or index >= count:
            continue
        bins[index] += 1
    return [(start + (index + 0.5) * step, bins[index]) for index in range(count)]


def _grid(xs: list[float], ys: list[float], x0: float, x1: float, y0: float, y1: float, n: int) -> list[list[int]]:
    grid = [[0 for _ in range(n)] for _ in range(n)]
    if x1 == x0 or y1 == y0:
        return grid
    for x_value, y_value in zip(xs, ys):
        col = int((x_value - x0) / (x1 - x0) * n)
        row = int((y_value - y0) / (y1 - y0) * n)
        if 0 <= col < n and 0 <= row < n:
            grid[row][col] += 1
    return grid


def _read_summary(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return next(csv.DictReader(handle))


def _read_history(path: Path) -> list[tuple[float, float]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [(float(row["train_loss"]), float(row["val_loss"])) for row in rows]


def scan_package(package: Path | None = None) -> dict:
    package = package or PACKAGE
    data_dir = package / "01_训练数据"
    files = []
    kept_a: list[float] = []
    kept_h: list[float] = []
    kept_f: list[float] = []
    all_f: list[float] = []
    kept_logq: list[float] = []
    sources_all: Counter[str] = Counter()
    sources_keep: Counter[str] = Counter()
    seen: set[tuple[str, ...]] = set()
    duplicate_rows = 0
    q_min = math.inf
    q_max = 0.0
    for path in sorted(data_dir.glob("training_q_*.csv")):
        total = 0
        kept = 0
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                total += 1
                key = tuple(f"{_f(row, name):.6f}" for name in GEO_FIELDS)
                if key in seen:
                    duplicate_rows += 1
                else:
                    seen.add(key)
                q_value = _f(row, "selected_Q")
                freq = _f(row, "selected_freq_MHz")
                all_f.append(freq)
                q_min = min(q_min, q_value)
                q_max = max(q_max, q_value)
                kind = row.get("source_kind") or "未标"
                sources_all[kind] += 1
                if q_value < Q_MIN:
                    continue
                kept += 1
                kept_a.append(_f(row, "a_ring_um"))
                kept_h.append(_f(row, "h_ring_um"))
                kept_f.append(freq)
                kept_logq.append(math.log10(q_value))
                sources_keep[kind] += 1
        files.append({"name": path.name, "total": total, "kept": kept, "dropped": total - kept})
    files.sort(key=lambda item: (-item["kept"], item["name"]))

    pred_true_logq: list[tuple[float, float]] = []
    pred_true_freq: list[tuple[float, float]] = []
    pred_path = package / "04_训练输出" / "final_mlp" / "test_predictions.csv"
    with pred_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            pred_true_logq.append((_f(row, "log10_Q"), _f(row, "pred_logQ")))
            pred_true_freq.append((_f(row, "selected_freq_MHz"), _f(row, "pred_freq_MHz")))

    default_summary = _read_summary(package / "03_最终模型" / "deep_model_summary.csv")
    retrain_summary = _read_summary(package / "04_训练输出" / "final_mlp" / "deep_model_summary.csv")
    return {
        "files": files,
        "n_rows": sum(item["total"] for item in files),
        "n_keep": sum(item["kept"] for item in files),
        "n_drop": sum(item["dropped"] for item in files),
        "duplicate_rows": duplicate_rows,
        "unique_geometries": len(seen),
        "a": kept_a,
        "h": kept_h,
        "freq": kept_f,
        "logq": kept_logq,
        "sources": sources_all.most_common(),
        "sources_keep": sources_keep.most_common(),
        "freq_all": all_f,
        "q_min": q_min,
        "q_max": q_max,
        "pred_logq": pred_true_logq,
        "pred_freq": pred_true_freq,
        "default_summary": default_summary,
        "retrain_summary": retrain_summary,
        "default_history": _read_history(package / "03_最终模型" / "torch_mlp_history.csv"),
        "retrain_history": _read_history(package / "04_训练输出" / "final_mlp" / "torch_mlp_history.csv"),
    }


def _polyline(
    points: list[tuple[float, float]],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    if not points or x1 == x0 or y1 == y0:
        return ""
    coords = []
    for x_value, y_value in points:
        x_pos = left + (x_value - x0) / (x1 - x0) * width
        y_pos = top + (y1 - y_value) / (y1 - y0) * height
        coords.append(f"{x_pos:.1f},{y_pos:.1f}")
    return " ".join(coords)


def _chart(
    series: list[dict],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    title: str,
    x_label: str,
    y_label: str,
) -> str:
    width, height = 920, 360
    left, right, top, bottom = 72, 24, 18, 40
    plot_w = width - left - right
    plot_h = height - top - bottom
    parts = [
        f"<svg viewBox='0 0 {width} {height}' class='chart' role='img'>",
        f"<title>{_esc(title)}</title>",
        f"<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' class='frame'></rect>",
    ]
    for fraction in (0, 0.5, 1):
        y_value = y0 + (y1 - y0) * fraction
        y_pos = top + (1 - fraction) * plot_h
        x_value = x0 + (x1 - x0) * fraction
        x_pos = left + fraction * plot_w
        parts.append(
            f"<line x1='{left}' y1='{y_pos:.1f}' x2='{left + plot_w}' y2='{y_pos:.1f}' class='grid'></line>"
            f"<text x='{left - 8}' y='{y_pos + 4:.1f}' class='tick' text-anchor='end'>{y_value:.3g}</text>"
            f"<text x='{x_pos:.1f}' y='{top + plot_h + 16}' class='tick' text-anchor='middle'>{x_value:.3g}</text>"
        )
    for item in series:
        coords = _polyline(item["points"], x0, x1, y0, y1, left, top, plot_w, plot_h)
        if not coords:
            continue
        dash = f" stroke-dasharray='{item['dash']}'" if item.get("dash") else ""
        parts.append(
            f"<polyline data-series='{_esc(item['name'])}' points='{coords}' fill='none' stroke='{item['color']}' stroke-width='2'{dash}></polyline>"
        )
    parts.append(
        f"<text x='{left + plot_w / 2:.0f}' y='{height - 4}' class='tick' text-anchor='middle'>{_esc(x_label)}</text>"
        f"<text x='16' y='{top + plot_h / 2:.0f}' class='tick' transform='rotate(-90 16 {top + plot_h / 2:.0f})' text-anchor='middle'>{_esc(y_label)}</text>"
        "</svg>"
    )
    return "".join(parts)


def _bars(rows: list[tuple[str, float, str]], title: str, suffix: str = "") -> str:
    peak = max((abs(value) for _, value, _ in rows), default=1) or 1
    width, label_w, value_w = 920, 280, 90
    height = 8 + len(rows) * 28
    span = width - label_w - value_w
    parts = [f"<svg viewBox='0 0 {width} {height}' class='chart' role='img'><title>{_esc(title)}</title>"]
    for index, (label, value, color) in enumerate(rows):
        y_pos = 8 + index * 28
        bar = span * abs(value) / peak
        shown = f"{value:.0f}{suffix}" if float(value).is_integer() else f"{value:.4g}{suffix}"
        parts.append(
            f"<text x='0' y='{y_pos + 16}' class='lab'>{_esc(label)}</text>"
            f"<rect x='{label_w}' y='{y_pos + 4}' width='{bar:.1f}' height='14' fill='{color}'></rect>"
            f"<text x='{label_w + bar + 8:.1f}' y='{y_pos + 16}' class='val'>{_esc(shown)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _heat(
    grid: list[list[int]],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    title: str,
    x_label: str,
    y_label: str,
    diagonal: bool = False,
) -> str:
    width, height = 920, 420
    left, right, top, bottom = 72, 24, 18, 40
    plot_w = width - left - right
    plot_h = height - top - bottom
    rows = len(grid)
    cols = len(grid[0]) if grid else 1
    peak = max((cell for row in grid for cell in row), default=1) or 1
    parts = [
        f"<svg viewBox='0 0 {width} {height}' class='chart' role='img'>",
        f"<title>{_esc(title)}</title>",
        f"<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' class='frame'></rect>",
    ]
    cell_w = plot_w / cols
    cell_h = plot_h / rows
    for row_index, row in enumerate(grid):
        for col_index, cell in enumerate(row):
            if cell <= 0:
                continue
            shade = 0.12 + 0.88 * math.sqrt(cell / peak)
            red = round(243 - shade * (243 - 36))
            green = round(239 - shade * (239 - 87))
            blue = round(230 - shade * (230 - 166))
            x_pos = left + col_index * cell_w
            y_pos = top + (rows - 1 - row_index) * cell_h
            parts.append(
                f"<rect x='{x_pos:.2f}' y='{y_pos:.2f}' width='{cell_w:.2f}' height='{cell_h:.2f}' fill='rgb({red},{green},{blue})'></rect>"
            )
    if diagonal:
        coords = _polyline([(x0, y0), (x1, y1)], x0, x1, y0, y1, left, top, plot_w, plot_h)
        parts.append(f"<polyline points='{coords}' fill='none' stroke='#9c2b1e' stroke-width='1.4'></polyline>")
    for fraction, value in ((0, x0), (1, x1)):
        x_pos = left + fraction * plot_w
        parts.append(f"<text x='{x_pos:.1f}' y='{top + plot_h + 16}' class='tick' text-anchor='middle'>{value:.3g}</text>")
    for fraction, value in ((0, y0), (1, y1)):
        y_pos = top + (1 - fraction) * plot_h
        parts.append(f"<text x='{left - 8}' y='{y_pos + 4:.1f}' class='tick' text-anchor='end'>{value:.3g}</text>")
    parts.append(
        f"<text x='{left + plot_w / 2:.0f}' y='{height - 4}' class='tick' text-anchor='middle'>{_esc(x_label)}</text>"
        f"<text x='16' y='{top + plot_h / 2:.0f}' class='tick' transform='rotate(-90 16 {top + plot_h / 2:.0f})' text-anchor='middle'>{_esc(y_label)}</text>"
        "</svg>"
    )
    return "".join(parts)


def _legend(items: list[tuple[str, str, str]]) -> str:
    buttons = [
        f"<button type='button' class='legend' data-series='{_esc(name)}' style='--c:{color}'>{_esc(label)}</button>"
        for name, label, color in items
    ]
    return "<div class='legend-row'>" + "".join(buttons) + "</div>"


def _table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{_esc(cell)}</th>" for cell in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<div class='table-wrap'><table class='plain'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _metric(summary: dict, key: str) -> float:
    return float(summary[key])


def build_page(package: Path | None = None) -> str:
    data = scan_package(package)
    default = data["default_summary"]
    retrain = data["retrain_summary"]
    file_rows = [
        [item["name"], str(item["total"]), str(item["kept"]), str(item["dropped"])] for item in data["files"]
    ]
    file_rows.append(["合计", str(data["n_rows"]), str(data["n_keep"]), str(data["n_drop"])])
    metric_rows = []
    for label, summary in (("03_最终模型", default), ("04_训练输出/final_mlp", retrain)):
        metric_rows.append(
            [
                label,
                summary["epochs_run"],
                f"{_metric(summary, 'logQ_MAE'):.5f}",
                f"{_metric(summary, 'logQ_R2'):.5f}",
                f"{100 * _metric(summary, 'Q_mean_rel_err'):.2f}%",
                f"{100 * _metric(summary, 'Q_p90_rel_err'):.2f}%",
                f"{_metric(summary, 'freq_MAE_MHz'):.5f}",
                f"{_metric(summary, 'freq_R2'):.5f}",
            ]
        )
    a_hist = _hist(data["a"], 122.0, 130.5, 0.5)
    h_hist = _hist(data["h"], 8.0, 13.75, 0.25)
    f_hist = _hist(data["freq_all"], 8.0, 10.25, 0.1)
    q_hist = _hist(data["logq"], 6.0, 8.0, 0.1)
    geo_grid = _grid(data["a"], data["h"], 122.0, 130.0, 8.0, 13.5, 18)
    logq_true = [item[0] for item in data["pred_logq"]]
    logq_pred = [item[1] for item in data["pred_logq"]]
    freq_true = [item[0] for item in data["pred_freq"]]
    freq_pred = [item[1] for item in data["pred_freq"]]
    logq_grid = _grid(logq_true, logq_pred, 6.0, 8.0, 6.0, 8.0, 28)
    freq_grid = _grid(freq_true, freq_pred, 8.0, 10.2, 8.0, 10.2, 28)
    top_sources = data["sources"][:8]
    other = sum(count for _, count in data["sources"][8:])
    keep_sources = dict(data["sources_keep"])
    source_rows = [(name, float(count), "#2457a6") for name, count in top_sources]
    if other:
        source_rows.append(("其余标签", float(other), "#8d8678"))
    guided = next(item["kept"] for item in data["files"] if item["name"].endswith("width6_guided_coverage.csv"))
    guided_share = 100.0 * guided / data["n_keep"]
    f_min, f_max = min(data["freq_all"]), max(data["freq_all"])
    kept_f_min, kept_f_max = min(data["freq"]), max(data["freq"])
    a_min, a_max = min(data["a"]), max(data["a"])
    h_min, h_max = min(data["h"]), max(data["h"])
    def _after_first(history: list[tuple[float, float]], index: int) -> list[tuple[float, float]]:
        return [(epoch, row[index]) for epoch, row in enumerate(history[1:], start=2)]

    loss_series = [
        {"name": "default-train", "color": "#2457a6", "points": _after_first(data["default_history"], 0)},
        {"name": "default-val", "color": "#2457a6", "dash": "5 4", "points": _after_first(data["default_history"], 1)},
        {"name": "retrain-train", "color": "#9c2b1e", "points": _after_first(data["retrain_history"], 0)},
        {"name": "retrain-val", "color": "#9c2b1e", "dash": "5 4", "points": _after_first(data["retrain_history"], 1)},
    ]
    loss_ymax = max(point[1] for item in loss_series for point in item["points"])
    calls = f"{data['n_keep']:,}".replace(",", " ")
    stats = [
        ("模态筛过后的行", f"{data['n_rows']:,}".replace(",", " ")),
        ("Q 不低于 100 万", calls),
        ("低于 100 万", f"{data['n_drop']:,}".replace(",", " ")),
        ("默认模型对数 Q 误差", f"{_metric(default, 'logQ_MAE'):.5f}"),
        ("默认模型频率误差", f"{_metric(default, 'freq_MAE_MHz'):.5f} MHz"),
        ("测试行", default["n_test"]),
    ]
    stat_html = "".join(
        f"<div class='stat'><span>{_esc(label)}</span><strong>{_esc(value)}</strong></div>"
        for label, value in stats
    )
    findings = [
        f"七个训练 CSV 合计 {data['n_rows']} 行，都已经通过呼吸模态筛选。Q 低于 100 万的有 {data['n_drop']} 行，剩下 {data['n_keep']} 行才进入网络的划分。",
        f"七个表的频率从 {f_min:.2f} 到 {f_max:.2f} MHz。Q 不低于 100 万的那一部分频率从 {kept_f_min:.2f} 到 {kept_f_max:.2f} MHz。品质因数最低大约 {data['q_min']:.3g}，最高大约 {data['q_max']:.3g}。",
        f"进入训练的几何：平均半径 {a_min:.2f} 到 {a_max:.2f} μm，平均环宽 {h_min:.2f} 到 {h_max:.2f} μm。",
        f"六阶 guided 文件通过 Q 过滤的有 {guided} 行，约占这 {data['n_keep']} 行的 {guided_share:.1f}%。",
        f"默认权重在 03_最终模型，跑了 {default['epochs_run']} 个 epoch。测试集对数 Q 平均绝对误差 {_metric(default, 'logQ_MAE'):.5f}，频率平均绝对误差 {_metric(default, 'freq_MAE_MHz'):.5f} MHz。",
        f"04_训练输出/final_mlp 是后来重训的，{retrain['epochs_run']} 个 epoch。对数 Q 误差 {_metric(retrain, 'logQ_MAE'):.5f}，频率误差 {_metric(retrain, 'freq_MAE_MHz'):.5f} MHz。两套不能当成同一个模型。",
        "这些品质因数和频率来自交付包里已经完成的仿真。不是本队今年的实测，也不能用来说明频率随温度更稳。表里没有温度。",
    ]
    finding_html = "".join(f"<li>{_esc(line)}</li>" for line in findings)
    legend = _legend(
        [
            ("default-train", "默认模型训练损失", "#2457a6"),
            ("default-val", "默认模型验证损失", "#2457a6"),
            ("retrain-train", "重训训练损失", "#9c2b1e"),
            ("retrain-val", "重训验证损失", "#9c2b1e"),
        ]
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>圆环代理：继承既有仿真</title>
<style>
:root {{ color-scheme: light; --ink:#1c1915; --paper:#f3efe6; --line:#d9d0c1; --muted:#5e574c; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:16px/1.55 "Segoe UI","PingFang SC","Noto Sans SC","Droid Sans Fallback","WenQuanYi Micro Hei",sans-serif; }}
main {{ max-width:1120px; margin:0 auto; padding:32px 20px 80px; }}
header h1 {{ font:600 40px/1.15 "Iowan Old Style","Palatino Linotype","Songti SC","Noto Serif SC",serif; margin:0 0 8px; }}
.kicker {{ letter-spacing:.16em; font-size:12px; color:var(--muted); margin:0 0 12px; }}
.lead {{ max-width:74ch; color:var(--muted); }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:20px 0; }}
.stat {{ background:#fffdf8; border:1px solid var(--line); padding:12px 14px; }}
.stat span {{ display:block; color:var(--muted); font-size:13px; }}
.stat strong {{ display:block; font:600 22px/1.25 "Iowan Old Style","Palatino Linotype","Noto Serif SC","Droid Sans Fallback",serif; overflow-wrap:anywhere; }}
section {{ margin-top:36px; }}
h2 {{ font:600 22px/1.3 "Iowan Old Style","Palatino Linotype","Songti SC",serif; margin:0 0 8px; }}
.note {{ color:var(--muted); margin:0 0 12px; max-width:78ch; }}
.chart {{ width:100%; height:auto; background:#fffdf8; border:1px solid var(--line); font-family:"Segoe UI","PingFang SC","Noto Sans SC","Droid Sans Fallback","WenQuanYi Micro Hei",sans-serif; }}
.frame {{ fill:#fffdf8; stroke:var(--line); }}
.grid {{ stroke:var(--line); stroke-width:1; }}
.lab {{ font-size:13px; fill:var(--ink); }}
.val, .tick {{ font-size:12px; fill:var(--muted); }}
.legend-row {{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 8px; }}
.legend {{ border:1px solid var(--line); background:#fffdf8; padding:4px 10px; cursor:pointer; font:inherit; }}
.legend::before {{ content:""; display:inline-block; width:10px; height:10px; background:var(--c); margin-right:6px; }}
.legend.off {{ opacity:.35; }}
.table-wrap {{ overflow-x:auto; margin:0 0 14px; }}
table.plain {{ border-collapse:collapse; width:100%; background:#fffdf8; }}
table.plain th, table.plain td {{ border:1px solid var(--line); padding:6px 8px; text-align:left; }}
.findings, ul {{ padding-left:1.2em; }}
footer {{ color:var(--muted); font-size:13px; margin-top:28px; }}
code {{ font-family:"Noto Sans Mono","DejaVu Sans Mono",monospace; font-size:0.92em; }}
</style>
</head>
<body>
<main>
<header>
<p class="kicker">2026 AIC · AI+学科交叉 · 继承既有圆环仿真</p>
<h1>已有的圆环仿真，收成一张能核对的图</h1>
<p class="lead">这页做的是继承和转化。继承的是交付包里已经跑完的圆环仿真和三层网络。转化是把筛选、几何分布和两套权重的测试误差摊开，让人不用翻 CSV 也能核对。品质因数和频率仍是那次仿真的结果，不是本队今年的实测。输入里没有温度。</p>
</header>
<div class="stats">{stat_html}</div>
<section>
<h2>读下来的几句</h2>
<ol class="findings">{finding_html}</ol>
</section>
<section id="files">
<h2>七个训练表怎么筛</h2>
<p class="note">行数直接数 <code>01_训练数据/</code> 里的 CSV。<code>width2</code> 到 <code>width8</code> 是最高傅里叶阶，不是环宽的微米数。Q 低于 100 万的行不进入网络划分。几何四舍五入到六位小数后，重复行有 {data['duplicate_rows']} 行，独立几何 {data['unique_geometries']} 个。</p>
{_table(["文件", "全部行", "Q≥100万", "低于100万"], file_rows)}
{_bars([(item["name"].removeprefix("training_q_").removesuffix(".csv"), float(item["kept"]), "#2457a6") for item in data["files"]], "通过 Q 过滤的行数")}
</section>
<section id="sources">
<h2>这些行当初怎么被抽出来</h2>
<p class="note">这是七个 CSV 的全部行，还没按 100 万过滤。<code>source_kind</code> 是抽样标签，不是一组把半径和环宽锁死的对照。过滤之后 <code>large_perturb</code> 从 {dict(data['sources'])['large_perturb']} 行降到 {keep_sources['large_perturb']} 行，所以不能从标签高低直接说「只因为更对称，品质因数就更高」。</p>
{_bars(source_rows, "全部行的抽样标签")}
</section>
<section id="geometry">
<h2>进入训练的半径和环宽</h2>
<p class="note">只画 Q 不低于 100 万的行。颜色越深，落在这一格的设计越多。半径大约 122 到 130 μm，环宽大约 8 到 13.5 μm。</p>
{_heat(geo_grid, 122, 130, 8, 13.5, "平均半径和平均环宽", "平均半径 μm", "平均环宽 μm")}
</section>
<section id="freq">
<h2>频率和品质因数落在哪一段</h2>
<p class="note">频率直方图用七个表的全部行，所以能看到 8 MHz 附近那一截。对数品质因数只画 Q 不低于 100 万的行。两边都是仿真表，不是温箱读数。</p>
{_bars([(f"{center - 0.05:.2f}–{center + 0.05:.2f}", float(count), "#2f6f62") for center, count in f_hist if count], "频率 MHz")}
{_bars([(f"{center - 0.05:.2f}–{center + 0.05:.2f}", float(count), "#8a4b2f") for center, count in q_hist if count], "log10(Q)")}
</section>
<section id="metrics">
<h2>两套权重的测试误差</h2>
<p class="note">数字从各自目录的 <code>deep_model_summary.csv</code> 抄来。两套划分相同：训练 {default['n_train']}、验证 {default['n_val']}、测试 {default['n_test']}。默认预测读 <code>03_最终模型</code>。找下一批几何时，只要重训权重还在，就会用 <code>04_训练输出/final_mlp</code>。对数 Q 误差 0.035 大约是品质因数差 8%。90% 分位仍有大约两成相对误差，适合挑哪一片几何大概更高，不适合写成精确测量。</p>
{_table(["目录", "epoch", "对数Q平均绝对误差", "对数Q的R²", "Q平均相对误差", "Q的90%分位相对误差", "频率平均绝对误差 MHz", "频率R²"], metric_rows)}
</section>
<section id="loss">
<h2>训练损失</h2>
<p class="note">实线是训练损失，虚线是验证损失。蓝色是默认模型，红色是重训。第 1 个 epoch 的训练损失大约 {data['default_history'][0][0]:.3f}，图从第 2 个 epoch 画起，免得后面的下降被压成一条。点图例可以只留一条。损失已经按对数 Q 和频率的权重算过，不是新的实验结果。</p>
{legend}
{_chart(loss_series, 2, max(len(data["default_history"]), len(data["retrain_history"])), 0, loss_ymax, "训练和验证损失", "epoch", "加权均方误差")}
</section>
<section id="pred">
<h2>重训模型在测试集上贴得有多近</h2>
<p class="note">只用 <code>04_训练输出/final_mlp/test_predictions.csv</code> 的 {len(data['pred_logq'])} 行。横轴是仿真值，纵轴是网络输出。红线是正好相等。格子颜色是行数。默认模型这一目录没有逐行预测文件，所以不在这里另画一套。</p>
{_heat(logq_grid, 6, 8, 6, 8, "对数品质因数", "仿真 log10(Q)", "预测 log10(Q)", diagonal=True)}
{_heat(freq_grid, 8, 10.2, 8, 10.2, "频率", "仿真频率 MHz", "预测频率 MHz", diagonal=True)}
</section>
<footer>
<p>页面由 <code>python -m plan_a.render</code> 读取 <code>圆环谐振器代理模型训练交付_20260929</code> 里的 CSV 生成。不重新训练，不调用 COMSOL。</p>
<p>本页是既有仿真的核对。没有本队今年的实测。不把这一页写成满分方案或国奖方案。</p>
</footer>
</main>
<script>
document.querySelectorAll(".legend").forEach((button) => {{
  button.addEventListener("click", () => {{
    const name = button.getAttribute("data-series");
    const section = button.closest("section");
    const lines = [...section.querySelectorAll("polyline[data-series]")];
    const mine = lines.filter((line) => line.getAttribute("data-series") === name);
    const solo = mine.length && mine.every((line) => line.style.opacity !== "0.12") && lines.some((line) => line.style.opacity === "0.12");
    lines.forEach((line) => {{
      const on = line.getAttribute("data-series") === name;
      line.style.opacity = solo ? "1" : (on ? "1" : "0.12");
    }});
    section.querySelectorAll(".legend").forEach((item) => {{
      item.classList.toggle("off", !solo && item.getAttribute("data-series") !== name);
    }});
    if (solo) section.querySelectorAll(".legend").forEach((item) => item.classList.remove("off"));
  }});
}});
</script>
</body>
</html>
"""


def write_page(package: Path | None = None, output: Path | None = None) -> Path:
    target = output or Path(__file__).resolve().parent / "output" / "index.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_page(package), encoding="utf-8")
    return target


def main() -> None:
    target = write_page()
    print(target)


if __name__ == "__main__":
    main()
