"""三条路线共用一条环境曲线、同一种速度式 PI，调用次数对齐。"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plan_b_sim.model import (
    R_TH_K_PER_W,
    THESIS_FEM_TURNOVER_C,
    THICKNESS_UM,
    C_TH_J_PER_K,
    TAU_S,
    Calls,
    Schedule,
    build_frequency_table,
    find_turnover,
    frequency_hz_counted,
    simulate,
    steady_power_w,
)

_CTX: dict = {}


@dataclass(frozen=True)
class CampaignConfig:
    schedule: Schedule
    kp_active: tuple[float, ...]
    ki_active: tuple[float, ...]
    pmax_active: tuple[float, ...]
    kp_joint: tuple[float, ...]
    ki_joint: tuple[float, ...]
    pmax_joint: tuple[float, ...]
    n_finalists: int
    rank_ppm_tol: float
    record_every: int


def max_config() -> CampaignConfig:
    return CampaignConfig(
        schedule=Schedule(),
        kp_active=tuple(float(x) for x in np.geomspace(1e-5, 3e-2, 15)),
        ki_active=tuple(float(x) for x in np.geomspace(1e-2, 30.0, 16)),
        pmax_active=(0.02, 0.03, 0.035, 0.039, 0.042, 0.05, 0.08, 0.12),
        kp_joint=tuple(float(x) for x in np.geomspace(1e-5, 3e-2, 10)),
        ki_joint=tuple(float(x) for x in np.geomspace(1e-2, 30.0, 8)),
        pmax_joint=(0.02, 0.03, 0.035, 0.039, 0.042, 0.05, 0.08, 0.12),
        n_finalists=12,
        rank_ppm_tol=1e-4,
        record_every=50,
    )


def _product(thicknesses, kps, kis, pmaxs):
    return [
        (float(h), float(kp), float(ki), float(pmax))
        for h in thicknesses
        for kp in kps
        for ki in kis
        for pmax in pmaxs
    ]


def build_grids(config: CampaignConfig) -> dict[str, list[tuple[float, float, float, float]]]:
    active = _product((20.0,), config.kp_active, config.ki_active, config.pmax_active)
    joint = _product(THICKNESS_UM, config.kp_joint, config.ki_joint, config.pmax_joint)
    if len(active) != len(joint):
        raise RuntimeError(f"主动网格 {len(active)} 与一起做网格 {len(joint)} 次数不同")
    if len(active) % 3 != 0:
        raise RuntimeError("网格长度必须能被三档厚度整除，被动路线才能重复跑满")
    reps = len(active) // 3
    passive = [(float(h), 0.0, 0.0, 0.0) for h in THICKNESS_UM for _ in range(reps)]
    return {"passive": passive, "active": active, "joint": joint}


def _search_task(task: tuple[float, float, float, float]):
    thickness, kp, ki, pmax = task
    calls = Calls()
    traj = simulate(
        _CTX["schedule"],
        thickness,
        kp,
        ki,
        pmax,
        calls,
        _CTX["tables"][thickness],
        exact_frequency=False,
    )
    return {
        "thickness_um": thickness,
        "kp": kp,
        "ki": ki,
        "pmax_w": pmax,
        "energy_j": traj.energy_j,
        "settling_s": traj.settling_s,
        "ppm": traj.max_abs_ppm_phase_b,
        "dev_c": traj.max_abs_dev_c_phase_b,
        "thermal_calls": calls.thermal,
        "frequency_calls": calls.frequency,
    }


def _rank_key(row: dict, turnover_error: dict[float, float], tol: float):
    settle = row["settling_s"] if row["settling_s"] is not None else 1e9
    return (
        math.floor(row["ppm"] / tol + 1e-12),
        row["energy_j"],
        turnover_error[row["thickness_um"]],
        settle,
        row["pmax_w"],
        row["kp"],
        row["ki"],
        row["thickness_um"],
    )


def _prepare_curves(calls: Calls):
    turnovers = {h: find_turnover(h, calls) for h in THICKNESS_UM}
    tables = {h: build_frequency_table(h, calls) for h in THICKNESS_UM}
    grid = np.arange(-40.0, 220.0 + 1e-9, 0.5)
    curves = {h: np.asarray(frequency_hz_counted(grid, h, calls), dtype=float) for h in THICKNESS_UM}
    return turnovers, tables, grid, curves


def _exact_row(schedule, row, calls: Calls):
    traj = simulate(
        schedule,
        row["thickness_um"],
        row["kp"],
        row["ki"],
        row["pmax_w"],
        calls,
        table=None,
        exact_frequency=True,
    )
    copied = dict(row)
    copied["energy_j"] = traj.energy_j
    copied["settling_s"] = traj.settling_s
    copied["ppm"] = traj.max_abs_ppm_phase_b
    copied["dev_c"] = traj.max_abs_dev_c_phase_b
    copied["exact"] = True
    return copied


def _run_route(name: str, grid, config: CampaignConfig, processes: int):
    calls = Calls()
    turnovers, tables, curve_t, curves = _prepare_curves(calls)
    turnover_error = {}
    for thickness, turn in turnovers.items():
        if turn.temperature_c is None:
            turnover_error[thickness] = 1e9
        else:
            turnover_error[thickness] = abs(turn.temperature_c - config.schedule.t_set_c)
    _CTX["schedule"] = config.schedule
    _CTX["tables"] = tables
    started = time.perf_counter()
    if processes <= 1 or len(grid) == 1:
        ranked_pool = [_search_task(task) for task in grid]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(processes) as pool:
            ranked_pool = pool.map(_search_task, grid, chunksize=16)
    search_s = time.perf_counter() - started
    for row in ranked_pool:
        calls.thermal += row["thermal_calls"]
        calls.frequency += row["frequency_calls"]
    ranked_pool.sort(key=lambda row: _rank_key(row, turnover_error, config.rank_ppm_tol))
    best_by_thickness = []
    seen = set()
    for row in ranked_pool:
        thickness = row["thickness_um"]
        if thickness in seen:
            continue
        seen.add(thickness)
        best_by_thickness.append(row)
    final_src = ranked_pool[: config.n_finalists]
    finalists = [_exact_row(config.schedule, row, calls) for row in final_src]
    finalists.sort(key=lambda row: _rank_key(row, turnover_error, config.rank_ppm_tol))
    winner = finalists[0]
    recorded = simulate(
        config.schedule,
        winner["thickness_um"],
        winner["kp"],
        winner["ki"],
        winner["pmax_w"],
        calls,
        table=None,
        exact_frequency=True,
        record_every=config.record_every,
    )
    # 公布的五个数来自这条重新积分的轨迹，和决赛圈用的是同一个公式。
    winner = dict(winner)
    winner["energy_j"] = recorded.energy_j
    winner["settling_s"] = recorded.settling_s
    winner["ppm"] = recorded.max_abs_ppm_phase_b
    winner["dev_c"] = recorded.max_abs_dev_c_phase_b
    if name == "passive" and winner["energy_j"] != 0.0:
        raise RuntimeError(f"被动路线加热能耗不是 0：{winner['energy_j']}")
    return {
        "name": name,
        "calls": calls,
        "turnovers": turnovers,
        "curve_t": curve_t,
        "curves": curves,
        "search_s": search_s,
        "n_settled": sum(row["settling_s"] is not None for row in ranked_pool),
        "leaderboard": ranked_pool[:15],
        "best_by_thickness": best_by_thickness,
        "finalists": finalists,
        "winner": winner,
        "samples": recorded.samples,
    }


def _perfect_hold_energy(schedule: Schedule) -> float:
    energy = 0.0
    for i in range(schedule.n_steps):
        t_ambient, _slew = schedule.ambient(i * schedule.dt_s)
        energy += max(0.0, schedule.t_set_c - t_ambient) / R_TH_K_PER_W * schedule.dt_s
    return energy


def _fmt_ppm(value: float) -> str:
    if value >= 0.01:
        return f"{value:.4f}"
    return f"{value:.6e}"


def _route_label(name: str, thickness: float) -> str:
    titles = {"passive": "只做被动", "active": "只做主动", "joint": "一起做"}
    return f"{titles[name]}，硅 {thickness:.0f} μm"


def _compact_candidate(row: dict) -> dict:
    return {
        "thickness_um": row["thickness_um"],
        "kp": row["kp"],
        "ki": row["ki"],
        "pmax_w": row["pmax_w"],
        "residual_ppm": row["ppm"],
        "energy_j": row["energy_j"],
        "settling_s": row["settling_s"],
        "dev_c": row["dev_c"],
    }


def _markdown(routes: list[dict], n_calls: int, perfect_hold_j: float) -> str:
    lines = [
        "| 路线 | 拐点误差 ℃ | 剩余频率偏差 ppm | 加热能耗 J | 停稳时间 | 模型调用次数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for route in routes:
        winner = route["winner"]
        turn = route["turnovers"][winner["thickness_um"]]
        if turn.temperature_c is None:
            err = ""
        else:
            err = f"{turn.temperature_c - 110.0:+.2f}"
        if route["name"] == "passive":
            settle = "不加热"
            energy = "0"
        else:
            settle = "" if winner["settling_s"] is None else f"{winner['settling_s']:.3f}"
            energy = f"{winner['energy_j']:.6f}"
        lines.append(
            f"| {_route_label(route['name'], winner['thickness_um'])} | {err} | {_fmt_ppm(winner['ppm'])} | {energy} | {settle} | {n_calls} |"
        )
    lines.append("")
    lines.extend(_table_notes(routes, n_calls, perfect_hold_j))
    return "\n".join(lines) + "\n"


def _table_notes(routes: list[dict], n_calls: int, perfect_hold_j: float) -> list[str]:
    by_name = {route["name"]: route for route in routes}
    lines = ["## 这张表怎么读", ""]
    lines.append("拐点误差是频率对温度的导数过零的温度减去 110 ℃。这条闭式公式的过零是局部极大。")
    for thickness in THICKNESS_UM:
        turn = by_name["passive"]["turnovers"][thickness]
        thesis = THESIS_FEM_TURNOVER_C[thickness]
        lines.append(
            f"- 硅 {thickness:.0f} μm：过零 {turn.temperature_c:.2f} ℃，"
            f"论文有限元 {thesis:.0f} ℃，差 {turn.temperature_c - thesis:+.2f} ℃，"
            f"110 ℃ 处斜率 {turn.tcf_at_110_ppm_per_c:.2f} ppm/℃。"
        )
    lines.append("差留在 `summary.json` 的 `gap_vs_thesis_fem_c`。公式没有改符号，也没有乘系数。")
    lines.append("")
    lines.append(
        "剩余频率偏差只看来回那一段，也就是环境离开 −40 ℃ 之后。前 2 秒升温写在停稳时间里。"
        "剩余频率偏差小于 0.0001 ppm 的候选先算同一档，然后比加热能耗，再比拐点离 110 ℃ 的绝对距离。"
    )
    lines.append("")
    lines.append("三档过零都在 110 ℃ 上面，最近的是 20 μm。主动路线的厚度固定是 20 μm，一起做在三档里选出的也是 20 μm。")
    lines.append("各厚度在搜索阶段排第一的候选如下。搜索阶段的频率来自 0.02 ℃ 查表，公布行用公式重算。")
    for route_name, title in (("passive", "只做被动"), ("joint", "一起做")):
        lines.append(f"- {title}")
        for row in by_name[route_name]["best_by_thickness"]:
            if route_name == "passive":
                settle = "不加热"
            elif row["settling_s"] is None:
                settle = "未停稳"
            else:
                settle = f"停稳 {row['settling_s']:.3f} s"
            lines.append(
                f"  - {row['thickness_um']:.0f} μm：偏差 {_fmt_ppm(row['ppm'])} ppm，"
                f"能耗 {row['energy_j']:.6f} J，{settle}，"
                f"功率上限 {row['pmax_w']:.3f} W"
            )
    active = by_name["active"]["winner"]
    joint = by_name["joint"]["winner"]
    lines.append("")
    lines.append(
        f"公布的主动控制器：Kp = {active['kp']:.6g} W/℃，Ki = {active['ki']:.6g} W/(℃·s)，"
        f"功率上限 {active['pmax_w']:.3f} W。"
    )
    lines.append(
        f"公布的一起做控制器：Kp = {joint['kp']:.6g} W/℃，Ki = {joint['ki']:.6g} W/(℃·s)，"
        f"功率上限 {joint['pmax_w']:.3f} W，硅 {joint['thickness_um']:.0f} μm。"
    )
    needed = steady_power_w(110.0, -40.0)
    lines.append(
        f"环境停在 −40 ℃ 时，把器件维持在 110 ℃ 需要 {needed:.6f} W。"
        "这个数等于温差除以论文公布的热阻，用来核对热路，不填进上表的能耗。"
    )
    lines.append(
        f"若温度从一开始就停在 110 ℃，这条环境曲线上的加热积分是 {perfect_hold_j:.6f} J。"
        f"主动路线的积分是 {active['energy_j']:.6f} J，一起做是 {joint['energy_j']:.6f} J。"
    )
    lines.append(f"三行模型调用次数都是 {n_calls}。被动三档算完以后，剩下的次数继续跑不加热的同一条轨迹。")
    return lines


def _svg(route: dict, path: Path) -> None:
    grid = route["curve_t"]
    series = []
    for thickness, freq in route["curves"].items():
        f_set = route["turnovers"][thickness].f_at_110_hz
        ppm = (freq - f_set) / f_set * 1e6
        series.append((thickness, ppm))
    width, height = 720, 420
    left, right, top, bottom = 70, 20, 24, 48
    t_min, t_max = -40.0, 220.0
    flat = np.concatenate([ppm[(grid >= t_min) & (grid <= t_max)] for _h, ppm in series])
    y_min = min(-100.0, float(np.min(flat)) * 1.05)
    y_max = max(100.0, float(np.max(flat)) * 1.05)

    def xy(t_c, ppm):
        x = left + (t_c - t_min) / (t_max - t_min) * (width - left - right)
        y = top + (y_max - ppm) / (y_max - y_min) * (height - top - bottom)
        return x, y

    colors = {20.0: "#b45309", 40.0: "#1d4ed8", 60.0: "#047857"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafaf9"/>',
        f'<text x="{left}" y="18" font-size="14" font-family="sans-serif">频率相对 110 ℃ 的偏差（公式，不是实测）</text>',
    ]
    x110, _ = xy(110.0, 0.0)
    parts.append(f'<line x1="{x110:.1f}" y1="{top}" x2="{x110:.1f}" y2="{height-bottom}" stroke="#a8a29e" stroke-dasharray="4 3"/>')
    _, y0 = xy(0.0, 0.0)
    parts.append(f'<line x1="{left}" y1="{y0:.1f}" x2="{width-right}" y2="{y0:.1f}" stroke="#d6d3d1"/>')
    for thickness, ppm in series:
        points = []
        for t_c, y_ppm in zip(grid, ppm):
            if t_min <= t_c <= t_max:
                x, y = xy(float(t_c), float(y_ppm))
                points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline fill="none" stroke="{colors[thickness]}" stroke-width="2" points="{" ".join(points)}"/>')
        turn = route["turnovers"][thickness].temperature_c
        if turn is not None and t_min <= turn <= t_max:
            f_set = route["turnovers"][thickness].f_at_110_hz
            # 标记用曲线上最接近拐点的点
            idx = int(np.argmin(np.abs(grid - turn)))
            y_ppm = float((route["curves"][thickness][idx] - f_set) / f_set * 1e6)
            x, y = xy(turn, y_ppm)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{colors[thickness]}"/>')
    parts.append(f'<text x="{left}" y="{height-16}" font-size="12" font-family="sans-serif" fill="#57534e">温度 ℃    竖线是 110 ℃    点是公式拐点</text>')
    legend_x = width - 160
    for i, thickness in enumerate(THICKNESS_UM):
        y = 40 + i * 18
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+24}" y2="{y}" stroke="{colors[thickness]}" stroke-width="2"/>')
        parts.append(f'<text x="{legend_x+30}" y="{y+4}" font-size="12" font-family="sans-serif">{thickness:.0f} μm</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _jsonable_turnovers(turnovers: dict) -> dict:
    out = {}
    for thickness, turn in turnovers.items():
        thesis = THESIS_FEM_TURNOVER_C[thickness]
        out[str(int(thickness))] = {
            "temperature_c": turn.temperature_c,
            "kind": turn.kind,
            "error_vs_110_c": None if turn.temperature_c is None else turn.temperature_c - 110.0,
            "tcf_at_110_ppm_per_c": turn.tcf_at_110_ppm_per_c,
            "f_mhz_25": turn.f_at_25_mhz,
            "f_hz_110": turn.f_at_110_hz,
            "thesis_fem_turnover_c": thesis,
            "gap_vs_thesis_fem_c": None if turn.temperature_c is None else turn.temperature_c - thesis,
        }
    return out


def _public_row(route: dict, n_calls: int) -> dict:
    winner = route["winner"]
    turn = route["turnovers"][winner["thickness_um"]]
    settling = "不加热" if route["name"] == "passive" else winner["settling_s"]
    return {
        "route": route["name"],
        "thickness_um": winner["thickness_um"],
        "kp_w_per_k": winner["kp"],
        "ki_w_per_k_s": winner["ki"],
        "pmax_w": winner["pmax_w"],
        "turnover_c": turn.temperature_c,
        "turnover_kind": turn.kind,
        "turnover_error_c": None if turn.temperature_c is None else turn.temperature_c - 110.0,
        "residual_ppm": winner["ppm"],
        "energy_j": 0.0 if route["name"] == "passive" else winner["energy_j"],
        "settling_s": settling,
        "max_abs_temperature_error_c": winner["dev_c"],
        "model_calls": n_calls,
    }


def run_campaign(config: CampaignConfig, out_dir: Path, processes: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    grids = build_grids(config)
    routes = []
    for name in ("passive", "active", "joint"):
        print(f"跑{name}，候选 {len(grids[name])}", flush=True)
        routes.append(_run_route(name, grids[name], config, processes))
    totals = [route["calls"].total for route in routes]
    if len(set(totals)) != 1:
        raise RuntimeError(f"三行调用次数不相等：{totals}")
    n_calls = totals[0]
    perfect = _perfect_hold_energy(config.schedule)
    table = _markdown(routes, n_calls, perfect)
    (out_dir / "effect_table.md").write_text(table, encoding="utf-8")
    _svg(routes[0], out_dir / "frequency_ppm.svg")
    grid = routes[0]["curve_t"]
    curve_lines = ["temperature_c,f_hz_20um,f_hz_40um,f_hz_60um,ppm_20um,ppm_40um,ppm_60um"]
    for i, t_c in enumerate(grid):
        freqs = []
        ppms = []
        for thickness in THICKNESS_UM:
            freq = float(routes[0]["curves"][thickness][i])
            f_set = routes[0]["turnovers"][thickness].f_at_110_hz
            freqs.append(f"{freq:.8f}")
            ppms.append(f"{(freq - f_set) / f_set * 1e6:.8f}")
        curve_lines.append(f"{float(t_c):.1f}," + ",".join(freqs + ppms))
    (out_dir / "frequency_curve.csv").write_text("\n".join(curve_lines) + "\n", encoding="utf-8")

    trace_lines = ["route,t_s,t_ambient_c,t_device_c,power_w,ppm"]
    for route in routes:
        for sample in route["samples"] or []:
            t_s, t_ambient, t_device, power, ppm = sample
            trace_lines.append(
                f"{route['name']},{t_s:.6f},{t_ambient:.6f},{t_device:.6f},{power:.8e},{ppm:.8e}"
            )
    (out_dir / "winner_traces.csv").write_text("\n".join(trace_lines) + "\n", encoding="utf-8")

    rows = [_public_row(route, n_calls) for route in routes]
    csv_header = ",".join(rows[0].keys())
    csv_body = [csv_header]
    for row in rows:
        csv_body.append(",".join("" if row[key] is None else str(row[key]) for key in rows[0].keys()))
    (out_dir / "effect_table.csv").write_text("\n".join(csv_body) + "\n", encoding="utf-8")

    summary = {
        "model_calls_each_route": n_calls,
        "call_definition": "频率公式每求值一次计 1，热路每积分一步计 1。查表插值不计。",
        "rank_ppm_tol": config.rank_ppm_tol,
        "schedule_s": {
            "dt": config.schedule.dt_s,
            "warmup": config.schedule.warmup_s,
            "ramp": config.schedule.ramp_s,
            "hold": config.schedule.hold_s,
            "end": config.schedule.t_end,
        },
        "thermal": {
            "R_K_per_W": R_TH_K_PER_W,
            "tau_s": TAU_S,
            "C_J_per_K": C_TH_J_PER_K,
            "steady_power_W_from_-40_to_110": steady_power_w(110.0, -40.0),
            "perfect_hold_energy_J": perfect,
            "note": "热阻和热时间常数用的是论文 COMSOL 公布值。稳态功率等于温差除以热阻，是积分器核对，不是效果表里的能耗。",
        },
        "frequency_check": _jsonable_turnovers(routes[0]["turnovers"]),
        "frequency_check_note": "闭式拐点与论文有限元 89/112/138 ℃ 的差写在 gap_vs_thesis_fem_c。没有乘系数去凑。三行的频率检查相同。",
        "routes": [],
        "table_markdown": table,
    }
    for route in routes:
        summary["routes"].append(
            {
                "name": route["name"],
                "grid": len(grids[route["name"]]),
                "search_seconds": route["search_s"],
                "n_settled": route["n_settled"],
                "frequency_calls": route["calls"].frequency,
                "thermal_calls": route["calls"].thermal,
                "winner": rows[["passive", "active", "joint"].index(route["name"])],
                "best_search_by_thickness": [_compact_candidate(row) for row in route["best_by_thickness"]],
                "finalists": [
                    {
                        "thickness_um": item["thickness_um"],
                        "kp": item["kp"],
                        "ki": item["ki"],
                        "pmax_w": item["pmax_w"],
                        "residual_ppm": item["ppm"],
                        "energy_j": item["energy_j"],
                        "settling_s": item["settling_s"],
                        "dev_c": item["dev_c"],
                    }
                    for item in route["finalists"]
                ],
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(table, flush=True)
    return summary


def main() -> None:
    out = Path(__file__).resolve().parent / "output"
    processes = min(4, os.cpu_count() or 1)
    run_campaign(max_config(), out, processes)
