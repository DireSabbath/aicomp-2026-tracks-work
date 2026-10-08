"""论文式 (2-1)–(2-9) 的频率，以及一阶热路。

弹性常数的温度式按论文排版解读为

    c(T) = c0 * [1 + a_ppm * 1e-6 * (T-25) + b_ppb * 1e-9 * (T-25)^2]

硅、钼的线性项和二次项都是减号；氧化硅两项都是加号；掺钪氮化铝两项都是减号。
指数按 ppm、ppb 取 -6 和 -9。没有为了靠近论文有限元拐点而改符号或乘系数。

热阻取论文 COMSOL 给出的 3.85 ℃/mW，热时间常数取 37 ms。
稳态功率等于温差除以这个热阻，只用来核对积分器。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

T_REF_C = 25.0
WIDTH_M = 143e-6
# 3.85 ℃/mW = 3850 K/W。来自论文 40 μm 器件、环境 -40 ℃、加热到 110 ℃ 的 COMSOL。
R_TH_K_PER_W = 3.85e3
TAU_S = 37e-3
C_TH_J_PER_K = TAU_S / R_TH_K_PER_W
THICKNESS_UM = (20.0, 40.0, 60.0)
T_SET_C = 110.0
# 论文有限元拐点，只作核对，不是本队结果。
THESIS_FEM_TURNOVER_C = {20.0: 89.0, 40.0: 112.0, 60.0: 138.0}

# 层顺序：ScAlN、硅、钼、氧化硅。厚度加权与顺序无关。
_T_SCALN_UM = 1.0
_T_MO_UM = 0.15
_T_SIO2_UM = 0.2
_RHO = (3244.0, 2330.0, 10200.0, 2200.0)
_ALPHA = (4.9e-6, 2.84e-6, 4.8e-6, 0.55e-6)


@dataclass
class Calls:
    frequency: int = 0
    thermal: int = 0

    @property
    def total(self) -> int:
        return self.frequency + self.thermal

    def add(self, other: "Calls") -> None:
        self.frequency += other.frequency
        self.thermal += other.thermal


def steady_power_w(t_device_c: float, t_ambient_c: float, r_th: float = R_TH_K_PER_W) -> float:
    return (t_device_c - t_ambient_c) / r_th


def _poly(c0: float, a_ppm: float, b_ppb: float, d: np.ndarray | float):
    return c0 * (1.0 + a_ppm * 1e-6 * d + b_ppb * 1e-9 * d * d)


def young_moduli_gpa(temperature_c: np.ndarray | float):
    """返回四层杨氏模量，单位 GPa，顺序为 ScAlN、硅、钼、氧化硅。"""
    d = np.asarray(temperature_c, dtype=float) - T_REF_C
    c11 = _poly(159.33, -27.86, -56.55, d)
    c12 = _poly(65.09, -130.38, -12.67, d)
    e_si = c11 - 2.0 * c12 * c12 / (c11 + c12)
    a11 = _poly(410.06, -39.65, -20.61, d)
    a12 = _poly(100.69, -39.67, -19.51, d)
    a13 = _poly(100.69, -41.22, -19.88, d)
    a33 = _poly(386.24, -40.13, -20.03, d)
    e_sc = (a11 - a12) * ((a11 + a12) * a33 - 2.0 * a13 * a13) / (a11 * a33 - a13 * a13)
    e_mo = _poly(329.0, -117.0, -24.9, d)
    e_ox = _poly(70.0, 204.0, 221.0, d)
    return e_sc, e_si, e_mo, e_ox


def frequency_hz(temperature_c, thickness_um: float):
    """式 (2-1)–(2-3)。面内热膨胀按刚度加权，密度按各层体积变化。"""
    temp = np.asarray(temperature_c, dtype=float)
    e_sc, e_si, e_mo, e_ox = young_moduli_gpa(temp)
    thicknesses = (
        _T_SCALN_UM,
        float(thickness_um),
        _T_MO_UM,
        _T_SIO2_UM,
    )
    moduli = (e_sc, e_si, e_mo, e_ox)
    stiffness = 0.0
    for modulus, thickness in zip(moduli, thicknesses):
        stiffness = stiffness + modulus * thickness
    alpha_eff = 0.0
    for modulus, thickness, alpha in zip(moduli, thicknesses, _ALPHA):
        alpha_eff = alpha_eff + modulus * thickness * alpha
    alpha_eff = alpha_eff / stiffness
    d = temp - T_REF_C
    lateral = (1.0 + alpha_eff * d) ** 2
    e_sum = 0.0
    rho_sum = 0.0
    t_sum = 0.0
    for modulus, thickness, alpha, rho0 in zip(moduli, thicknesses, _ALPHA, _RHO):
        t_i = thickness * (1.0 + alpha * d)
        volume_scale = lateral * (1.0 + alpha * d)
        e_sum = e_sum + modulus * t_i
        rho_sum = rho_sum + (rho0 / volume_scale) * t_i
        t_sum = t_sum + t_i
    e_gpa = e_sum / t_sum
    rho = rho_sum / t_sum
    width = WIDTH_M * (1.0 + alpha_eff * d)
    freq = np.sqrt(e_gpa * 1e9 / rho) / (2.0 * width)
    if np.isscalar(temperature_c):
        return float(freq)
    return freq


def count_frequency(calls: Calls, n: int = 1) -> None:
    calls.frequency += int(n)


def frequency_hz_counted(temperature_c, thickness_um: float, calls: Calls):
    temp = np.asarray(temperature_c, dtype=float)
    count_frequency(calls, temp.size)
    return frequency_hz(temp if temp.ndim else float(temp), thickness_um)


@dataclass(frozen=True)
class Turnover:
    temperature_c: float | None
    kind: str
    tcf_at_110_ppm_per_c: float
    f_at_25_mhz: float
    f_at_110_hz: float


def find_turnover(thickness_um: float, calls: Calls) -> Turnover:
    """在 [-40, 280] ℃ 里找 df/dT 的第一个过零点，再用二分收紧。"""
    half = 0.05
    grid = np.arange(-40.0, 280.0 + 1e-9, 0.5)
    plus = frequency_hz_counted(grid + half, thickness_um, calls)
    minus = frequency_hz_counted(grid - half, thickness_um, calls)
    slope = (plus - minus) / (2.0 * half)
    bracket = None
    for i in range(slope.size - 1):
        left = float(slope[i])
        right = float(slope[i + 1])
        if left == 0.0:
            bracket = (float(grid[i]), float(grid[i]), left, left)
            break
        if left * right < 0.0:
            bracket = (float(grid[i]), float(grid[i + 1]), left, right)
            break
    f110 = float(frequency_hz_counted(110.0, thickness_um, calls))
    f109 = float(frequency_hz_counted(109.5, thickness_um, calls))
    f111 = float(frequency_hz_counted(110.5, thickness_um, calls))
    tcf = (f111 - f109) / f110 * 1e6
    f25 = float(frequency_hz_counted(T_REF_C, thickness_um, calls)) / 1e6
    if bracket is None:
        return Turnover(None, "none", tcf, f25, f110)
    a, b, dfa, dfb = bracket
    if a == b:
        kind = "flat"
        return Turnover(a, kind, tcf, f25, f110)
    for _ in range(50):
        mid = 0.5 * (a + b)
        fp = float(frequency_hz_counted(mid + half, thickness_um, calls))
        fm = float(frequency_hz_counted(mid - half, thickness_um, calls))
        dfm = (fp - fm) / (2.0 * half)
        if dfa * dfm <= 0.0:
            b, dfb = mid, dfm
        else:
            a, dfa = mid, dfm
    kind = "maximum" if dfa > 0.0 else "minimum"
    return Turnover(0.5 * (a + b), kind, tcf, f25, f110)


@dataclass(frozen=True)
class FrequencyTable:
    t0: float
    step: float
    values: np.ndarray
    f_at_set_hz: float

    def ppm(self, temperature_c: float) -> float:
        if temperature_c <= self.t0:
            freq = float(self.values[0])
        else:
            x = (temperature_c - self.t0) / self.step
            i = int(x)
            if i >= self.values.size - 1:
                freq = float(self.values[-1])
            else:
                w = x - i
                freq = float(self.values[i] * (1.0 - w) + self.values[i + 1] * w)
        return (freq - self.f_at_set_hz) / self.f_at_set_hz * 1e6


def build_frequency_table(
    thickness_um: float,
    calls: Calls,
    t0: float = -50.0,
    t1: float = 400.0,
    step: float = 0.02,
) -> FrequencyTable:
    n = int(round((t1 - t0) / step)) + 1
    grid = t0 + step * np.arange(n)
    values = frequency_hz_counted(grid, thickness_um, calls)
    f_set = float(frequency_hz_counted(T_SET_C, thickness_um, calls))
    return FrequencyTable(t0, step, np.asarray(values, dtype=float), f_set)


def thermal_step(temperature_c: float, t_ambient0: float, slew_c_per_s: float, dt_s: float, power_w: float) -> float:
    """Tamb 在步内线性变化、功率保持不变时的精确解。"""
    a_coeff = t_ambient0 + power_w * R_TH_K_PER_W - TAU_S * slew_c_per_s
    return (temperature_c - a_coeff) * math.exp(-dt_s / TAU_S) + a_coeff + slew_c_per_s * dt_s


@dataclass(frozen=True)
class Schedule:
    dt_s: float = 1e-3
    warmup_s: float = 2.0
    ramp_s: float = 30.0
    hold_s: float = 5.0
    t_cold_c: float = -40.0
    t_hot_c: float = 105.0
    t_set_c: float = T_SET_C
    band_c: float = 1.0

    @property
    def t_warm_end(self) -> float:
        return self.warmup_s

    @property
    def t_up_end(self) -> float:
        return self.warmup_s + self.ramp_s

    @property
    def t_hold_hot_end(self) -> float:
        return self.t_up_end + self.hold_s

    @property
    def t_down_end(self) -> float:
        return self.t_hold_hot_end + self.ramp_s

    @property
    def t_end(self) -> float:
        return self.t_down_end + self.hold_s

    @property
    def n_steps(self) -> int:
        return int(round(self.t_end / self.dt_s))

    def ambient(self, t_s: float) -> tuple[float, float]:
        span = self.t_hot_c - self.t_cold_c
        if t_s < self.t_warm_end:
            return self.t_cold_c, 0.0
        if t_s < self.t_up_end:
            return self.t_cold_c + span * (t_s - self.t_warm_end) / self.ramp_s, span / self.ramp_s
        if t_s < self.t_hold_hot_end:
            return self.t_hot_c, 0.0
        if t_s < self.t_down_end:
            return self.t_hot_c - span * (t_s - self.t_hold_hot_end) / self.ramp_s, -span / self.ramp_s
        return self.t_cold_c, 0.0


@dataclass
class Trajectory:
    energy_j: float
    settling_s: float | None
    max_abs_ppm_phase_b: float
    max_abs_dev_c_phase_b: float
    max_temperature_c: float
    end_temperature_c: float
    samples: list[tuple[float, float, float, float, float]] | None = None


def simulate(
    schedule: Schedule,
    thickness_um: float,
    kp: float,
    ki: float,
    pmax_w: float,
    calls: Calls,
    table: FrequencyTable | None,
    exact_frequency: bool,
    record_every: int = 0,
) -> Trajectory:
    """速度式 PI。功率状态本身被限幅，所以积分不会在饱和后继续堆积。

    主动和一起做共用这一套。被动把 Kp、Ki、功率上限都设为 0。
    """
    temperature = schedule.t_cold_c
    power = 0.0
    error_prev = schedule.t_set_c - temperature
    energy = 0.0
    max_ppm = 0.0
    max_dev = 0.0
    max_temperature = temperature
    last_outside = -1
    samples: list[tuple[float, float, float, float, float]] | None = [] if record_every else None
    f_set = None
    if exact_frequency:
        f_set = float(frequency_hz_counted(schedule.t_set_c, thickness_um, calls))
    n = schedule.n_steps
    dt = schedule.dt_s
    for i in range(n):
        t_s = i * dt
        t_ambient, slew = schedule.ambient(t_s)
        error = schedule.t_set_c - temperature
        delta_p = kp * (error - error_prev) + ki * error * dt
        power = power + delta_p
        if power > pmax_w:
            power = pmax_w
        elif power < 0.0:
            power = 0.0
        error_prev = error
        temperature = thermal_step(temperature, t_ambient, slew, dt, power)
        calls.thermal += 1
        energy += power * dt
        if temperature > max_temperature:
            max_temperature = temperature
        t_after = (i + 1) * dt
        in_trip = t_after >= schedule.t_warm_end
        if in_trip or (samples is not None and record_every and i % record_every == 0):
            if exact_frequency:
                freq = float(frequency_hz_counted(temperature, thickness_um, calls))
                ppm = (freq - f_set) / f_set * 1e6
            else:
                assert table is not None
                ppm = table.ppm(temperature)
        else:
            ppm = 0.0
        if in_trip:
            abs_ppm = abs(ppm)
            if abs_ppm > max_ppm:
                max_ppm = abs_ppm
            dev = abs(temperature - schedule.t_set_c)
            if dev > max_dev:
                max_dev = dev
        if abs(temperature - schedule.t_set_c) > schedule.band_c:
            last_outside = i
        if samples is not None and record_every and i % record_every == 0:
            samples.append((t_after, t_ambient, temperature, power, ppm if in_trip or exact_frequency else 0.0))
    if last_outside >= n - 1:
        settling = None
    elif last_outside < 0:
        settling = dt
    else:
        # 最后一次出带的下一步结束时，温度已经进带，并且之后没有再出去。
        settling = (last_outside + 2) * dt
    return Trajectory(energy, settling, max_ppm, max_dev, max_temperature, temperature, samples)
