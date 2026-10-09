import tempfile
import unittest
from pathlib import Path

import numpy as np

from plan_b_sim.campaign import CampaignConfig, build_grids, max_config, run_campaign
from plan_b_sim.model import (
    R_TH_K_PER_W,
    TAU_S,
    Calls,
    Schedule,
    build_frequency_table,
    find_turnover,
    frequency_hz,
    simulate,
    steady_power_w,
    thermal_step,
    young_moduli_gpa,
)


class FormulaTests(unittest.TestCase):
    def test_silicon_young_modulus_at_25c(self):
        _sc, e_si, e_mo, e_ox = young_moduli_gpa(25.0)
        expected = 159.33 - 2.0 * 65.09**2 / (159.33 + 65.09)
        self.assertAlmostEqual(float(e_si), expected, places=9)
        self.assertAlmostEqual(float(e_mo), 329.0, places=9)
        self.assertAlmostEqual(float(e_ox), 70.0, places=9)

    def test_frequency_is_near_26_mhz_and_falls_as_silicon_gets_thicker(self):
        freqs = [frequency_hz(25.0, h) for h in (20.0, 40.0, 60.0)]
        for freq in freqs:
            self.assertGreater(freq, 20e6)
            self.assertLess(freq, 30e6)
        self.assertGreater(freqs[0], freqs[1])
        self.assertGreater(freqs[1], freqs[2])

    def test_scalar_and_array_frequency_match(self):
        temps = np.array([-40.0, 25.0, 110.0, 160.0])
        bulk = frequency_hz(temps, 40.0)
        for temp, freq in zip(temps, bulk):
            self.assertAlmostEqual(float(freq), frequency_hz(float(temp), 40.0), places=6)

    def test_turnovers_keep_thickness_order_and_are_not_fitted(self):
        calls = Calls()
        found = {h: find_turnover(h, calls) for h in (20.0, 40.0, 60.0)}
        temps = [found[h].temperature_c for h in (20.0, 40.0, 60.0)]
        self.assertTrue(all(t is not None for t in temps))
        self.assertLess(temps[0], temps[1])
        self.assertLess(temps[1], temps[2])
        for turn in found.values():
            self.assertEqual(turn.kind, "maximum")
            self.assertGreater(turn.temperature_c, 150.0)
        # 论文有限元是 89 / 112 / 138 ℃。闭式结果必须离这三档足够远，避免有人乘系数去凑。
        for h, thesis in ((20.0, 89.0), (40.0, 112.0), (60.0, 138.0)):
            self.assertGreater(abs(found[h].temperature_c - thesis), 40.0)

    def test_table_interpolation_stays_under_a_millippm(self):
        calls = Calls()
        table = build_frequency_table(20.0, calls, t0=-40.0, t1=160.0, step=0.02)
        f_set = frequency_hz(110.0, 20.0)
        for temp in np.linspace(-40.0, 150.0, 25):
            exact = abs(frequency_hz(float(temp), 20.0) - f_set) / f_set * 1e6
            # ppm() 带符号，这里比绝对偏差
            signed = (frequency_hz(float(temp), 20.0) - f_set) / f_set * 1e6
            self.assertAlmostEqual(table.ppm(float(temp)), signed, delta=1e-3)
            self.assertAlmostEqual(abs(table.ppm(float(temp))), exact, delta=1e-3)

    def test_steady_power_uses_published_resistance(self):
        power = steady_power_w(110.0, -40.0)
        self.assertAlmostEqual(power, 150.0 / 3850.0, places=12)
        self.assertAlmostEqual(R_TH_K_PER_W, 3850.0, places=12)

    def test_thermal_step_holds_equilibrium_and_approaches_ambient(self):
        power = steady_power_w(110.0, -40.0)
        held = thermal_step(110.0, -40.0, 0.0, 1e-3, power)
        self.assertAlmostEqual(held, 110.0, places=9)
        temp = -40.0
        for _ in range(4000):
            temp = thermal_step(temp, 25.0, 0.0, 1e-3, 0.0)
        self.assertAlmostEqual(temp, 25.0, places=3)

    def test_low_power_cannot_reach_110_from_cold(self):
        temp = -40.0
        power = 0.02
        for _ in range(5000):
            temp = thermal_step(temp, -40.0, 0.0, 1e-3, power)
        self.assertLess(temp, 50.0)
        self.assertAlmostEqual(temp, -40.0 + power * R_TH_K_PER_W, places=2)

    def test_controller_holds_setpoint_when_power_cap_is_enough(self):
        schedule = Schedule(dt_s=1e-3, warmup_s=0.2, ramp_s=0.4, hold_s=0.1)
        calls = Calls()
        traj = simulate(schedule, 20.0, 1e-2, 10.0, 0.08, calls, table=None, exact_frequency=True)
        self.assertIsNotNone(traj.settling_s)
        self.assertLess(traj.max_abs_dev_c_phase_b, 0.05)
        self.assertGreater(traj.energy_j, 0.0)
        self.assertGreater(calls.thermal, 0)
        self.assertGreater(calls.frequency, 0)
        self.assertAlmostEqual(TAU_S, 0.037)


class CampaignTests(unittest.TestCase):
    def test_max_grids_have_equal_length(self):
        grids = build_grids(max_config())
        lengths = {name: len(grid) for name, grid in grids.items()}
        self.assertEqual(len(set(lengths.values())), 1)
        self.assertGreater(lengths["active"], 1000)

    def test_tiny_campaign_is_repeatable_and_balances_calls(self):
        config = CampaignConfig(
            schedule=Schedule(dt_s=0.01, warmup_s=0.05, ramp_s=0.1, hold_s=0.05),
            kp_active=(1e-3, 1e-2),
            ki_active=(1.0, 10.0),
            pmax_active=(0.02, 0.05, 0.1),
            kp_joint=(1e-3, 1e-2),
            ki_joint=(1.0, 10.0),
            pmax_joint=(0.1,),
            n_finalists=3,
            rank_ppm_tol=1e-4,
            record_every=5,
        )
        grids = build_grids(config)
        self.assertEqual(len(grids["passive"]), len(grids["active"]))
        self.assertEqual(len(grids["active"]), len(grids["joint"]))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first = run_campaign(config, out, processes=1)
            table = (out / "effect_table.md").read_text(encoding="utf-8")
            second = run_campaign(config, out, processes=1)
            self.assertEqual(table, second["table_markdown"])
            self.assertEqual(len({route["winner"]["model_calls"] for route in first["routes"]}), 1)
            passive = first["routes"][0]["winner"]
            self.assertEqual(passive["route"], "passive")
            self.assertEqual(passive["energy_j"], 0.0)
            self.assertEqual(passive["settling_s"], "不加热")
            self.assertNotIn("190", table)
            self.assertIn("不加热", table)


if __name__ == "__main__":
    unittest.main()
