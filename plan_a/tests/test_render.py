"""离线页面必须对上交付包里的行数和两套权重的测试误差。"""

import unittest

from plan_a.render import build_page, scan_package


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.data = scan_package()
        self.html = build_page()

    def test_scan_matches_the_delivered_tables(self):
        self.assertEqual(self.data["n_rows"], 21371)
        self.assertEqual(self.data["n_keep"], 19320)
        self.assertEqual(self.data["n_drop"], 2051)
        self.assertEqual(self.data["duplicate_rows"], 94)
        self.assertEqual(self.data["unique_geometries"], 21277)
        files = {item["name"]: item for item in self.data["files"]}
        guided = files["training_q_width6_guided_coverage.csv"]
        self.assertEqual(guided["kept"], 14181)
        self.assertEqual(guided["dropped"], 1768)
        self.assertEqual(dict(self.data["sources"])["high_h_coverage"], 4900)
        self.assertEqual(dict(self.data["sources"])["large_perturb"], 4159)
        self.assertEqual(len(self.data["default_history"]), 220)
        self.assertEqual(len(self.data["retrain_history"]), 209)
        self.assertEqual(len(self.data["pred_logq"]), 3864)
        self.assertAlmostEqual(min(self.data["freq_all"]), 8.00520593855, places=6)
        self.assertLess(min(self.data["freq"]), 9.40)
        self.assertGreater(min(self.data["freq"]), 9.30)

    def test_page_quotes_the_logged_metrics(self):
        for token in (
            "21371",
            "19320",
            "2051",
            "14181",
            "4900",
            "4159",
            "0.03515",
            "0.03728",
            "0.00453",
            "0.00409",
            "12673",
            "3864",
            "8.01",
            "10.14",
            "不是本队今年的实测",
            "不把这一页写成满分方案或国奖方案",
        ):
            self.assertIn(token, self.html)
        self.assertIn("<svg", self.html)
        self.assertIn("polyline", self.html)

    def test_page_keeps_the_citation_boundary(self):
        for banned in ("武汉", "学长", "190 ppb", "±190", "指导教师：", "1361.5168"):
            self.assertNotIn(banned, self.html)
