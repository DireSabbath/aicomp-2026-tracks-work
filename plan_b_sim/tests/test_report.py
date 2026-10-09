"""校赛技术方案核对圆环交付包的行数和两套权重误差，并守住引用边界。"""

import unittest

from plan_a.render import scan_package


ROOT_TITLE = "圆环代理预测频率与品质因数"


def _fmt_metrics(summary: dict) -> dict[str, str]:
    return {
        "epochs": str(int(summary["epochs_run"])),
        "n_train": str(int(summary["n_train"])),
        "n_val": str(int(summary["n_val"])),
        "n_test": str(int(summary["n_test"])),
        "logq": f"{float(summary['logQ_MAE']):.5f}",
        "logq_r2": f"{float(summary['logQ_R2']):.5f}",
        "q_rel": f"{float(summary['Q_mean_rel_err']) * 100:.2f}%",
        "q_p90": f"{float(summary['Q_p90_rel_err']) * 100:.2f}%",
        "freq": f"{float(summary['freq_MAE_MHz']):.5f}",
        "freq_r2": f"{float(summary['freq_R2']):.5f}",
    }


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        cls.report = (root / "技术方案.md").read_text(encoding="utf-8")
        cls.data = scan_package()

    def test_title_and_abstract_fit_the_submission_limits(self):
        self.assertIn(ROOT_TITLE, self.report)
        self.assertLessEqual(len(ROOT_TITLE), 20)
        intro = next(line for line in self.report.splitlines() if line.startswith("问题："))
        self.assertLessEqual(len(intro), 300)
        default = _fmt_metrics(self.data["default_summary"])
        retrain = _fmt_metrics(self.data["retrain_summary"])
        for token in (
            str(self.data["n_rows"]),
            str(self.data["n_keep"]),
            default["logq"],
            default["freq"],
            retrain["logq"],
            retrain["freq"],
            default["n_test"],
        ):
            self.assertIn(token, intro)

    def test_outline_quotes_the_package_logs(self):
        for heading in ("（一）", "（二）", "（三）", "（四）", "（五）", "（六）", "（七）", "（八）"):
            self.assertIn(heading, self.report)
        self.assertIn("python3 -m plan_a.render", self.report)
        self.assertIn("大模型", self.report)
        self.assertIn(str(self.data["n_rows"]), self.report)
        self.assertIn(str(self.data["n_keep"]), self.report)
        self.assertIn(str(self.data["n_drop"]), self.report)
        self.assertIn(str(self.data["duplicate_rows"]), self.report)
        self.assertIn(str(self.data["unique_geometries"]), self.report)
        for item in self.data["files"]:
            self.assertIn(item["name"], self.report)
            self.assertIn(str(item["total"]), self.report)
            self.assertIn(str(item["kept"]), self.report)
            self.assertIn(str(item["dropped"]), self.report)
        sources = dict(self.data["sources"])
        kept_sources = dict(self.data["sources_keep"])
        self.assertIn(str(sources["high_h_coverage"]), self.report)
        self.assertIn(str(sources["large_perturb"]), self.report)
        self.assertIn(str(kept_sources["large_perturb"]), self.report)
        self.assertIn(f"{min(self.data['freq_all']):.2f}", self.report)
        self.assertIn(f"{max(self.data['freq_all']):.2f}", self.report)
        self.assertIn(f"{min(self.data['freq']):.2f}", self.report)
        for summary in (self.data["default_summary"], self.data["retrain_summary"]):
            for token in _fmt_metrics(summary).values():
                self.assertIn(token, self.report)

    def test_score_claim_and_identity_boundaries(self):
        self.assertIn("不把这一版写成满分方案或国奖方案", self.report)
        self.assertIn("plan_b_sim", self.report)
        self.assertNotIn("python3 -m plan_b_sim", self.report)
        for banned in (
            "武汉",
            "学长",
            "满分项目",
            "冲击国奖",
            "可达满分",
            "保证满分",
            "±190",
            "190 ppb",
            "1361.5168",
        ):
            self.assertNotIn(banned, self.report)
        self.assertNotRegex(self.report, r"指导教师[:：]\s*\S+")
