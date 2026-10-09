"""校赛技术方案必须整段包含日志里的效果表，并守住引用边界。"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "技术方案.md"
TABLE = ROOT / "plan_b_sim" / "output" / "effect_table.md"


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.report = REPORT.read_text(encoding="utf-8")
        self.table = TABLE.read_text(encoding="utf-8").strip()

    def test_effect_table_is_copied_from_the_log(self):
        self.assertIn(self.table, self.report)

    def test_title_and_abstract_fit_the_submission_limits(self):
        self.assertIn("片内恒温谐振器三路线对照", self.report)
        self.assertLessEqual(len("片内恒温谐振器三路线对照"), 20)
        intro = next(
            line
            for line in self.report.splitlines()
            if line.startswith("问题：")
        )
        self.assertLessEqual(len(intro), 300)
        self.assertIn("1361.5168", intro)
        self.assertIn("162.88", intro)
        self.assertIn("89", intro)

    def test_closed_form_gap_is_stated(self):
        for token in (
            "162.88",
            "184.26",
            "191.54",
            "+73.88",
            "+72.26",
            "+53.54",
            "89 ℃",
            "112 ℃",
            "138 ℃",
            "局部极大",
            "没有改符号",
        ):
            self.assertIn(token, self.report)

    def test_outline_sections_and_reproduction_command(self):
        for heading in ("（一）", "（二）", "（三）", "（四）", "（五）", "（六）", "（七）", "（八）"):
            self.assertIn(heading, self.report)
        self.assertIn("python3 -m plan_b_sim", self.report)
        self.assertIn("大模型", self.report)

    def test_score_claim_and_identity_boundaries(self):
        self.assertIn("不把这一版写成满分方案或国奖方案", self.report)
        for banned in (
            "武汉",
            "学长",
            "满分项目",
            "冲击国奖",
            "可达满分",
            "保证满分",
            "±190",
            "190 ppb",
        ):
            self.assertNotIn(banned, self.report)
        # 规则要求材料不写这些，正文只作为禁止项出现，不填写具体名称。
        self.assertNotRegex(self.report, r"指导教师[:：]\s*\S+")
