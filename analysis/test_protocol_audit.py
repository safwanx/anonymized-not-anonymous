import tempfile
import unittest
from collections import Counter
from pathlib import Path

from analysis.protocol_audit import FILENAME_RE, read_csv, run, split_audit


REPO_ROOT = Path(__file__).resolve().parents[1]


class ProtocolAuditTest(unittest.TestCase):
    def test_all_metadata_rows(self):
        rows = read_csv(REPO_ROOT / "protocol/metadata.csv")
        self.assertEqual(len(rows), 21600)
        self.assertEqual(len({r['filename'] for r in rows}), 21600)
        self.assertEqual(Counter(r['pool'] for r in rows),
                         {'pool_a': 9000, 'pool_b': 11880, 'bridge': 720})
        self.assertEqual(len({r['person'] for r in rows}), 38)
        self.assertEqual(len({r['action'] for r in rows}), 120)
        for row in rows:
            parsed = FILENAME_RE.search(row['filename']).groupdict()
            for key, value in parsed.items():
                self.assertEqual(int(row[key]), int(value))
            self.assertEqual(row['rgb_path'], 'rgb/' + row['filename'])
            self.assertEqual(row['skeleton_path'], 'skeletons/' + row['filename'].replace('_rgb.avi', '.skeleton'))

    def test_split_counts_and_corrected_chances(self):
        rows, chances = split_audit(
            REPO_ROOT / "protocol" / "splits", REPO_ROOT / "protocol" / "metadata.csv"
        )
        counts = {row["protocol"]: row["gallery_identity_count"] for row in rows}
        self.assertEqual(
            counts,
            {"crossview_cam2": 38, "crossview_cam3": 38, "crosssetup": 18, "crossrange": 3},
        )
        self.assertAlmostEqual(chances["crossview_cam2"]["macro"], 0.04858552631578948)
        self.assertAlmostEqual(chances["crossview_cam2"]["micro"], 0.050125)
        self.assertAlmostEqual(chances["crosssetup"]["macro"], 0.10511363636363633)
        self.assertAlmostEqual(chances["crossrange"]["macro"], 1 / 3)

    def test_end_to_end_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run(REPO_ROOT, output)
            expected = {
                "gallery_and_chance_audit.csv",
                "pose_corrected_null.csv",
                "protocol_results_unaveraged.csv",
                "identity_pool_extrapolation_audit.csv",
                "cpu_audit_summary.md",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            summary = (output / "cpu_audit_summary.md").read_text(encoding="utf-8")
            self.assertIn("9.88% vs 4.86%", summary)
            self.assertIn("not measurements on 106 people", summary)


if __name__ == "__main__":
    unittest.main()
