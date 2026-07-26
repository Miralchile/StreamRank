import csv
import tempfile
import unittest
from pathlib import Path

from streamrank.data.loader import load_interactions
from streamrank.data.prepare import prepare_kuairand_pure


class PrepareKuaiRandTests(unittest.TestCase):
    def test_prepares_deterministic_enriched_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            output = root / "output"
            raw.mkdir()
            (raw / "user_features_pure.csv").write_text(
                "user_id,user_active_degree\n1,high_active\n2,middle_active\n",
                encoding="utf-8",
            )
            (raw / "video_features_basic_pure.csv").write_text(
                "video_id,author_id,video_type,upload_dt,video_duration,tag\n"
                "10,99,NORMAL,2022-04-01,12000,7,8\n",
                encoding="utf-8",
            )
            header = (
                "user_id,video_id,date,time_ms,is_click,is_like,is_follow,is_hate,"
                "long_view,is_rand,tab\n"
            )
            rows = "1,10,20220410,1649548800000,1,0,0,0,1,0,1\n"
            for filename in (
                "log_standard_4_08_to_4_21_pure.csv",
                "log_standard_4_22_to_5_08_pure.csv",
                "log_random_4_22_to_5_08_pure.csv",
            ):
                (raw / filename).write_text(header + rows, encoding="utf-8")
            report = prepare_kuairand_pure(raw, output, users=0, seed=7)
            self.assertEqual(report["interactions"], 3)
            self.assertEqual(report["item_metadata_coverage"], 1.0)
            events = list(load_interactions(output / "interactions.csv"))
            self.assertEqual(events[0].category, "tag:7")
            self.assertEqual(events[0].author_id, 99)
            self.assertGreater(events[0].upload_time_ms, 0)
            self.assertEqual([event.logging_policy for event in events].count("random"), 1)
            with (output / "interactions.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)


if __name__ == "__main__":
    unittest.main()
