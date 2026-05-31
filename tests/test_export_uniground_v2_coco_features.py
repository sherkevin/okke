from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from export_uniground_v2_coco_features import build_records, load_captions, load_instances


class ExportUniGroundV2CocoFeaturesTest(unittest.TestCase):
    def test_build_records_creates_support_contradiction_and_abstain(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_dir = root / "images"
            image_dir.mkdir()
            image_path = image_dir / "sample.jpg"
            Image.new("RGB", (32, 32), color="white").save(image_path)

            instances_json = root / "instances.json"
            captions_json = root / "captions.json"

            instances_json.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "sample.jpg"}],
                        "categories": [
                            {"id": 1, "name": "cat"},
                            {"id": 2, "name": "dog"},
                            {"id": 3, "name": "car"},
                        ],
                        "annotations": [
                            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100},
                            {"image_id": 1, "category_id": 2, "bbox": [10, 10, 12, 12], "area": 144},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            captions_json.write_text(
                json.dumps({"annotations": [{"image_id": 1, "caption": "a cat and a dog on the floor"}]}),
                encoding="utf-8",
            )

            categories, image_paths, image_to_annotations = load_instances(instances_json)
            image_to_captions = load_captions(captions_json)
            records, selected_images = build_records(
                image_dir=image_dir,
                categories=categories,
                image_paths=image_paths,
                image_to_annotations=image_to_annotations,
                image_to_captions=image_to_captions,
                max_images=1,
                max_regions_per_record=2,
                seed=7,
            )

            self.assertEqual(len(selected_images), 1)
            self.assertEqual(len(records), 3)
            self.assertEqual({record["label_name"] for record in records}, {"support", "contradiction", "abstain"})
            self.assertTrue(all(record["prefix_text"] for record in records))
            support_record = next(record for record in records if record["label_name"] == "support")
            self.assertIn(support_record["candidate_text"], {"cat", "dog"})
            self.assertEqual(len(support_record["region_boxes"]), 1)
            self.assertEqual(support_record["caption_text"], "a cat and a dog on the floor")


if __name__ == "__main__":
    unittest.main()
