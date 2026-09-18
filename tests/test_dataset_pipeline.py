"""
Unit tests for the backend dataset pipeline and data loader (dataset_pipeline.py).
Verifies CSV loading, feature engineering, 80/20 split, model evaluation, and anomaly coordinate extraction.
"""

import unittest
from pathlib import Path
from dataset_pipeline import (
    generate_hackathon_sample_files,
    AISDataLoader,
    SARImageMaskDataLoader,
    BackendAnomalyInferenceEngine
)

class TestDatasetPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.paths = generate_hackathon_sample_files(output_dir="dataset_storage_test", num_ais_records=300, num_sar_samples=20)

    def test_01_load_and_engineer_features(self):
        df = AISDataLoader.load_telemetry_csv(self.paths["ais_csv"])
        self.assertGreater(len(df), 0)
        self.assertIn("mmsi", df.columns)
        self.assertIn("lat", df.columns)
        self.assertIn("lon", df.columns)

        feat_df = AISDataLoader.engineer_kinematic_features(df)
        for col in AISDataLoader.FEATURE_COLS:
            self.assertIn(col, feat_df.columns)

    def test_02_train_val_split_80_20(self):
        df = AISDataLoader.load_telemetry_csv(self.paths["ais_csv"])
        feat_df = AISDataLoader.engineer_kinematic_features(df)
        split = AISDataLoader.get_train_val_split(feat_df, test_size=0.20, random_state=42)

        total = len(feat_df)
        train_len = len(split["train_df"])
        val_len = len(split["val_df"])

        self.assertEqual(train_len + val_len, total)
        self.assertAlmostEqual(val_len / total, 0.20, delta=0.02)
        self.assertEqual(split["X_train"].shape[1], len(AISDataLoader.FEATURE_COLS))

    def test_03_backend_inference_and_anomaly_coordinates(self):
        df = AISDataLoader.load_telemetry_csv(self.paths["ais_csv"])
        feat_df = AISDataLoader.engineer_kinematic_features(df)
        split = AISDataLoader.get_train_val_split(feat_df, test_size=0.20, random_state=42)

        engine = BackendAnomalyInferenceEngine(model_type="random_forest")
        results = engine.train_and_evaluate(split)
        self.assertIn("validation_f1_score", results)
        self.assertGreaterEqual(results["validation_f1_score"], 0.85)

        anomaly_points = engine.predict_anomaly_coordinates(split["val_df"])
        self.assertIsInstance(anomaly_points, list)
        if len(anomaly_points) > 0:
            point = anomaly_points[0]
            self.assertIn("lat", point)
            self.assertIn("lon", point)
            self.assertIn("mmsi", point)
            self.assertIn("anomaly_confidence", point)

    def test_04_sar_manifest_split(self):
        sar_split = SARImageMaskDataLoader.load_manifest_and_split(self.paths["sar_manifest"], test_size=0.20)
        self.assertEqual(sar_split["total_samples"], 20)
        self.assertEqual(sar_split["train_samples"], 16)
        self.assertEqual(sar_split["val_samples"], 4)

if __name__ == "__main__":
    unittest.main()
