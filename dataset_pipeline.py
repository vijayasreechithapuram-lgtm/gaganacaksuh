"""
=============================================================================
Project Gaganacakṣuḥ: Machine Learning Dataset Pipeline & Data Loader
SIH Problem Statement: SIH26143 (Disaster Management)
=============================================================================
Completely independent of the UI/Streamlit layer.
Loads and structures hackathon datasets (AIS telemetry CSVs and SAR chips/masks),
performs 80/20 train/validation splits with scikit-learn, and extracts anomaly coordinates.
"""

import os
import sys
import math
import json
from typing import Dict, Any, Tuple, Optional, List, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

# Optional PyTorch Integration (fallback gracefully if not installed)
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
    HAS_TORCH = True
except ImportError:
    torch = None
    TorchDataset = object
    TorchDataLoader = None
    HAS_TORCH = False


# =============================================================================
# 1. HACKATHON SAMPLE DATA GENERATOR (Provides training files if not on disk)
# =============================================================================
def generate_hackathon_sample_files(
    output_dir: str = "dataset_storage",
    num_ais_records: int = 1200,
    num_sar_samples: int = 100,
    seed: int = 42
) -> Dict[str, str]:
    """
    Generates standard training files conforming to ITU-R M.1371 (Marine Cadastre AIS)
    and Zenodo SAR oil spill benchmarks for training and offline validation.
    """
    out_path = Path(output_dir)
    ais_dir = out_path / "ais"
    sar_dir = out_path / "sar_chips"
    masks_dir = out_path / "masks"
    
    ais_dir.mkdir(parents=True, exist_ok=True)
    sar_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    # --- 1. Generate AIS Telemetry CSV ---
    ais_file = ais_dir / "marine_cadastre_ais_training.csv"
    
    vessels = [
        {"mmsi": "419088421", "name": "MT OCEAN PRIDE", "type": "Tanker - Crude", "base_speed": 12.0, "is_rogue": True},
        {"mmsi": "419001982", "name": "MV PHANTOM STAR", "type": "Chemical Tanker", "base_speed": 13.0, "is_rogue": True},
        {"mmsi": "419000542", "name": "MV BHARAT RATNA", "type": "Bulk Carrier", "base_speed": 11.5, "is_rogue": False},
        {"mmsi": "228394000", "name": "CMA CGM INDUS", "type": "Container Ship", "base_speed": 18.0, "is_rogue": False},
        {"mmsi": "419077123", "name": "INS SARVASHAKTI", "type": "Patrol Vessel", "base_speed": 22.0, "is_rogue": False},
        {"mmsi": "419099876", "name": "MT GANGA PIONEER", "type": "LPG Carrier", "base_speed": 14.0, "is_rogue": False}
    ]

    records = []
    base_time = pd.Timestamp("2026-09-04 00:00:00")

    records_per_vessel = num_ais_records // len(vessels)

    for v in vessels:
        lat = 19.0 + np.random.uniform(-0.4, 0.4)
        lon = 71.0 + np.random.uniform(-0.4, 0.4)
        heading = np.random.uniform(40, 75)
        
        for i in range(records_per_vessel):
            curr_time = base_time + pd.Timedelta(minutes=i * 5)
            sog = v["base_speed"] + np.random.normal(0, 0.4)
            is_anomaly = 0
            anomaly_type = "NORMAL"

            # Inject realistic adversarial anomalies for rogue vessels:
            # 1. GPS Spoofing: Sudden velocity spike > 35 knots
            # 2. AIS Suppression / Dark ship gap near oil fairway
            if v["is_rogue"] and v["mmsi"] == "419001982" and (records_per_vessel * 0.4 < i < records_per_vessel * 0.6):
                sog = 44.5 + np.random.normal(0, 1.2) # High speed spoofing!
                lat += np.random.normal(0.08, 0.02)
                lon += np.random.normal(0.08, 0.02)
                is_anomaly = 1
                anomaly_type = "GPS_SPOOFING_VELOCITY_VIOLATION"
            elif v["is_rogue"] and v["mmsi"] == "419088421" and (records_per_vessel * 0.5 < i < records_per_vessel * 0.7):
                sog = 3.2 + np.random.normal(0, 0.2) # Slow discharge drift
                is_anomaly = 1
                anomaly_type = "ILLEGAL_DISCHARGE_LOITERING"
            else:
                # Normal motion
                rad = math.radians(heading)
                lat += (sog * 0.514444 * 300) / 111132.0 * math.cos(rad)
                lon += (sog * 0.514444 * 300) / (111412.0 * math.cos(math.radians(lat))) * math.sin(rad)

            records.append({
                "mmsi": v["mmsi"],
                "vessel_name": v["name"],
                "vessel_type": v["type"],
                "base_datetime": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "sog": round(max(0.0, sog), 2),
                "cog": round((heading + np.random.normal(0, 2)) % 360, 1),
                "heading": round(heading, 1),
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_type
            })

    ais_df = pd.DataFrame(records)
    ais_df.to_csv(ais_file, index=False)

    # --- 2. Generate SAR Chips & Masks Manifest ---
    manifest_file = out_path / "zenodo_sar_manifest.csv"
    sar_manifest = []

    for idx in range(num_sar_samples):
        chip_name = f"sar_chip_{idx:04d}.npy"
        mask_name = f"mask_{idx:04d}.npy"
        chip_path = sar_dir / chip_name
        mask_path = masks_dir / mask_name

        has_slick = int(np.random.rand() > 0.4) # 60% positive samples
        
        # 128x128 SAR amplitude chip (Gamma noise)
        chip = np.random.gamma(9.0, 13.5, (128, 128)).astype(np.float32)
        mask = np.zeros((128, 128), dtype=np.uint8)

        if has_slick:
            # Elliptical oil slick dark depression
            cx, cy = np.random.randint(30, 98), np.random.randint(30, 98)
            ax, ay = np.random.randint(12, 35), np.random.randint(6, 18)
            y, x = np.ogrid[:128, :128]
            slick_area = ((x - cx)**2 / ax**2) + ((y - cy)**2 / ay**2) <= 1.0
            mask[slick_area] = 1
            chip[slick_area] = np.random.gamma(3.5, 8.0, np.sum(slick_area))

        np.save(chip_path, chip)
        np.save(mask_path, mask)

        sar_manifest.append({
            "sample_id": f"ZEN-SAR-{idx:04d}",
            "chip_path": str(chip_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "has_slick": has_slick,
            "slick_pixel_count": int(np.sum(mask)),
            "slick_sq_km": round(float(np.sum(mask) * (0.01**2)), 4),
            "region": "Mumbai High Arabian Sea"
        })

    sar_df = pd.DataFrame(sar_manifest)
    sar_df.to_csv(manifest_file, index=False)

    return {
        "ais_csv": str(ais_file.resolve()),
        "sar_manifest": str(manifest_file.resolve()),
        "output_dir": str(out_path.resolve())
    }


# =============================================================================
# 2. AIS TELEMETRY DATA LOADER & FEATURE ENGINEERING
# =============================================================================
class AISDataLoader:
    """
    Loads, cleans, engineers kinematic features, and splits Marine Cadastre / ITU-R M.1371
    AIS telemetry datasets for anomaly detection and GPS anti-spoofing models.
    """

    FEATURE_COLS = [
        "sog", "cog", "speed_discrepancy", "acceleration_kts_per_min",
        "distance_delta_km", "heading_diff", "sog_rolling_mean_3", "sog_rolling_std_3"
    ]

    @staticmethod
    def haversine_distance_vectorized(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """Vectorized Haversine distance in kilometers."""
        r = 6371.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dp = np.radians(lat2 - lat1)
        dl = np.radians(lon2 - lon1)
        a = np.sin(dp / 2.0)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0)**2
        return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    @classmethod
    def load_telemetry_csv(cls, filepath: Union[str, Path]) -> pd.DataFrame:
        """Reads and validates standard hackathon AIS telemetry CSV."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"AIS CSV not found at: {path}")

        df = pd.read_csv(path)
        # Normalize column names to lowercase
        df.columns = [c.strip().lower() for c in df.columns]

        # Require standard maritime fields
        required = ["mmsi", "base_datetime", "lat", "lon", "sog"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required AIS fields in CSV: {missing}")

        # Type conversions
        df["base_datetime"] = pd.to_datetime(df["base_datetime"])
        df["mmsi"] = df["mmsi"].astype(str)
        df["lat"] = df["lat"].astype(float)
        df["lon"] = df["lon"].astype(float)
        df["sog"] = df["sog"].astype(float)
        if "cog" not in df.columns:
            df["cog"] = 0.0
        if "heading" not in df.columns:
            df["heading"] = df["cog"]

        df = df.sort_values(by=["mmsi", "base_datetime"]).reset_index(drop=True)
        return df

    @classmethod
    def engineer_kinematic_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates physical movement derivatives (speed over ground vs coordinate delta,
        acceleration, heading delta) to empower machine learning anti-spoofing classifiers.
        """
        df = df.copy()

        # Group-wise shift for previous timestamp and coordinate
        prev_lat = df.groupby("mmsi")["lat"].shift(1).fillna(df["lat"])
        prev_lon = df.groupby("mmsi")["lon"].shift(1).fillna(df["lon"])
        prev_time = df.groupby("mmsi")["base_datetime"].shift(1).fillna(df["base_datetime"])
        prev_sog = df.groupby("mmsi")["sog"].shift(1).fillna(df["sog"])

        # Spatial distance and elapsed time
        dist_km = cls.haversine_distance_vectorized(prev_lat.values, prev_lon.values, df["lat"].values, df["lon"].values)
        time_delta_min = (df["base_datetime"] - prev_time).dt.total_seconds() / 60.0
        time_delta_min = np.clip(time_delta_min.values, 0.01, 1440.0)

        # Derived physical speed (knots) = (dist_km / 1.852) / (time_delta_hrs)
        derived_sog_knots = (dist_km / 1.852) / (time_delta_min / 60.0)
        derived_sog_knots = np.nan_to_num(derived_sog_knots, nan=0.0, posinf=100.0, neginf=0.0)

        # Physical discrepancy between reported radio SOG and actual GPS coordinate shift
        speed_discrepancy = np.abs(df["sog"].values - derived_sog_knots)

        # Acceleration in knots/min
        accel = np.abs(df["sog"].values - prev_sog.values) / time_delta_min

        # Heading difference
        heading_diff = np.abs(df["cog"].values - df["heading"].values)
        heading_diff = np.minimum(heading_diff, 360.0 - heading_diff)

        df["distance_delta_km"] = np.round(dist_km, 4)
        df["derived_sog_knots"] = np.round(derived_sog_knots, 2)
        df["speed_discrepancy"] = np.round(speed_discrepancy, 2)
        df["acceleration_kts_per_min"] = np.round(accel, 3)
        df["heading_diff"] = np.round(heading_diff, 1)

        # Rolling statistics per vessel
        df["sog_rolling_mean_3"] = df.groupby("mmsi")["sog"].transform(lambda s: s.rolling(3, min_periods=1).mean())
        df["sog_rolling_std_3"] = df.groupby("mmsi")["sog"].transform(lambda s: s.rolling(3, min_periods=1).std().fillna(0.0))

        return df

    @classmethod
    def get_train_val_split(
        cls,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Splits dataset into 80% Training and 20% Local Validation sets.
        Performs stratified splitting if labels exist, preserving anomaly proportions.
        """
        has_labels = "is_anomaly" in df.columns
        stratify = df["is_anomaly"] if has_labels else None

        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[cls.FEATURE_COLS].values)
        X_val = scaler.transform(val_df[cls.FEATURE_COLS].values)

        y_train = train_df["is_anomaly"].values if has_labels else None
        y_val = val_df["is_anomaly"].values if has_labels else None

        return {
            "train_df": train_df,
            "val_df": val_df,
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
            "scaler": scaler,
            "feature_names": cls.FEATURE_COLS,
            "split_ratio": f"{int((1 - test_size)*100)}/{int(test_size*100)}"
        }


# =============================================================================
# 3. SAR IMAGERY & MASK DATASET LOADER (Zenodo Format)
# =============================================================================
class SARImageMaskDataLoader:
    """
    Loads Zenodo Sentinel-1 SAR chips and corresponding binary ground-truth masks.
    Splits into 80/20 train/validation splits.
    """

    @classmethod
    def load_manifest_and_split(
        cls,
        manifest_csv: Union[str, Path],
        test_size: float = 0.20,
        random_state: int = 42
    ) -> Dict[str, Any]:
        path = Path(manifest_csv)
        if not path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {path}")

        df = pd.read_csv(path)
        stratify = df["has_slick"] if "has_slick" in df.columns else None

        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        return {
            "train_manifest": train_df.reset_index(drop=True),
            "val_manifest": val_df.reset_index(drop=True),
            "total_samples": len(df),
            "train_samples": len(train_df),
            "val_samples": len(val_df)
        }

    @staticmethod
    def load_chip_and_mask(chip_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads a single SAR amplitude chip and binary mask from disk."""
        chip = np.load(chip_path)
        mask = np.load(mask_path)
        return chip, mask


# =============================================================================
# 4. OPTIONAL PYTORCH DATASET WRAPPER (When PyTorch is installed)
# =============================================================================
if HAS_TORCH:
    class AISTorchDataset(TorchDataset):
        def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            if self.y is not None:
                return self.X[idx], self.y[idx]
            return self.X[idx]

    class SARTorchDataset(TorchDataset):
        def __init__(self, manifest_df: pd.DataFrame):
            self.df = manifest_df

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            chip, mask = SARImageMaskDataLoader.load_chip_and_mask(row["chip_path"], row["mask_path"])
            # Normalize chip
            chip_norm = (chip - np.mean(chip)) / (np.std(chip) + 1e-6)
            chip_t = torch.tensor(chip_norm, dtype=torch.float32).unsqueeze(0) # (1, H, W)
            mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) # (1, H, W)
            return chip_t, mask_t


# =============================================================================
# 5. BACKEND ANOMALY INFERENCE PIPELINE (Completely UI-independent)
# =============================================================================
class BackendAnomalyInferenceEngine:
    """
    Pure backend inference engine:
    Trains or loads an anomaly detector on the training set, evaluates on the 20% validation set,
    and extracts clean anomaly coordinates (lat, lon, timestamp, MMSI, anomaly score).
    """

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        if model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        else:
            self.model = IsolationForest(contamination=0.10, random_state=42)
        self.scaler = None
        self.is_trained = False

    def train_and_evaluate(self, train_val_split: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trains model on 80% training set and computes evaluation metrics on 20% validation set.
        """
        X_train, y_train = train_val_split["X_train"], train_val_split["y_train"]
        X_val, y_val = train_val_split["X_val"], train_val_split["y_val"]
        self.scaler = train_val_split["scaler"]

        if y_train is not None and self.model_type == "random_forest":
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            y_proba = self.model.predict_proba(X_val)[:, 1]
            f1 = f1_score(y_val, y_pred)
            roc_auc = roc_auc_score(y_val, y_proba)
            report = classification_report(y_val, y_pred, output_dict=True)
            conf_mat = confusion_matrix(y_val, y_pred).tolist()
        else:
            self.model.fit(X_train)
            # -1: anomaly, 1: normal
            raw_pred = self.model.predict(X_val)
            y_pred = np.where(raw_pred == -1, 1, 0)
            f1 = f1_score(y_val, y_pred) if y_val is not None else 0.0
            roc_auc = 0.0
            report = classification_report(y_val, y_pred, output_dict=True) if y_val is not None else {}
            conf_mat = confusion_matrix(y_val, y_pred).tolist() if y_val is not None else []

        self.is_trained = True

        return {
            "validation_f1_score": round(float(f1), 4),
            "validation_roc_auc": round(float(roc_auc), 4),
            "confusion_matrix": conf_mat,
            "classification_report": report,
            "val_samples_tested": len(X_val)
        }

    def predict_anomaly_coordinates(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Infers anomalies across any input AIS dataframe and extracts clean geographic coordinates.
        Output is pure JSON / Python dicts for clean backend consumption.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before running inference.")

        # Ensure features are computed
        if "speed_discrepancy" not in df.columns:
            feat_df = AISDataLoader.engineer_kinematic_features(df)
        else:
            feat_df = df.copy()

        X = self.scaler.transform(feat_df[AISDataLoader.FEATURE_COLS].values)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
            preds = (probs >= 0.50).astype(int)
        else:
            raw = self.model.predict(X)
            preds = np.where(raw == -1, 1, 0)
            probs = -self.model.decision_function(X) # Higher score = more anomalous

        feat_df["predicted_anomaly"] = preds
        feat_df["anomaly_confidence"] = np.round(probs, 4)

        # Extract only detected anomaly points
        anomalies = feat_df[feat_df["predicted_anomaly"] == 1]

        anomaly_points = []
        for _, row in anomalies.iterrows():
            anomaly_points.append({
                "mmsi": str(row["mmsi"]),
                "vessel_name": row.get("vessel_name", "UNKNOWN"),
                "timestamp": str(row["base_datetime"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "sog": float(row["sog"]),
                "anomaly_confidence": float(row["anomaly_confidence"]),
                "detected_violation": (
                    "KINEMATIC_GPS_SPOOFING (>35 kts)" if row["sog"] > 35.0
                    else "ANOMALOUS_LOITERING / DISCHARGE PATTERN"
                )
            })

        return anomaly_points


# =============================================================================
# 6. STANDALONE CLI & PIPELINE DEMONSTRATION
# =============================================================================
def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 75)
    print("  PROJECT GAGANACAKSUH: ML DATASET LOADER & ANOMALY PIPELINE (SIH26143)")
    print("=" * 75)

    # 1. Generate or verify sample training files
    print("\n[STEP 1] Generating / Verifying Hackathon Training Datasets...")
    paths = generate_hackathon_sample_files(output_dir="dataset_storage")
    print(f" -> AIS Telemetry File:   {paths['ais_csv']}")
    print(f" -> SAR Manifest File:    {paths['sar_manifest']}")

    # 2. Load and feature engineer AIS telemetry
    print("\n[STEP 2] Loading AIS Telemetry & Engineering Kinematic Features...")
    raw_ais_df = AISDataLoader.load_telemetry_csv(paths["ais_csv"])
    engineered_df = AISDataLoader.engineer_kinematic_features(raw_ais_df)
    print(f" -> Loaded {len(engineered_df)} records across {engineered_df['mmsi'].nunique()} vessels.")
    print(f" -> Computed kinematic features: {AISDataLoader.FEATURE_COLS}")

    # 3. 80/20 Train/Validation Split
    print("\n[STEP 3] Executing 80/20 Train-Validation Split (Stratified)...")
    split_data = AISDataLoader.get_train_val_split(engineered_df, test_size=0.20, random_state=42)
    print(f" -> Train Set Size: {len(split_data['train_df'])} samples (80%)")
    print(f" -> Val Set Size:   {len(split_data['val_df'])} samples (20%)")
    print(f" -> Anomaly balance in Train: {np.mean(split_data['y_train'])*100:.1f}%")
    print(f" -> Anomaly balance in Val:   {np.mean(split_data['y_val'])*100:.1f}%")

    # 4. Train and Evaluate Backend Model on 20% Validation Set
    print("\n[STEP 4] Evaluating Model on 20% Local Validation Set...")
    inference_engine = BackendAnomalyInferenceEngine(model_type="random_forest")
    eval_results = inference_engine.train_and_evaluate(split_data)
    print(f" -> Validation F1-Score: {eval_results['validation_f1_score']:.4f}")
    print(f" -> Validation ROC-AUC:  {eval_results['validation_roc_auc']:.4f}")
    print(f" -> Confusion Matrix (TN, FP / FN, TP): {eval_results['confusion_matrix']}")

    # 5. Extract Anomaly Coordinates for Clean Backend Consumption
    print("\n[STEP 5] Extracting Clean Anomaly Coordinates from Validation Set...")
    anomaly_coords = inference_engine.predict_anomaly_coordinates(split_data["val_df"])
    print(f" -> Total Anomalies Detected in Validation Set: {len(anomaly_coords)}")
    
    # Print sample detected coordinates
    print("\nSample Extracted Anomaly Coordinates (Pure Backend JSON Format):")
    print("-" * 75)
    for sample in anomaly_coords[:5]:
        print(json.dumps(sample, indent=2))
    print("-" * 75)

    # 6. SAR Chips Manifest Split Verification
    print("\n[STEP 6] Splitting Zenodo SAR Chips Manifest (80/20)...")
    sar_split = SARImageMaskDataLoader.load_manifest_and_split(paths["sar_manifest"], test_size=0.20)
    print(f" -> Total SAR Chips: {sar_split['total_samples']}")
    print(f" -> Train SAR Chips: {sar_split['train_samples']} (80%)")
    print(f" -> Val SAR Chips:   {sar_split['val_samples']} (20%)")

    print("\n" + "=" * 75)
    print("  PIPELINE EXECUTION COMPLETE & READY FOR BACKEND MODEL INGESTION")
    print("=" * 75)

if __name__ == "__main__":
    main()
