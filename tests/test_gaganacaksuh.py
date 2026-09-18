"""
Comprehensive Unit Test Suite for Project Gaganacakṣuḥ Core Engines
Tests Cryptography, RBAC, WMS Simulation, U-Net Segmentation, Hydrodynamics,
Zero-Trust AIS Spoofing Detection, ISO/IEC 27037 Ledger, and Rerouting.
"""

import unittest
import numpy as np
from core.crypto_auth import (
    authenticate_user,
    verify_session_token,
    ROLE_COMMAND_OFFICER,
    ROLE_TRIBUNAL_JUDGE
)
from core.satellite_wms import generate_synthetic_sar_scene, get_wms_capabilities_mock
from core.unet_segmentation import UNetZenodoInferenceEngine
from core.hydrodynamics import HydrodynamicBackcastEngine
from core.zero_trust_ais import ZeroTrustAISAuditor
from core.forensic_ledger import ForensicLedgerEngine
from core.rerouting_engine import TacticalReroutingEngine
from data.zenodo_sar_samples import load_benchmark_sar_scene

class TestGaganacaksuhCore(unittest.TestCase):

    def test_01_authentication_and_token_generation(self):
        # 1. Test Command Officer Login
        officer = authenticate_user("officer.icg", "CoastGuard@2026")
        self.assertIsNotNone(officer)
        self.assertEqual(officer["role"], ROLE_COMMAND_OFFICER)
        token = officer["session_token"]
        self.assertTrue(token.startswith("GGN."))
        
        # Verify Token
        claims = verify_session_token(token)
        self.assertIsNotNone(claims)
        self.assertEqual(claims["user"], "officer.icg")
        self.assertEqual(claims["role"], ROLE_COMMAND_OFFICER)

        # 2. Test Tribunal Judge Login
        judge = authenticate_user("judge.tribunal", "Justice@Maritime2026")
        self.assertIsNotNone(judge)
        self.assertEqual(judge["role"], ROLE_TRIBUNAL_JUDGE)

        # 3. Test Invalid Credentials
        invalid = authenticate_user("officer.icg", "WrongPassword123")
        self.assertIsNone(invalid)

    def test_02_satellite_wms_generation(self):
        scene = generate_synthetic_sar_scene(width=128, height=128, seed=42)
        self.assertEqual(scene["sar_amplitude"].shape, (128, 128))
        self.assertEqual(scene["ground_truth_mask"].shape, (128, 128))
        self.assertGreater(scene["spill_pixel_area"], 0)
        self.assertGreater(scene["spill_sq_km"], 0)

        wms_cap = get_wms_capabilities_mock("mumbai_high")
        self.assertIn("Copernicus", wms_cap["service"])
        self.assertEqual(len(wms_cap["active_layers"]), 4)

    def test_03_unet_segmentation_inference(self):
        scene = load_benchmark_sar_scene("mumbai_high")
        unet = UNetZenodoInferenceEngine(confidence_threshold=0.5)
        
        results = unet.predict_segmentation(
            sar_amplitude=scene["sar_amplitude"],
            ndvi_optical=scene["ndvi"],
            origin_lat=scene["metadata"]["center_lat"],
            origin_lon=scene["metadata"]["center_lon"]
        )
        
        self.assertGreaterEqual(results["mean_confidence_score"], 0.94)
        self.assertGreater(results["total_area_sq_km"], 0.1)
        self.assertGreater(results["contours_detected"], 0)
        self.assertEqual(len(results["centroid"]), 2)
        self.assertIn("weights_hash", results["model_metadata"])

    def test_04_hydrodynamic_backcasting(self):
        engine = HydrodynamicBackcastEngine(
            current_u_ms=0.30,
            current_v_ms=-0.15,
            wind_speed_ms=7.0,
            wind_dir_deg=240.0
        )
        slick_centroid = (19.4167, 71.3333)
        backcast = engine.simulate_lagrangian_backcast(
            slick_centroid=slick_centroid,
            backcast_hours=10.0,
            num_particles=200,
            seed=42
        )
        
        self.assertIn("origin_release_coord", backcast)
        self.assertEqual(len(backcast["origin_polygon"]), 5)
        self.assertGreater(backcast["origin_uncertainty_meters"], 50.0)
        self.assertEqual(len(backcast["trajectory_steps"]), 9)

        # Fay Age estimation test
        fay = engine.estimate_fay_spill_age(area_sq_km=1.85, estimated_volume_bbl=400)
        self.assertGreater(fay["estimated_age_hours"], 2.0)
        self.assertIn("Phase", fay["fay_regime"])

    def test_05_zero_trust_ais_and_dark_ships(self):
        auditor = ZeroTrustAISAuditor(max_speed_threshold_knots=35.0)
        origin_coord = (19.348, 71.240)
        
        audit = auditor.audit_vessel_records(origin_coord)
        
        self.assertGreater(audit["total_vessels_tracked"], 0)
        self.assertGreater(audit["spoofed_vessels_detected"], 0)
        self.assertGreater(audit["dark_vessels_detected"], 0)
        self.assertIsNotNone(audit["primary_suspect"])
        
        # Check that MT OCEAN PRIDE is ranked high attribution
        suspect = audit["primary_suspect"]
        self.assertEqual(suspect["vessel_name"], "MT OCEAN PRIDE")
        self.assertGreater(suspect["attribution_score"], 90.0)
        
        # Check alerts
        alert_codes = [a["code"] for a in audit["security_alerts"]]
        self.assertIn("AIS_KINEMATIC_ANOMALY", alert_codes)
        self.assertIn("AIS_TRANSPONDER_SUPPRESSION", alert_codes)
        self.assertIn("SAR_RADAR_HULL_NON_REPORTING", alert_codes)

    def test_06_forensic_ledger_tamper_detection_and_pdf(self):
        ledger = ForensicLedgerEngine()
        
        docket = ledger.build_evidence_docket(
            docket_id="DOCKET-2026-MH-0042",
            case_officer="Cmdr. Rajesh Verma",
            service_number="ICG-WEST-7741",
            jurisdiction="Western Naval Command / Mumbai Admiralty Bench",
            satellite_metadata={"orbit": "Descending #117", "timestamp": "2026-09-04T06:14:22Z"},
            spill_metrics={"centroid": [19.4167, 71.3333], "total_area_sq_km": 1.84, "mean_confidence_score": 0.9482},
            backcast_data={"origin_release_coord": [19.348, 71.240], "origin_uncertainty_meters": 350.0},
            suspect_data={"mmsi": "419088421", "vessel_name": "MT OCEAN PRIDE", "attribution_score": 96.4}
        )

        self.assertIn("master_seal_hash", docket)
        self.assertEqual(len(docket["master_seal_hash"]), 64)

        # 1. Test clean verification
        clean_check = ledger.verify_tamper_integrity(docket)
        self.assertTrue(clean_check["seal_intact"])
        self.assertEqual(clean_check["iso_27037_verdict"], "VALID_PROBATIVE_VALUE")

        # 2. Test tampering simulation (adversary alters 1 character in suspect MMSI)
        tamper_check = ledger.verify_tamper_integrity(
            docket,
            tampered_key="suspect_attribution_hash",
            tampered_value="0000000000000000000000000000000000000000000000000000000000000000"
        )
        self.assertFalse(tamper_check["seal_intact"])
        self.assertEqual(tamper_check["iso_27037_verdict"], "INVALIDATED_INTEGRITY_BREACH")

        # 3. Test PDF generation
        pdf_bytes = ledger.generate_court_dossier_pdf(docket)
        self.assertGreater(len(pdf_bytes), 2000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_07_tactical_rerouting_engine(self):
        rerouter = TacticalReroutingEngine(buffer_radius_km=5.0)
        slick_centroid = (19.4167, 71.3333)
        
        exclusion = rerouter.generate_exclusion_zone(slick_centroid, [])
        self.assertIn("exclusion_polygon", exclusion)
        self.assertEqual(len(exclusion["exclusion_polygon"]), 5)

        test_vessels = [
            {"mmsi": "419000542", "vessel_name": "MV BHARAT RATNA", "last_known_pos": [19.45, 71.55]}
        ]
        reroutes = rerouter.compute_vessel_reroutes(test_vessels, exclusion)
        self.assertEqual(len(reroutes), 1)
        self.assertIn("diversion_waypoint", reroutes[0])

        dispatch = rerouter.dispatch_vhf_navtex_broadcast("DOCKET-001", exclusion, reroutes)
        self.assertIn("NAVTEX", dispatch["navtex_broadcast"])
        self.assertIn("@Gaganacaksuh", dispatch["telegram_dispatch"]["channel"])

if __name__ == "__main__":
    unittest.main()
