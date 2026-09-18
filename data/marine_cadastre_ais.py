"""
Project Gaganacakṣuḥ: Marine Cadastre AIS Telemetry Dataset Generator
Provides standardized ITU-R M.1371 schemas with verified normal traffic,
adversarial GPS spoofing maneuvers (>35 knots), and dark vessel non-reporting profiles.
"""

from typing import List, Dict, Any

def get_marine_cadastre_ais_records() -> List[Dict[str, Any]]:
    """
    Returns high-fidelity AIS records representing vessel tracks across the
    Mumbai High tanker fairway corridor over the last 18 hours.
    Includes:
      - 1 Suspect Dark Vessel (turned off AIS near origin point, physical radar detection only)
      - 1 GPS Spoofing Vessel (MMSI 419001982: reports 43.8 knots to fake location)
      - 3 Legitimate commercial vessels (Crude Oil Tanker, Bulk Carrier, Container Ship)
    """
    records = [
        # --- VESSEL 1: SUSPECT OIL TANKER "MT OCEAN PRIDE" (MMSI: 419088421) ---
        # Discharges bilge/sludge, deliberately switches off AIS for 4 hours at origin
        {
            "mmsi": "419088421",
            "vessel_name": "MT OCEAN PRIDE",
            "vessel_type": "Tanker - Crude Oil",
            "flag": "Panama (PA)",
            "imo": "9384421",
            "callsign": "3EWP8",
            "length_m": 248.0,
            "beam_m": 42.0,
            "draft_m": 14.8,
            "telemetry": [
                {"timestamp": "2026-09-04T00:00:00Z", "lat": 19.1200, "lon": 70.8500, "sog": 12.4, "cog": 65.0, "status": "Under way using engine"},
                {"timestamp": "2026-09-04T01:30:00Z", "lat": 19.2350, "lon": 71.0500, "sog": 11.8, "cog": 62.0, "status": "Under way using engine"},
                {"timestamp": "2026-09-04T03:00:00Z", "lat": 19.3480, "lon": 71.2400, "sog": 6.2, "cog": 58.0, "status": "Under way using engine"}, # Decelerating near release origin
                # AIS GOES DARK BETWEEN 03:30Z and 07:30Z (Illegal Discharge Window)
                {"timestamp": "2026-09-04T07:45:00Z", "lat": 19.5200, "lon": 71.5500, "sog": 13.1, "cog": 70.0, "status": "Under way using engine"},
                {"timestamp": "2026-09-04T09:00:00Z", "lat": 19.6400, "lon": 71.7800, "sog": 13.5, "cog": 68.0, "status": "Under way using engine"}
            ],
            "last_known_pos": [19.6400, 71.7800],
            "has_ais_gap": True,
            "gap_start": "2026-09-04T03:15:00Z",
            "gap_end": "2026-09-04T07:35:00Z",
            "gap_duration_hrs": 4.33
        },

        # --- VESSEL 2: GPS SPOOFER "MV PHANTOM STAR" (MMSI: 419001982) ---
        # Injects fake NMEA coordinates showing 44.5 knots across 80 nautical miles
        {
            "mmsi": "419001982",
            "vessel_name": "MV PHANTOM STAR",
            "vessel_type": "Chemical Tanker",
            "flag": "Liberia (LR)",
            "imo": "9421182",
            "callsign": "A8TQ2",
            "length_m": 182.0,
            "beam_m": 28.0,
            "draft_m": 9.5,
            "telemetry": [
                {"timestamp": "2026-09-04T02:00:00Z", "lat": 18.9500, "lon": 71.1000, "sog": 13.2, "cog": 45.0, "status": "Under way using engine"},
                {"timestamp": "2026-09-04T03:00:00Z", "lat": 19.1000, "lon": 71.2500, "sog": 13.0, "cog": 44.0, "status": "Under way using engine"},
                # SPOOFING TRIGGER: Jumps from 19.10 to 19.82 in 60 mins -> 44.5 knots SOG!
                {"timestamp": "2026-09-04T04:00:00Z", "lat": 19.8200, "lon": 71.9500, "sog": 44.5, "cog": 110.0, "status": "Under way using engine"},
                {"timestamp": "2026-09-04T05:30:00Z", "lat": 19.9500, "lon": 72.1000, "sog": 41.8, "cog": 105.0, "status": "Under way using engine"}
            ],
            "last_known_pos": [19.9500, 72.1000],
            "has_ais_gap": False
        },

        # --- VESSEL 3: LEGITIMATE BULK CARRIER "MV BHARAT RATNA" (MMSI: 419000542) ---
        {
            "mmsi": "419000542",
            "vessel_name": "MV BHARAT RATNA",
            "vessel_type": "Bulk Carrier",
            "flag": "India (IN)",
            "imo": "9550542",
            "callsign": "AWVY",
            "length_m": 225.0,
            "beam_m": 32.2,
            "draft_m": 12.0,
            "telemetry": [
                {"timestamp": "2026-09-04T01:00:00Z", "lat": 19.8000, "lon": 70.9000, "sog": 11.2, "cog": 120.0, "status": "Under way using engine"},
                {"timestamp": "2026-09-04T03:00:00Z", "lat": 19.6200, "lon": 71.2200, "sog": 11.4, "cog": 122.0, "status": "Under way using engine"},
                {"timestamp": "2026-09-04T05:00:00Z", "lat": 19.4500, "lon": 71.5500, "sog": 11.1, "cog": 120.0, "status": "Under way using engine"},
                {"timestamp": "2026-09-04T07:00:00Z", "lat": 19.2800, "lon": 71.8600, "sog": 11.3, "cog": 121.0, "status": "Under way using engine"}
            ],
            "last_known_pos": [19.2800, 71.8600],
            "has_ais_gap": False
        },

        # --- VESSEL 4: CONTAINER SHIP "CMA CGM INDUS" (MMSI: 228394000) ---
        {
            "mmsi": "228394000",
            "vessel_name": "CMA CGM INDUS",
            "vessel_type": "Cargo / Container Ship",
            "flag": "France (FR)",
            "imo": "9722650",
            "callsign": "FMGI",
            "length_m": 300.0,
            "beam_m": 48.0,
            "draft_m": 13.5,
            "telemetry": [
                {"timestamp": "2026-09-04T02:00:00Z", "lat": 19.0500, "lon": 71.7000, "sog": 18.2, "cog": 330.0, "status": "Under way using engine"},
                {"timestamp": "2026-09-04T04:00:00Z", "lat": 19.3800, "lon": 71.5000, "sog": 18.0, "cog": 330.0, "status": "Under way using engine"},
                {"timestamp": "2026-09-04T06:00:00Z", "lat": 19.7000, "lon": 71.3000, "sog": 18.1, "cog": 332.0, "status": "Under way using engine"}
            ],
            "last_known_pos": [19.7000, 71.3000],
            "has_ais_gap": False
        }
    ]
    return records

def get_sar_radar_hull_detections() -> List[Dict[str, Any]]:
    """
    Physical radar cross-section (RCS) targets extracted from Sentinel-1 SAR pass.
    Reflects bright dihedral returns from steel hulls.
    """
    return [
        {
            "detection_id": "SAR-RADAR-TGT-001",
            "lat": 19.3620,
            "lon": 71.2650,
            "rcs_db": 42.5,
            "estimated_length_m": 240,
            "timestamp": "2026-09-04T06:14:22Z",
            "description": "Physical large vessel echo detected at origin corridor without matching active AIS ping!"
        },
        {
            "detection_id": "SAR-RADAR-TGT-002",
            "lat": 19.4510,
            "lon": 71.5490,
            "rcs_db": 38.2,
            "estimated_length_m": 220,
            "timestamp": "2026-09-04T06:14:22Z",
            "description": "Correlated with MV BHARAT RATNA (MMSI: 419000542)"
        },
        {
            "detection_id": "SAR-RADAR-TGT-003",
            "lat": 19.7010,
            "lon": 71.2980,
            "rcs_db": 45.1,
            "estimated_length_m": 310,
            "timestamp": "2026-09-04T06:14:22Z",
            "description": "Correlated with CMA CGM INDUS (MMSI: 228394000)"
        }
    ]
