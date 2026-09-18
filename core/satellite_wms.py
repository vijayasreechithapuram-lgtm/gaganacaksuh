"""
Project Gaganacakṣuḥ: Satellite Raster Ingestion & Copernicus WMS Engine
Supports Sentinel-1 SAR (C-Band), ISRO EOS-04, Sentinel-2 Optical MSI, and NOAA VIIRS.
Features automated look-alike discrimination (mineral oil slick vs biogenic algal blooms).
"""

import numpy as np
import math
from typing import Dict, Any, List, Tuple

COPERNICUS_CDSE_WMS_BASE = "https://sh.dataspace.copernicus.eu/ogc/wms"
SENTINEL_HUB_WMS_BASE = "https://services.sentinel-hub.com/ogc/wms"

# Open-Source NASA GIBS WMS (Web Mercator EPSG:3857, No API key or instance ID required)
NASA_GIBS_WMS_BASE = "https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi"
NASA_GIBS_LAYER = "MODIS_Terra_CorrectedReflectance_TrueColor"
NASA_GIBS_ATTR = "NASA GIBS / MODIS"
NASA_GIBS_FORMAT = "image/jpeg"

REGIONS = {
    "mumbai_high": {
        "name": "Mumbai High Offshore Basin (Arabian Sea)",
        "lat": 19.4167,
        "lon": 71.3333,
        "bounds": [19.15, 71.05, 19.68, 71.60],
        "tactical_zone": "WESTERN NAVAL COMMAND / EEZ SECTOR 4",
        "primary_threat": "Crude Carrier Bunker Discharge & Offshore Platform Leak"
    },
    "gulf_of_kachchh": {
        "name": "Gulf of Kachchh Port Approaches (Gujarat)",
        "lat": 22.4500,
        "lon": 69.5000,
        "bounds": [22.20, 69.20, 22.70, 69.80],
        "tactical_zone": "NORTH-WEST MARITIME COMMAND / DEENDAYAL PORT",
        "primary_threat": "Heavy VLCC Tanker Traffic & Single Point Mooring"
    },
    "cochin_traffic_sep": {
        "name": "Cochin Traffic Separation Scheme (Kerala)",
        "lat": 9.9312,
        "lon": 76.1000,
        "bounds": [9.70, 75.85, 10.15, 76.35],
        "tactical_zone": "SOUTHERN NAVAL COMMAND / COCHIN COAST",
        "primary_threat": "International Shipping Lane Bilge Dumps"
    }
}

CONSTELLATIONS = {
    "sentinel_1": {
        "id": "S1_SAR",
        "name": "Sentinel-1 (ESA / Copernicus)",
        "type": "C-Band Synthetic Aperture Radar (SAR)",
        "polarization": "VV + VH (Interferometric Wide Swath)",
        "resolution": "10m x 10m",
        "revisit": "Sub-daily constellation pass",
        "all_weather": True,
        "night_capable": True,
        "primary_utility": "Radar dark spot backscatter reduction detection"
    },
    "isro_eos04": {
        "id": "EOS04_SAR",
        "name": "ISRO EOS-04 (RISAT-1A)",
        "type": "C-Band Polarimetric Radar (ISRO)",
        "polarization": "Circular / Hybrid Quad-Pol",
        "resolution": "3m - 25m",
        "revisit": "Scheduled sovereign tasking",
        "all_weather": True,
        "night_capable": True,
        "primary_utility": "Sovereign validation & dual-pol co-registration"
    },
    "sentinel_2": {
        "id": "S2_MSI",
        "name": "Sentinel-2 MSI (Copernicus)",
        "type": "Multi-Spectral Optical (13 Bands)",
        "bands": ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)", "B11 (SWIR)"],
        "resolution": "10m - 20m",
        "all_weather": False,
        "night_capable": False,
        "primary_utility": "Chlorophyll/NDVI false positive rejection (algal blooms)"
    },
    "noaa_viirs": {
        "id": "VIIRS_DNB",
        "name": "Suomi NPP / NOAA-20 VIIRS",
        "type": "Day/Night Band Radiance & Thermal IR",
        "resolution": "375m - 750m",
        "all_weather": True,
        "night_capable": True,
        "primary_utility": "Nighttime sea-surface temperature & flare/thermal contrast"
    }
}

def generate_synthetic_sar_scene(
    width: int = 256,
    height: int = 256,
    seed: int = 42,
    spill_coords: Tuple[int, int] = (120, 135),
    spill_axes: Tuple[int, int] = (45, 18),
    angle_deg: float = 32.0,
    with_lookalike: bool = True
) -> Dict[str, Any]:
    """
    Generates a synthetic yet mathematically accurate Sentinel-1 C-Band SAR amplitude chip
    with Rayleigh/Gamma speckle noise, ocean clutter, a true mineral oil slick dark spot,
    and an optional biogenic look-alike (algal bloom / low-wind patch) for discrimination testing.
    """
    np.random.seed(seed)
    
    # Baseline sea clutter (Gamma distributed radar backscatter)
    # Mean sea radar cross-section ~ 120 (0-255 grayscale)
    shape, scale = 9.0, 13.5
    sea_clutter = np.random.gamma(shape, scale, (height, width))
    sea_clutter = np.clip(sea_clutter, 15, 240).astype(np.float32)
    
    # 1. Mineral Oil Slick: Strong capillary wave damping -> dark spot (low backscatter ~ 25-45)
    y, x = np.ogrid[:height, :width]
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    
    # Rotated coordinates for primary spill
    dx = x - spill_coords[0]
    dy = y - spill_coords[1]
    rx = dx * cos_a + dy * sin_a
    ry = -dx * sin_a + dy * cos_a
    
    # Irregular spill contour using sinusoidal perturbation
    theta = np.arctan2(ry, rx)
    dist_r = np.sqrt((rx / spill_axes[0])**2 + (ry / spill_axes[1])**2)
    edge_noise = 0.18 * np.sin(5 * theta) + 0.12 * np.cos(7 * theta)
    slick_mask = (dist_r + edge_noise) <= 1.0
    
    # Tail / drift extension
    tail_mask = ((rx > 0) & (rx < spill_axes[0] * 1.6) & (np.abs(ry) < (spill_axes[1] * 0.45 * (1 - rx / (spill_axes[0] * 1.8)))))
    full_slick_mask = np.logical_or(slick_mask, tail_mask)
    
    # Apply backscatter depression (damping ratio ~ 8-12 dB)
    sar_amplitude = sea_clutter.copy()
    sar_amplitude[full_slick_mask] = np.random.gamma(4.0, 7.5, np.sum(full_slick_mask))
    sar_amplitude = np.clip(sar_amplitude, 0, 255).astype(np.uint8)
    
    # 2. Look-alike mask (Biogenic slick / Algal patch or low wind shelter)
    lookalike_mask = np.zeros((height, width), dtype=bool)
    if with_lookalike:
        lx = x - 60
        ly = y - 60
        lr = np.sqrt((lx / 30.0)**2 + (ly / 22.0)**2)
        lookalike_mask = lr <= 1.0
        # Mild damping for biogenic slick (less sharp gradient)
        sar_amplitude[lookalike_mask] = (0.55 * sar_amplitude[lookalike_mask] + 0.45 * 65).astype(np.uint8)

    # 3. Sentinel-2 Optical MSI simulated bands
    # Water: low NIR, moderate Blue/Green
    # Mineral Oil: subtle sheen in SWIR/Sunglint, low NDVI (< 0)
    # Algal Bloom (Lookalike): very high NIR reflectance, high NDVI (> 0.45)
    s2_nir = np.full((height, width), 40, dtype=np.float32)
    s2_red = np.full((height, width), 50, dtype=np.float32)
    
    # Mineral oil: low NIR, slight oil sheen signature
    s2_nir[full_slick_mask] = 28.0
    s2_red[full_slick_mask] = 55.0
    
    # Biogenic look-alike: vegetation/algal bloom response (red edge spike)
    s2_nir[lookalike_mask] = 160.0
    s2_red[lookalike_mask] = 40.0
    
    # NDVI = (NIR - RED) / (NIR + RED + 1e-6)
    ndvi = (s2_nir - s2_red) / (s2_nir + s2_red + 1e-6)
    
    return {
        "sar_amplitude": sar_amplitude,
        "ground_truth_mask": full_slick_mask.astype(np.uint8),
        "lookalike_mask": lookalike_mask.astype(np.uint8),
        "ndvi": ndvi,
        "width": width,
        "height": height,
        "spill_pixel_area": int(np.sum(full_slick_mask)),
        "spill_sq_km": float(np.sum(full_slick_mask) * (0.01 * 0.01)), # 10m pixel = 0.01km x 0.01km = 0.0001 km2
        "lookalike_sq_km": float(np.sum(lookalike_mask) * (0.01 * 0.01))
    }

def get_wms_capabilities_mock(region_key: str = "mumbai_high") -> Dict[str, Any]:
    """Returns simulated live WMS handshake metadata for Copernicus Data Space."""
    reg = REGIONS.get(region_key, REGIONS["mumbai_high"])
    return {
        "service": "Copernicus Data Space Ecosystem WMS 1.3.0",
        "endpoint": f"{COPERNICUS_CDSE_WMS_BASE}?service=WMS&version=1.3.0&request=GetCapabilities",
        "crs": "EPSG:4326 (WGS84) & EPSG:3857 (Web Mercator)",
        "target_region": reg["name"],
        "bbox": reg["bounds"],
        "tactical_zone": reg["tactical_zone"],
        "active_layers": [
            {
                "layer_id": "S1_SAR_IW_VV",
                "title": "Sentinel-1 IW GRDH VV Amplitude (dB)",
                "timestamp": "2026-09-04T06:14:22Z",
                "orbit": "Descending Pass #117",
                "status": "ONLINE (STREAMING 10m PIXELS)"
            },
            {
                "layer_id": "S1_SAR_IW_VH",
                "title": "Sentinel-1 IW GRDH VH Cross-Pol",
                "timestamp": "2026-09-04T06:14:22Z",
                "orbit": "Descending Pass #117",
                "status": "ONLINE (STREAMING 10m PIXELS)"
            },
            {
                "layer_id": "S2_MSI_L2A_RGB",
                "title": "Sentinel-2 MSI Level-2A True Color",
                "timestamp": "2026-09-04T05:42:08Z",
                "orbit": "Relative Orbit R062",
                "status": "ONLINE (CO-REGISTERED)"
            },
            {
                "layer_id": "S2_MSI_SWIR_NDVI",
                "title": "Sentinel-2 False Color & Algal Bloom NDVI Index",
                "timestamp": "2026-09-04T05:42:08Z",
                "orbit": "Relative Orbit R062",
                "status": "ONLINE (CO-REGISTERED)"
            }
        ]
    }
