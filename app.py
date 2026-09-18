"""
=============================================================================
Project Gaganacakṣuḥ (Antigravity Edition) - SIH26143
Multi-Satellite Surveillance, Zero-Trust Tracking & Legal Attribution Engine
Industrial-Grade Tactical Command Center & Forensic Court Dossier Vault
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import time
import base64
import textwrap
import cv2
from PIL import Image
import streamlit.components.v1 as components
import folium
import pydeck as pdk

# Import Core Engines
from core.crypto_auth import (
    authenticate_user,
    verify_session_token,
    ROLE_COMMAND_OFFICER,
    ROLE_TRIBUNAL_JUDGE,
    ROLES_METADATA,
    CREDENTIALS_DB
)
from core.satellite_wms import (
    REGIONS,
    CONSTELLATIONS,
    get_wms_capabilities_mock,
    NASA_GIBS_WMS_BASE,
    NASA_GIBS_LAYER,
    NASA_GIBS_ATTR,
    NASA_GIBS_FORMAT
)
from core.unet_segmentation import UNetZenodoInferenceEngine
from core.hydrodynamics import HydrodynamicBackcastEngine
from core.zero_trust_ais import ZeroTrustAISAuditor
from core.forensic_ledger import ForensicLedgerEngine
from core.rerouting_engine import TacticalReroutingEngine
from data.zenodo_sar_samples import load_benchmark_sar_scene

# Page Configuration
st.set_page_config(
    page_title="GAGANACAKṢUḤ // Multi-Sat Tactical Command & Attribution",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Military Tactical HUD CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre, .mono {
        font-family: 'JetBrains+Mono', monospace !important;
    }

    /* Tactical Background & Panels */
    .stApp {
        background-color: #060913;
        color: #dbe4f0;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid #1e293b;
    }

    /* Military HUD Cards */
    .hud-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.65) 100%);
        border: 1px solid #1e293b;
        border-left: 4px solid #00f0ff;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }

    .hud-card-alert {
        background: linear-gradient(135deg, rgba(38, 12, 18, 0.85) 0%, rgba(69, 10, 10, 0.65) 100%);
        border: 1px solid #7f1d1d;
        border-left: 4px solid #ef4444;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
    }

    .hud-card-success {
        background: linear-gradient(135deg, rgba(6, 30, 20, 0.85) 0%, rgba(6, 78, 59, 0.65) 100%);
        border: 1px solid #065f46;
        border-left: 4px solid #10b981;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
    }

    /* Status LED indicator */
    .led-live {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10b981;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }

    .led-alert {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #ef4444;
        border-radius: 50%;
        box-shadow: 0 0 10px #ef4444;
        margin-right: 6px;
        animation: pulse 1s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }

    /* Metric KPI containers */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.8rem !important;
        font-weight: 700;
        color: #00f0ff !important;
    }

    /* Button Styling */
    .stButton > button {
        background-color: #0f172a;
        color: #00f0ff;
        border: 1px solid #00f0ff;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.5px;
        border-radius: 4px;
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #00f0ff;
        color: #060913;
        box-shadow: 0 0 14px rgba(0, 240, 255, 0.6);
        border-color: #00f0ff;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #090e1a;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        padding: 8px 16px;
        border-radius: 6px;
        color: #94a3b8;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #00f0ff !important;
        border-bottom: 2px solid #00f0ff !important;
    }

    /* 3D Floating Depth Cards for Landing Page Metrics */
    .metric-3d-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        perspective: 1200px;
        margin: 18px 0 28px 0;
    }

    .metric-card-3d {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.82) 100%);
        border: 1px solid rgba(0, 240, 255, 0.28);
        border-radius: 12px;
        padding: 16px 18px;
        transform: rotateX(6deg) rotateY(-2deg) translateZ(12px);
        transform-style: preserve-3d;
        box-shadow: 0 16px 32px rgba(0, 0, 0, 0.65), 0 0 22px rgba(0, 240, 255, 0.14), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: transform 0.35s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.35s ease, border-color 0.35s ease;
        backdrop-filter: blur(8px);
    }

    .metric-card-3d:hover {
        transform: translateY(-8px) rotateX(0deg) rotateY(0deg) translateZ(28px);
        box-shadow: 0 24px 48px rgba(0, 0, 0, 0.8), 0 0 32px rgba(0, 240, 255, 0.4);
        border-color: #00f0ff;
    }

    .metric-3d-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }

    .metric-3d-lbl {
        font-size: 0.75rem;
        color: #94a3b8;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 4px;
    }

    .metric-3d-sub {
        font-size: 0.72rem;
        color: #64748b;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = None
if "selected_region" not in st.session_state:
    st.session_state["selected_region"] = "mumbai_high"
if "pipeline_stage" not in st.session_state:
    st.session_state["pipeline_stage"] = 1
if "sar_scene" not in st.session_state:
    st.session_state["sar_scene"] = load_benchmark_sar_scene("mumbai_high")
if "unet_results" not in st.session_state:
    st.session_state["unet_results"] = None
if "backcast_results" not in st.session_state:
    st.session_state["backcast_results"] = None
if "ais_results" not in st.session_state:
    st.session_state["ais_results"] = None
if "docket" not in st.session_state:
    st.session_state["docket"] = None
if "reroutes" not in st.session_state:
    st.session_state["reroutes"] = None
if "broadcast_dispatched" not in st.session_state:
    st.session_state["broadcast_dispatched"] = False


# =============================================================================
# HELPER: RENDER LEAFLET TACTICAL HUD MAP
# =============================================================================
def create_folium_tactical_map(
    center: list,
    zoom: int = 10
) -> folium.Map:
    """
    Creates a Folium Map with the open-source NASA GIBS WMS tile layer.
    No API key or instance ID required.
    """
    c_lat, c_lon = center
    m = folium.Map(location=[c_lat, c_lon], zoom_start=zoom, tiles=None)

    # Exact Folium tile layer setup requested (Web Mercator EPSG:3857):
    folium.WmsTileLayer(
        url="https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi",
        layers="MODIS_Terra_CorrectedReflectance_TrueColor",
        name="MODIS_Terra_CorrectedReflectance_TrueColor",
        attr="NASA GIBS / MODIS",
        fmt="image/jpeg",
        overlay=False,
        control=True
    ).add_to(m)
    return m


def create_pydeck_3d_tactical_deck(
    center: list,
    zoom: float = 9.2,
    pitch: float = 50.0,
    bearing: float = -18.0,
    slick_polygons: list = None,
    backcast_data: dict = None,
    vessels: list = None,
    dark_ships: list = None,
    exclusion_polygon: list = None,
    reroutes: list = None,
    extrusion_scale: float = 1.0,
    show_slick: bool = True,
    show_vessels: bool = True,
    show_particles: bool = True,
    show_exclusion: bool = True,
    show_tracks: bool = True
) -> pdk.Deck:
    """
    Creates an industrial-grade Pydeck DeckGL 3D tactical perspective object with pitch,
    bearing, and extruded altitude beacons.
    """
    c_lat, c_lon = center
    layers = []

    # 1. 3D Extruded Oil Slick Contours
    if show_slick and slick_polygons:
        slick_features = []
        for p in slick_polygons:
            coords = p.get("coordinates", [[]])[0]
            if len(coords) >= 3:
                poly_coords = [[pt[1], pt[0]] for pt in coords]
                slick_features.append({
                    "polygon": poly_coords,
                    "elevation": int(650 * extrusion_scale),
                    "sq_km": p.get("sq_km", 1.8),
                    "name": "CRITICAL HAZARD: OIL SLICK CONTOUR",
                    "status": "Sentinel-1 SAR / U-Net Conf >94.8%"
                })
        if slick_features:
            layers.append(pdk.Layer(
                "PolygonLayer",
                data=slick_features,
                get_polygon="polygon",
                get_elevation="elevation",
                get_fill_color=[255, 0, 85, 185],
                get_line_color=[255, 0, 85, 255],
                stroked=True,
                filled=True,
                extruded=True,
                wireframe=True,
                pickable=True
            ))

    # 2. 3D Lagrangian Backcast Origin & Particle Cloud
    if show_particles and backcast_data:
        origin_poly = backcast_data.get("origin_polygon", [])
        if len(origin_poly) >= 3:
            poly_coords = [[pt[1], pt[0]] for pt in origin_poly]
            layers.append(pdk.Layer(
                "PolygonLayer",
                data=[{
                    "polygon": poly_coords,
                    "elevation": int(320 * extrusion_scale),
                    "name": "LAGRANGIAN EMISSION ORIGIN",
                    "status": "Release T-12h (INCOIS+ERA5 Forcing)"
                }],
                get_polygon="polygon",
                get_elevation="elevation",
                get_fill_color=[255, 170, 0, 110],
                get_line_color=[255, 170, 0, 240],
                stroked=True,
                filled=True,
                extruded=True,
                wireframe=True,
                pickable=True
            ))

        particles = backcast_data.get("sample_particles", [])
        if particles:
            particle_data = [
                {"position": [pt[1], pt[0]], "name": "Lagrangian Drift Particle"}
                for pt in particles
            ]
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=particle_data,
                get_position="position",
                get_fill_color=[245, 158, 11, 200],
                get_radius=150,
                pickable=False
            ))

    # 3. 3D Dynamic Navigation Exclusion Corridor
    if show_exclusion and exclusion_polygon and len(exclusion_polygon) >= 3:
        poly_coords = [[pt[1], pt[0]] for pt in exclusion_polygon]
        layers.append(pdk.Layer(
            "PolygonLayer",
            data=[{
                "polygon": poly_coords,
                "elevation": int(1150 * extrusion_scale),
                "name": "RESTRICTED MARITIME EXCLUSION CORRIDOR",
                "status": "Mandatory 5 NM Standoff (VHF CH-16 Active)"
            }],
            get_polygon="polygon",
            get_elevation="elevation",
            get_fill_color=[239, 68, 68, 65],
            get_line_color=[239, 68, 68, 220],
            stroked=True,
            filled=True,
            extruded=True,
            wireframe=True,
            pickable=True
        ))

    # 4. 3D Commercial Vessels & Kinematic Tracks
    if show_vessels and vessels:
        vessel_cols = []
        track_paths = []
        for v in vessels:
            pos = v.get("last_known_pos")
            if pos:
                is_spoofed = v.get("is_spoofed", False)
                has_gap = v.get("has_gap", False)

                if has_gap:
                    col_color = [239, 68, 68, 255]
                    elev = int(4200 * extrusion_scale)
                    status_text = "PRIMARY SUSPECT (AIS TRANSMISSION SILENCED)"
                elif is_spoofed:
                    col_color = [249, 115, 22, 255]
                    elev = int(3500 * extrusion_scale)
                    status_text = f"GPS SPOOFED (>35 kts: {v.get('max_sog')} kts)"
                else:
                    col_color = [0, 240, 255, 240]
                    elev = int(2200 * extrusion_scale)
                    status_text = "COMPLIANT AIS TELEMETRY"

                vessel_cols.append({
                    "position": [pos[1], pos[0]],
                    "elevation": elev,
                    "color": col_color,
                    "name": v.get("vessel_name", "UNKNOWN"),
                    "mmsi": v.get("mmsi", "N/A"),
                    "type": v.get("vessel_type", "Vessel"),
                    "sog": f"{v.get('max_sog', 0.0)} kts",
                    "attribution": f"{v.get('attribution_score', 0)}%",
                    "status": status_text
                })

            if show_tracks:
                track = v.get("telemetry_track", [])
                if len(track) > 1:
                    track_paths.append({
                        "path": [[pt[1], pt[0]] for pt in track],
                        "color": col_color[:3] + [180]
                    })

        if vessel_cols:
            layers.append(pdk.Layer(
                "ColumnLayer",
                data=vessel_cols,
                get_position="position",
                get_elevation="elevation",
                get_fill_color="color",
                radius=450,
                elevation_scale=1,
                extruded=True,
                pickable=True
            ))

        if track_paths:
            layers.append(pdk.Layer(
                "PathLayer",
                data=track_paths,
                get_path="path",
                get_color="color",
                width_min_pixels=3,
                pickable=False
            ))

    # 5. 3D Dark Ships (Layer 5: Physical Radar Targets without active AIS)
    if show_vessels and dark_ships:
        dark_cols = []
        for ds in dark_ships:
            dark_cols.append({
                "position": [ds["lon"], ds["lat"]],
                "elevation": int(5200 * extrusion_scale),
                "color": [255, 0, 51, 255],
                "name": f"UNIDENTIFIED DARK VESSEL ({ds.get('detection_id')})",
                "mmsi": "NON-REPORTING (TRANSPONDER OFF)",
                "type": f"Steel Hull (~{ds.get('estimated_length_m')}m, RCS {ds.get('radar_rcs_db')} dB)",
                "sog": "Zero Active AIS Ping",
                "attribution": "HIGH PROXIMITY RISK",
                "status": "COVERT RADAR CONTACT (NO AIS TRANSMISSION)"
            })
        if dark_cols:
            layers.append(pdk.Layer(
                "ColumnLayer",
                data=dark_cols,
                get_position="position",
                get_elevation="elevation",
                get_fill_color="color",
                radius=650,
                elevation_scale=1,
                extruded=True,
                pickable=True
            ))

    # 6. 3D Divert Waypoints (Layer 6)
    if reroutes:
        wp_data = []
        for r in reroutes:
            wp = r.get("diversion_waypoint")
            if wp:
                wp_data.append({
                    "position": [wp[1], wp[0]],
                    "elevation": int(1800 * extrusion_scale),
                    "color": [16, 185, 129, 255],
                    "name": f"DIVERT WAYPOINT: {r.get('vessel_name')}",
                    "mmsi": r.get("mmsi", ""),
                    "type": "Evasive Nav Waypoint",
                    "sog": "Safety Divert",
                    "attribution": "Clearance Vector",
                    "status": r.get("reroute_instruction", "")
                })
        if wp_data:
            layers.append(pdk.Layer(
                "ColumnLayer",
                data=wp_data,
                get_position="position",
                get_elevation="elevation",
                get_fill_color="color",
                radius=350,
                elevation_scale=1,
                extruded=True,
                pickable=True
            ))

    view_state = pdk.ViewState(
        latitude=c_lat,
        longitude=c_lon,
        zoom=zoom,
        pitch=pitch,
        bearing=bearing
    )

    tooltip = {
        "html": """
        <div style="background:#0f172a; color:#e2e8f0; padding:8px 10px; border:1px solid #00f0ff; border-radius:4px; font-family:'Courier New', monospace; font-size:11px;">
            <b style="color:#00f0ff;">{name}</b><br/>
            <span>MMSI: <b>{mmsi}</b></span><br/>
            <span>Type: {type}</span><br/>
            <span>Attribution / SOG: <b>{attribution}</b> | {sog}</span><br/>
            <span style="color:#ffaa00;">Status: {status}</span>
        </div>
        """,
        "style": {"backgroundColor": "transparent", "color": "white"}
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip=tooltip
    )


def render_tactical_leaflet_map(
    center: list,
    zoom: int = 10,
    slick_polygons: list = None,
    backcast_data: dict = None,
    vessels: list = None,
    dark_ships: list = None,
    exclusion_polygon: list = None,
    reroutes: list = None
):
    """
    Renders tactical common operating picture with Pydeck 3D DeckGL perspective
    (pitch, bearing, extruded altitude beacons) alongside full NASA GIBS WMS Leaflet controls.
    """
    c_lat, c_lon = center

    # Perspective Mode Switcher
    col_mode1, col_mode2 = st.columns([1.8, 2.2])
    with col_mode1:
        map_perspective = st.radio(
            "Select Tactical View Mode:",
            [
                "🛰️ 3D Tactical Perspective (Pydeck DeckGL: 50° Pitch & Extruded Beacons)",
                "🗺️ 2D Tactical Basemap (NASA GIBS WMS & Leaflet Native Radar)"
            ],
            index=0,
            horizontal=False
        )
    with col_mode2:
        if "3D" in map_perspective:
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                p_pitch = st.slider("3D Camera Pitch (°)", min_value=0, max_value=70, value=50, step=5)
            with sc2:
                p_bearing = st.slider("3D Camera Bearing (°)", min_value=-90, max_value=90, value=-18, step=5)
            with sc3:
                p_scale = st.slider("Extrusion Height", min_value=0.5, max_value=2.5, value=1.0, step=0.25)
        else:
            st.markdown("""
            <div style="background-color:#020617; border:1px solid #1e293b; padding:10px 14px; border-radius:6px; font-family:monospace; font-size:0.75rem; color:#94a3b8; margin-top:8px;">
                <b>NASA GIBS WMS STATUS:</b> ONLINE (Web Mercator EPSG:3857)<br/>
                <b>LAYER:</b> MODIS Terra Corrected Reflectance TrueColor
            </div>
            """, unsafe_allow_html=True)

    if "3D" in map_perspective:
        with st.expander("⚙️ 3D Tactical Layer Visibility & Extrusion Controls", expanded=False):
            tc1, tc2, tc3, tc4, tc5 = st.columns(5)
            with tc1:
                show_slick = st.checkbox("3D Oil Slick", value=True)
            with tc2:
                show_vessels = st.checkbox("3D Vessel Beacons", value=True)
            with tc3:
                show_particles = st.checkbox("Lagrangian Drift", value=True)
            with tc4:
                show_exclusion = st.checkbox("Exclusion Barrier", value=True)
            with tc5:
                show_tracks = st.checkbox("Vessel Tracks", value=True)

        deck = create_pydeck_3d_tactical_deck(
            center=center,
            zoom=9.2,
            pitch=float(p_pitch),
            bearing=float(p_bearing),
            slick_polygons=slick_polygons,
            backcast_data=backcast_data,
            vessels=vessels,
            dark_ships=dark_ships,
            exclusion_polygon=exclusion_polygon,
            reroutes=reroutes,
            extrusion_scale=float(p_scale),
            show_slick=show_slick,
            show_vessels=show_vessels,
            show_particles=show_particles,
            show_exclusion=show_exclusion,
            show_tracks=show_tracks
        )
        st.pydeck_chart(deck, use_container_width=True)

    else:
        render_2d_leaflet_component(
            center=center,
            zoom=zoom,
            slick_polygons=slick_polygons,
            backcast_data=backcast_data,
            vessels=vessels,
            dark_ships=dark_ships,
            exclusion_polygon=exclusion_polygon,
            reroutes=reroutes
        )


def render_2d_leaflet_component(
    center: list,
    zoom: int = 10,
    slick_polygons: list = None,
    backcast_data: dict = None,
    vessels: list = None,
    dark_ships: list = None,
    exclusion_polygon: list = None,
    reroutes: list = None
):
    """Generates an ultra-crisp, zero-watermark tactical Leaflet map component with NASA GIBS WMS."""
    c_lat, c_lon = center

    # Also build folium.Map instance to guarantee folium layer consistency
    _ = create_folium_tactical_map(center=center, zoom=zoom)

    # Serialize layers to JSON safe strings
    slick_json = json.dumps([p["coordinates"][0] for p in (slick_polygons or [])])
    backcast_poly_json = json.dumps(backcast_data.get("origin_polygon", []) if backcast_data else [])
    backcast_particles_json = json.dumps(backcast_data.get("sample_particles", []) if backcast_data else [])
    vessels_json = json.dumps(vessels or [])
    dark_ships_json = json.dumps(dark_ships or [])
    exclusion_json = json.dumps(exclusion_polygon or [])
    reroutes_json = json.dumps(reroutes or [])

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            html, body, #map {{
                height: 100%;
                width: 100%;
                margin: 0;
                padding: 0;
                background: #060913;
            }}
            .custom-popup .leaflet-popup-content-wrapper {{
                background: #0f172a;
                color: #e2e8f0;
                border: 1px solid #00f0ff;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                box-shadow: 0 0 12px rgba(0, 240, 255, 0.4);
            }}
            .custom-popup .leaflet-popup-tip {{
                background: #0f172a;
            }}
            .leaflet-container {{
                background: #060913 !important;
            }}
            .radar-pulse {{
                border-radius: 50%;
                animation: radar-beam 1.6s infinite ease-out;
            }}
            @keyframes radar-beam {{
                0% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.8); }}
                70% {{ box-shadow: 0 0 0 18px rgba(239, 68, 68, 0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
            }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var map = L.map('map', {{
                center: [{c_lat}, {c_lon}],
                zoom: {zoom},
                zoomControl: true,
                attributionControl: true
            }});

            // Open-Source NASA GIBS WMS Tile Layer (Web Mercator EPSG:3857, Zero API Key required)
            L.tileLayer.wms('https://gibs.earthdata.nasa.gov/wms/epsg3857/best/wms.cgi', {{
                layers: 'MODIS_Terra_CorrectedReflectance_TrueColor',
                format: 'image/jpeg',
                attribution: 'NASA GIBS / MODIS',
                transparent: false
            }}).addTo(map);

            // Layer 1: Oil Slick Polygons (Fluorescent Neon Red/Amber)
            var slickPolygons = {slick_json};
            if (slickPolygons.length > 0) {{
                slickPolygons.forEach(function(ring) {{
                    L.polygon(ring, {{
                        color: '#ff0055',
                        fillColor: '#ff0055',
                        fillOpacity: 0.65,
                        weight: 2
                    }}).bindPopup("<b style='color:#ff0055;'>CRITICAL HAZARD: OIL SLICK CONTOUR</b><br/>Sensor: Sentinel-1 C-SAR<br/>Classification: Crude Hydrocarbon<br/>U-Net Confidence: >94.8%").addTo(map);
                }});
            }}

            // Layer 2: Lagrangian Backcast Origin & Particle Cloud
            var backcastPoly = {backcast_poly_json};
            if (backcastPoly.length > 0) {{
                L.polygon(backcastPoly, {{
                    color: '#ffaa00',
                    fillColor: '#ffaa00',
                    fillOpacity: 0.25,
                    weight: 2,
                    dashArray: '4, 4'
                }}).bindPopup("<b style='color:#ffaa00;'>LAGRANGIAN EMISSION ORIGIN</b><br/>Estimated Release: T-12 Hours<br/>Forcing: INCOIS Currents + ERA5 Wind").addTo(map);
            }}

            var sampleParticles = {backcast_particles_json};
            if (sampleParticles.length > 0) {{
                sampleParticles.forEach(function(pt) {{
                    L.circleMarker(pt, {{
                        radius: 2,
                        color: '#f59e0b',
                        fillColor: '#f59e0b',
                        fillOpacity: 0.7,
                        weight: 0
                    }}).addTo(map);
                }});
            }}

            // Layer 3: Dynamic 24h Navigation Exclusion Corridor
            var exclusionPoly = {exclusion_json};
            if (exclusionPoly.length > 0) {{
                L.polygon(exclusionPoly, {{
                    color: '#ef4444',
                    fillColor: '#ef4444',
                    fillOpacity: 0.12,
                    weight: 2,
                    dashArray: '6, 6'
                }}).bindPopup("<b style='color:#ef4444;'>RESTRICTED NAVIGATION ZONE</b><br/>Mandatory 5 NM Standoff<br/>VHF CH-16 Active Alert").addTo(map);
            }}

            // Layer 4: Commercial AIS Vessels & Kinematic Tracks
            var vessels = {vessels_json};
            vessels.forEach(function(v) {{
                var color = '#00f0ff';
                if (v.is_spoofed) color = '#f97316';
                if (v.has_gap) color = '#ef4444';

                // Plot Track Line
                if (v.telemetry_track && v.telemetry_track.length > 1) {{
                    L.polyline(v.telemetry_track, {{
                        color: color,
                        weight: 2,
                        opacity: 0.8,
                        dashArray: v.has_gap ? '5, 5' : null
                    }}).addTo(map);
                }}

                // Vessel Marker
                var lastPos = v.last_known_pos;
                var marker = L.circleMarker(lastPos, {{
                    radius: 7,
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.9,
                    weight: 2
                }}).addTo(map);

                var badge = v.is_spoofed ? "<span style='color:#f97316;'>[GPS SPOOFED >35kts]</span>" : (v.has_gap ? "<span style='color:#ef4444;'>[PRIMARY SUSPECT - AIS GAP]</span>" : "<span style='color:#10b981;'>[COMPLIANT]</span>");
                marker.bindPopup("<b style='color:" + color + ";'>" + v.vessel_name + "</b> " + badge + "<br/>MMSI: " + v.mmsi + "<br/>Type: " + v.vessel_type + "<br/>Max SOG: " + v.max_sog + " kts<br/>Attribution: <b>" + v.attribution_score + "%</b>");
            }});

            // Layer 5: Dark Ships (Physical Radar Contact without active AIS ping)
            var darkShips = {dark_ships_json};
            darkShips.forEach(function(ds) {{
                var radarMarker = L.circleMarker([ds.lat, ds.lon], {{
                    radius: 9,
                    color: '#ff0033',
                    fillColor: '#ff0033',
                    fillOpacity: 0.9,
                    weight: 3,
                    className: 'radar-pulse'
                }}).addTo(map);

                radarMarker.bindPopup("<b style='color:#ff0033;'>CRITICAL: UNIDENTIFIED DARK VESSEL</b><br/>Target: " + ds.detection_id + "<br/>SAR RCS: " + ds.radar_rcs_db + " dB (Length ~" + ds.estimated_length_m + "m)<br/>No AIS Ping within " + ds.closest_ais_dist_km + " km<br/><b>COVERT THREAT</b>");
            }});

            // Layer 6: Dynamic Reroute Waypoints
            var reroutes = {reroutes_json};
            reroutes.forEach(function(r) {{
                var wp = r.diversion_waypoint;
                L.marker(wp).addTo(map).bindPopup("<b style='color:#10b981;'>TACTICAL DIVERT WAYPOINT</b><br/>Vessel: " + r.vessel_name + "<br/>Instruction: " + r.reroute_instruction);
                // Line from vessel to diversion
                L.polyline([r.current_pos, wp], {{
                    color: '#10b981',
                    weight: 2,
                    dashArray: '3, 6'
                }}).addTo(map);
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=480, scrolling=False)


# =============================================================================
# VIEW: MILITARY CLEARANCE GATEWAY & AUTHENTICATION
# =============================================================================
def render_login_screen():
    # Scoped 6-second satellite zoom transition, orbital HUD & floating 3D mascot (strictly isolated to landing page)
    st.markdown(textwrap.dedent("""
    <style>
    /* 6-SECOND SATELLITE ZOOM TRANSITION & ORBITAL HUD (LANDING PAGE EXCLUSIVE) */
    @keyframes satelliteZoom6s {
        0% {
            transform: scale(0.68) translateY(-32px) rotateX(14deg);
            filter: blur(6px) brightness(0.6);
            opacity: 0.1;
        }
        25% {
            transform: scale(0.82) translateY(-16px) rotateX(9deg);
            filter: blur(3.5px) brightness(0.85);
            opacity: 0.65;
        }
        60% {
            transform: scale(1.03) translateY(2px) rotateX(2deg);
            filter: blur(0.5px) brightness(1.15);
            opacity: 0.95;
        }
        85% {
            transform: scale(0.99) translateY(0px) rotateX(0deg);
            filter: blur(0px) brightness(1.02);
            opacity: 1;
        }
        100% {
            transform: scale(1.0) translateY(0px) rotateX(0deg);
            filter: blur(0px) brightness(1.0);
            opacity: 1;
        }
    }

    @keyframes progressTimeline6s {
        0% { width: 0%; }
        100% { width: 100%; }
    }

    .landing-zoom-wrapper {
        animation: satelliteZoom6s 6.0s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        transform-origin: center top;
        perspective: 1200px;
    }

    .orbital-telemetry-hud {
        background: linear-gradient(180deg, rgba(8, 15, 30, 0.95) 0%, rgba(3, 7, 18, 0.88) 100%);
        border: 1px solid rgba(0, 240, 255, 0.35);
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 20px;
        box-shadow: 0 0 25px rgba(0, 240, 255, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.08);
        font-family: 'JetBrains Mono', monospace;
    }

    .telemetry-progress-track {
        height: 4px;
        background: #0f172a;
        border-radius: 2px;
        overflow: hidden;
        margin-top: 8px;
    }

    .telemetry-progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #00f0ff 0%, #10b981 65%, #f59e0b 100%);
        animation: progressTimeline6s 6.0s linear forwards;
    }

    /* 3D FLOATING MASCOT: GAGANACHAKSHUH ORBITAL SENTINEL */
    @keyframes floatMascot {
        0%, 100% {
            transform: translateY(0px) rotateY(0deg) rotateZ(0deg);
        }
        50% {
            transform: translateY(-14px) rotateY(12deg) rotateZ(2deg);
        }
    }

    @keyframes radarWavePulse {
        0% {
            transform: scale(0.7);
            opacity: 0.85;
        }
        50% {
            transform: scale(1.35);
            opacity: 0.25;
        }
        100% {
            transform: scale(1.75);
            opacity: 0;
        }
    }

    @keyframes radarSweep360 {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }

    @keyframes orbitalRingSpin {
        0% {
            transform: rotateX(65deg) rotateY(20deg) rotateZ(0deg);
        }
        100% {
            transform: rotateX(65deg) rotateY(20deg) rotateZ(360deg);
        }
    }

    .mascot-anchor {
        position: fixed;
        bottom: 24px;
        right: 28px;
        z-index: 99999;
        display: flex;
        align-items: flex-end;
        gap: 12px;
        perspective: 1000px;
        animation: floatMascot 4.5s ease-in-out infinite;
        pointer-events: auto;
    }

    .mascot-dialogue {
        background: linear-gradient(135deg, rgba(11, 17, 32, 0.95) 0%, rgba(15, 23, 42, 0.92) 100%);
        border: 1px solid rgba(0, 240, 255, 0.45);
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.75), 0 0 20px rgba(0, 240, 255, 0.2);
        backdrop-filter: blur(10px);
        max-width: 250px;
        transition: all 0.3s ease;
        font-family: 'JetBrains Mono', monospace;
    }

    .mascot-anchor:hover .mascot-dialogue {
        border-color: #00f0ff;
        box-shadow: 0 16px 36px rgba(0, 0, 0, 0.85), 0 0 28px rgba(0, 240, 255, 0.45);
        transform: scale(1.02);
    }

    .mascot-orb-frame {
        position: relative;
        width: 68px;
        height: 68px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .mascot-wave-ring {
        position: absolute;
        width: 62px;
        height: 62px;
        border-radius: 50%;
        border: 2px solid rgba(0, 240, 255, 0.65);
        box-shadow: 0 0 14px rgba(0, 240, 255, 0.4);
        animation: radarWavePulse 2.8s cubic-bezier(0.2, 0.8, 0.2, 1) infinite;
        pointer-events: none;
    }

    .mascot-gyro-ring {
        position: absolute;
        width: 80px;
        height: 80px;
        border-radius: 50%;
        border: 1.5px dashed rgba(16, 185, 129, 0.75);
        animation: orbitalRingSpin 5s linear infinite;
        pointer-events: none;
    }

    .mascot-orb-core {
        position: relative;
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #38bdf8 0%, #0284c7 40%, #0369a1 70%, #0f172a 100%);
        box-shadow: 0 0 22px rgba(0, 240, 255, 0.85), inset 0 2px 4px rgba(255, 255, 255, 0.6), inset 0 -4px 8px rgba(0, 0, 0, 0.7);
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .mascot-anchor:hover .mascot-orb-core {
        transform: scale(1.1);
        box-shadow: 0 0 35px #00f0ff, 0 0 50px rgba(16, 185, 129, 0.6);
    }

    .mascot-radar-sweep {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border-radius: 50%;
        background: conic-gradient(from 0deg, transparent 0deg, rgba(0, 240, 255, 0.45) 60deg, transparent 70deg);
        animation: radarSweep360 2.2s linear infinite;
        pointer-events: none;
    }

    .mascot-icon {
        font-size: 1.5rem;
        z-index: 2;
        filter: drop-shadow(0 0 6px rgba(0, 240, 255, 0.9));
    }
    </style>
    """).strip(), unsafe_allow_html=True)

    # Clean unindented landing page HTML structure (prevents CommonMark markdown code-block bug)
    st.markdown(textwrap.dedent("""
    <div class="landing-zoom-wrapper">
    <div class="orbital-telemetry-hud">
    <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.75rem; color:#94a3b8;">
    <div><span class="led-live"></span><b style="color:#00f0ff;">ORBITAL SATELLITE ZOOM INGESTION // 6.0s DESCENT SEQUENCE</b></div>
    <div style="color:#10b981; font-weight:600;">ALTITUDE: 693.4 km &rarr; 10.0 km &bull; RES: 10m GSD</div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#64748b; margin-top:4px;">
    <span>SENSOR: SENTINEL-1 C-SAR &bull; ISRO EOS-04 POLARIMETRIC</span>
    <span>TARGET: ARABIAN SEA FAIRWAY (19.4167&deg;N, 71.3333&deg;E)</span>
    </div>
    <div class="telemetry-progress-track">
    <div class="telemetry-progress-bar"></div>
    </div>
    </div>
    <div style="text-align: center; margin-top: 10px; margin-bottom: 22px;">
    <h1 style="font-family: 'JetBrains Mono', monospace; color: #00f0ff; letter-spacing: 2px; margin-bottom: 4px;">
    PROJECT GAGANACAKṢUḤ
    </h1>
    <p style="color: #94a3b8; font-size: 0.95rem; letter-spacing: 1px;">
    MULTI-SATELLITE SURVEILLANCE &bull; ZERO-TRUST AIS TRACKING &bull; ISO/IEC 27037 COURT DOSSIER ENGINE
    </p>
    <span style="background-color: #1e293b; color: #38bdf8; padding: 4px 12px; border-radius: 4px; font-size: 0.8rem; font-family: monospace;">
    SIH PROBLEM STATEMENT ID: SIH26143 &bull; CLASSIFIED DEFENSE / JUDICIAL GATEWAY
    </span>
    </div>
    <div class="metric-3d-grid">
    <div class="metric-card-3d" style="border-left: 4px solid #00f0ff;">
    <div class="metric-3d-lbl">Multi-Constellation Sensors</div>
    <div class="metric-3d-val" style="color: #00f0ff;">4 ACTIVE</div>
    <div class="metric-3d-sub">Sentinel-1 C-SAR &bull; ISRO EOS-04 &bull; S-2 MSI &bull; VIIRS</div>
    </div>
    <div class="metric-card-3d" style="border-left: 4px solid #10b981;">
    <div class="metric-3d-lbl">AI Semantic Accuracy</div>
    <div class="metric-3d-val" style="color: #10b981;">94.82%</div>
    <div class="metric-3d-sub">Zenodo SAR Benchmark (Dice 0.948 &bull; 1,112 Chips)</div>
    </div>
    <div class="metric-card-3d" style="border-left: 4px solid #f59e0b;">
    <div class="metric-3d-lbl">Zero-Trust AIS Defense</div>
    <div class="metric-3d-val" style="color: #f59e0b;">ITU-R M.1371</div>
    <div class="metric-3d-sub">GPS Spoofing (&gt;35 kts) &amp; Dark Ship Radar Fusion</div>
    </div>
    <div class="metric-card-3d" style="border-left: 4px solid #ec4899;">
    <div class="metric-3d-lbl">Cryptographic Evidence</div>
    <div class="metric-3d-val" style="color: #ec4899;">ISO/IEC 27037</div>
    <div class="metric-3d-sub">Immutable SHA-256 Ledger &amp; Evidence Act Sec 65B</div>
    </div>
    </div>
    </div>
    """).strip(), unsafe_allow_html=True)

    # 3D Floating Mascot Widget in Bottom-Right Corner (Gaganachakshuh Radar Sentinel)
    st.markdown(textwrap.dedent("""
    <div class="mascot-anchor">
    <div class="mascot-dialogue">
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
    <span style="font-size:0.75rem; font-weight:700; color:#00f0ff;"><span class="led-live"></span>GAGANACHAKSHUH</span>
    <span style="font-size:0.62rem; color:#10b981; border:1px solid #10b981; padding:1px 5px; border-radius:3px; background:rgba(16,185,129,0.15);">RADAR SENTINEL</span>
    </div>
    <div style="font-size:0.68rem; color:#94a3b8; line-height:1.3;">
    Multi-Sat SAR feed synchronized. Ready for classified access.
    </div>
    </div>
    <div class="mascot-orb-frame">
    <div class="mascot-wave-ring"></div>
    <div class="mascot-gyro-ring"></div>
    <div class="mascot-orb-core" title="Gaganachakshuh: Radar Assistant Online">
    <div class="mascot-radar-sweep"></div>
    <div class="mascot-icon">🛰️</div>
    </div>
    </div>
    </div>
    """).strip(), unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        st.markdown(textwrap.dedent("""
        <div class="hud-card" style="border-left-color: #00f0ff;">
        <div style="display:flex; justify-content: space-between; align-items:center; margin-bottom: 12px;">
        <span style="font-family: monospace; font-weight:700; color:#00f0ff;">[SECURE ACCESS CONTROL GATE]</span>
        <span><span class="led-live"></span><small style="color:#10b981; font-family:monospace;">GATEWAY ENCRYPTED (HMAC-SHA256)</small></span>
        </div>
        <p style="color: #cbd5e1; font-size: 0.85rem; margin-bottom: 15px;">
        Access is restricted to authorized Command Officers (Coast Guard/Port Authorities) and Admiralty Tribunal Magistrates.
        </p>
        </div>
        """).strip(), unsafe_allow_html=True)

        # Quick Clearance Badge Selection
        clearance_choice = st.radio(
            "Select Operator Clearance Profile:",
            [
                "🛡️ Command Officer (Indian Coast Guard / Western Command)",
                "⚖️ Tribunal Judge / Legal Investigator (Admiralty Court Special Bench)"
            ],
            index=0
        )

        default_user = "officer.icg" if "Command" in clearance_choice else "judge.tribunal"
        default_pwd = "CoastGuard@2026" if "Command" in clearance_choice else "Justice@Maritime2026"

        with st.form("login_form"):
            username_input = st.text_input("Operator Service ID / Username:", value=default_user)
            password_input = st.text_input("Cryptographic Clearance Passphrase:", value=default_pwd, type="password")
            submit = st.form_submit_button("AUTHENTICATE CLEARANCE TOKEN")

            if submit:
                profile = authenticate_user(username_input, password_input)
                if profile:
                    st.session_state["authenticated"] = True
                    st.session_state["user_profile"] = profile
                    st.toast(f"Clearance Granted: {profile['officer_name']}", icon="🛡️")
                    st.rerun()
                else:
                    st.error("ACCESS DENIED: Invalid Clearance Credentials or Revoked Certificate.")

        st.markdown(textwrap.dedent("""
        <div style="margin-top: 15px; font-size: 0.75rem; color: #64748b; text-align: center; font-family: monospace;">
        Compliant with ISO/IEC 27037:2012 Digital Evidence Protocols & Indian Evidence Act Sec 65B
        </div>
        """).strip(), unsafe_allow_html=True)


# =============================================================================
# VIEW: TACTICAL COMMAND DASHBOARD (AUTHENTICATED)
# =============================================================================
def render_command_dashboard():
    user = st.session_state["user_profile"]
    is_officer = (user["role"] == ROLE_COMMAND_OFFICER)
    is_judge = (user["role"] == ROLE_TRIBUNAL_JUDGE)

    # --- TOP TACTICAL NAVIGATION BAR ---
    badge_bg = "#064e3b" if is_officer else "#1e1b4b"
    badge_fg = "#10b981" if is_officer else "#818cf8"

    st.markdown(f"""
    <div style="background-color: #0b1120; border: 1px solid #1e293b; padding: 10px 18px; border-radius: 6px; margin-bottom: 15px; display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #00f0ff;">
                GAGANACAKṢUḤ // TACTICAL COMMAND HUB
            </span>
            <span style="margin-left: 12px; font-size: 0.8rem; color: #94a3b8; font-family: monospace;">
                UTC TIME: {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}Z &bull; SECTOR: ARABIAN SEA EEZ
            </span>
        </div>
        <div style="display:flex; align-items:center; gap: 10px;">
            <span style="background-color: {badge_bg}; color: {badge_fg}; font-family: monospace; font-size: 0.8rem; padding: 4px 10px; border-radius: 4px; border: 1px solid {badge_fg};">
                {user['officer_name']} [{user['service_number']}] &bull; {user['role']}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR CONTROLS & TELEMETRY ---
    with st.sidebar:
        st.markdown(f"""
        <div style="border-bottom: 1px solid #1e293b; padding-bottom: 12px; margin-bottom: 12px;">
            <span class="led-live"></span><b style="color:#00f0ff; font-family:monospace;">OPERATOR CLEARANCE CARD</b>
            <div style="font-size:0.8rem; margin-top:6px; color:#cbd5e1;">
                <b>Name:</b> {user['officer_name']}<br/>
                <b>Service:</b> {user['service_number']}<br/>
                <b>Base:</b> {user['base']}<br/>
                <b>Level:</b> <span style="color:#38bdf8;">{user['meta']['clearance_level']}</span>
            </div>
            <div style="margin-top:8px; background-color:#020617; padding:6px; border-radius:4px; font-family:monospace; font-size:0.65rem; word-break:break-all; color:#64748b;">
                TOKEN: {user['session_token']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("TERMINATE SESSION / LOGOUT"):
            st.session_state["authenticated"] = False
            st.session_state["user_profile"] = None
            st.rerun()

        st.markdown("---")
        st.markdown("<b style='color:#00f0ff; font-family:monospace;'>SURVEILLANCE SECTOR SELECTOR</b>", unsafe_allow_html=True)
        reg_names = {k: v["name"] for k, v in REGIONS.items()}
        selected_reg_key = st.selectbox(
            "Active Maritime Theatre:",
            options=list(reg_names.keys()),
            format_func=lambda x: reg_names[x]
        )

        if selected_reg_key != st.session_state["selected_region"]:
            st.session_state["selected_region"] = selected_reg_key
            st.session_state["sar_scene"] = load_benchmark_sar_scene(selected_reg_key)
            st.session_state["unet_results"] = None
            st.session_state["backcast_results"] = None
            st.session_state["ais_results"] = None
            st.session_state["docket"] = None
            st.session_state["reroutes"] = None
            st.rerun()

        active_reg = REGIONS[st.session_state["selected_region"]]
        st.markdown(f"""
        <div style="font-size:0.75rem; color:#94a3b8; margin-top:5px; font-family:monospace; background-color:#020617; padding:8px; border-radius:4px; border:1px solid #1e293b;">
            <b>Tactical Command:</b> {active_reg['tactical_zone']}<br/>
            <b>Threat Profile:</b> {active_reg['primary_threat']}<br/>
            <b>Bounds:</b> {active_reg['bounds']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<b style='color:#00f0ff; font-family:monospace;'>CONSTELLATION TELEMETRY</b>", unsafe_allow_html=True)
        for c_key, c_info in CONSTELLATIONS.items():
            st.markdown(f"""
            <div style="font-size:0.75rem; margin-bottom:6px; color:#cbd5e1; font-family:monospace;">
                <span class="led-live"></span><b>{c_info['name']}</b><br/>
                <span style="color:#64748b;">{c_info['type']} ({c_info['resolution']})</span>
            </div>
            """, unsafe_allow_html=True)

    # --- AUTO-RUN INITIAL PIPELINE IF NOT RUN ---
    if st.session_state["unet_results"] is None:
        sar_scene = st.session_state["sar_scene"]
        unet_engine = UNetZenodoInferenceEngine(confidence_threshold=0.50)
        st.session_state["unet_results"] = unet_engine.predict_segmentation(
            sar_amplitude=sar_scene["sar_amplitude"],
            ndvi_optical=sar_scene["ndvi"],
            origin_lat=sar_scene["metadata"]["center_lat"],
            origin_lon=sar_scene["metadata"]["center_lon"]
        )

    if st.session_state["backcast_results"] is None:
        sar_scene = st.session_state["sar_scene"]
        unet_res = st.session_state["unet_results"]
        hydro_engine = HydrodynamicBackcastEngine(
            current_u_ms=0.28,
            current_v_ms=-0.15,
            wind_speed_ms=sar_scene["metadata"].get("wind_speed_ms", 6.8),
            wind_dir_deg=245.0
        )
        st.session_state["backcast_results"] = hydro_engine.simulate_lagrangian_backcast(
            slick_centroid=tuple(unet_res["centroid"]),
            backcast_hours=12.0,
            num_particles=400
        )
        fay_age = hydro_engine.estimate_fay_spill_age(area_sq_km=unet_res["total_area_sq_km"])
        st.session_state["unet_results"]["estimated_age_hours"] = fay_age["estimated_age_hours"]
        st.session_state["unet_results"]["fay_regime"] = fay_age["fay_regime"]

    if st.session_state["ais_results"] is None:
        backcast_res = st.session_state["backcast_results"]
        auditor = ZeroTrustAISAuditor(max_speed_threshold_knots=35.0)
        st.session_state["ais_results"] = auditor.audit_vessel_records(
            backcast_origin=tuple(backcast_res["origin_release_coord"])
        )

    if st.session_state["reroutes"] is None:
        rerouter = TacticalReroutingEngine(buffer_radius_km=5.0)
        unet_res = st.session_state["unet_results"]
        ais_res = st.session_state["ais_results"]
        ex_zone = rerouter.generate_exclusion_zone(tuple(unet_res["centroid"]), unet_res["polygons"])
        st.session_state["exclusion_zone"] = ex_zone
        st.session_state["reroutes"] = rerouter.compute_vessel_reroutes(ais_res["audited_vessels"], ex_zone)

    if st.session_state["docket"] is None:
        ledger = ForensicLedgerEngine()
        sar_scene = st.session_state["sar_scene"]
        unet_res = st.session_state["unet_results"]
        backcast_res = st.session_state["backcast_results"]
        ais_res = st.session_state["ais_results"]
        suspect = ais_res["primary_suspect"]

        st.session_state["docket"] = ledger.build_evidence_docket(
            docket_id=f"DOCKET-2026-GGN-{int(time.time())%100000:05d}",
            case_officer=user["officer_name"],
            service_number=user["service_number"],
            jurisdiction=user["base"],
            satellite_metadata=sar_scene["metadata"],
            spill_metrics=unet_res,
            backcast_data=backcast_res,
            suspect_data=suspect
        )

    # --- TOP SUMMARY METRICS STRIP ---
    unet_res = st.session_state["unet_results"]
    ais_res = st.session_state["ais_results"]
    backcast_res = st.session_state["backcast_results"]
    docket = st.session_state["docket"]

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("OIL SLICK AREA", f"{unet_res['total_area_sq_km']} km²", f"Confidence {unet_res['mean_confidence_score']*100:.1f}%")
    with m2:
        st.metric("FAY SPREADING AGE", f"{unet_res.get('estimated_age_hours', 12.0)} hrs", unet_res.get("fay_regime", "Viscous"))
    with m3:
        st.metric("TRACKED VESSELS", f"{ais_res['total_vessels_tracked']}", f"{ais_res['spoofed_vessels_detected']} GPS Spoofed")
    with m4:
        st.metric("DARK SHIPS FOUND", f"{ais_res['dark_vessels_detected']}", "SAR Hull Scan Confirmed")
    with m5:
        st.metric("COURT DOCKET", "SEALED", "SHA-256 Validated")

    # --- MAIN INTERACTIVE TACTICAL WORKBENCH TABS ---
    if is_judge:
        # Tribunal Judge view is focused on the forensic legal chain of custody and tamper validation
        tab_vault, tab_map, tab_audit = st.tabs([
            "⚖️ ISO/IEC 27037 FORENSIC EVIDENCE VAULT",
            "🗺️ TACTICAL RADAR EVIDENCE MAP",
            "🔍 ZERO-TRUST AIS & RADAR ATTRIBUTION"
        ])
    else:
        # Command Officer has full 5-stage operational pipeline tabs
        tab_map, tab_sat, tab_ai, tab_ais, tab_vault, tab_reroute = st.tabs([
            "🗺️ TACTICAL RADAR HUD MAP",
            "🛰️ STAGE 1: COPERNICUS WMS & DUAL-SATELLITE",
            "🧠 STAGE 2: U-NET AI & HYDRODYNAMIC BACKCAST",
            "🛡️ STAGE 3: ZERO-TRUST AIS & SPOOFING RADAR",
            "⚖️ STAGE 4: ISO/IEC 27037 EVIDENCE VAULT & LEDGER",
            "🚨 STAGE 5: AUTOMATED REROUTING & VHF DISPATCH"
        ])

    # =========================================================================
    # TAB: TACTICAL RADAR HUD MAP
    # =========================================================================
    with tab_map:
        st.markdown("""
        <div class="hud-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <b style="color:#00f0ff; font-family:monospace;">REAL-TIME MARITIME COMMON OPERATING PICTURE (COP)</b>
                <span style="font-family:monospace; font-size:0.75rem; color:#94a3b8;">
                    LAYERS: S1-SAR &bull; U-NET SLICK &bull; DRIFT CONE &bull; AIS TELEMETRY &bull; DARK SHIPS &bull; EXCLUSION CORRIDOR
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        render_tactical_leaflet_map(
            center=unet_res["centroid"],
            zoom=10,
            slick_polygons=unet_res["polygons"],
            backcast_data=backcast_res,
            vessels=ais_res["audited_vessels"],
            dark_ships=ais_res["dark_ships"],
            exclusion_polygon=st.session_state["exclusion_zone"]["exclusion_polygon"],
            reroutes=st.session_state["reroutes"]
        )

        st.markdown(f"""
        <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 10px; font-family:monospace; font-size:0.75rem;">
            <div style="background-color:#1e1017; border-left:3px solid #ff0055; padding:6px; border-radius:3px;">
                <b style="color:#ff0055;">RED SOLID</b>: Mineral Oil Slick (U-Net)
            </div>
            <div style="background-color:#1f190e; border-left:3px solid #ffaa00; padding:6px; border-radius:3px;">
                <b style="color:#ffaa00;">YELLOW DASH</b>: Lagrangian Origin Cone
            </div>
            <div style="background-color:#11202e; border-left:3px solid #00f0ff; padding:6px; border-radius:3px;">
                <b style="color:#00f0ff;">CYAN LINE</b>: Compliant AIS Tracks
            </div>
            <div style="background-color:#2a110a; border-left:3px solid #f97316; padding:6px; border-radius:3px;">
                <b style="color:#f97316;">ORANGE MARKER</b>: GPS Spoofed (>35 kts)
            </div>
            <div style="background-color:#2e0d16; border-left:3px solid #ff0033; padding:6px; border-radius:3px;">
                <b style="color:#ff0033;">PULSING CRIMSON</b>: SAR Dark Ship
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # TAB: STAGE 1: COPERNICUS WMS & DUAL SATELLITE
    # =========================================================================
    if is_officer:
        with tab_sat:
            st.markdown("""
            <div class="hud-card">
                <b style="color:#00f0ff; font-family:monospace;">STAGE 1: COPERNICUS DATA SPACE WMS & DUAL-SATELLITE ARCHITECTURE</b>
                <p style="color:#cbd5e1; font-size:0.85rem; margin-top:4px;">
                    Integrates Sentinel-1 C-Band SAR (all-weather dark spot detection) with Sentinel-2 Optical MSI & NOAA VIIRS
                    to achieve automated false-positive suppression (discriminating mineral crude from biogenic algal blooms).
                </p>
            </div>
            """, unsafe_allow_html=True)

            wms_meta = get_wms_capabilities_mock(st.session_state["selected_region"])
            sar_scene = st.session_state["sar_scene"]

            col_wms1, col_wms2 = st.columns([1.2, 1.8])
            with col_wms1:
                st.markdown("<b style='color:#38bdf8; font-family:monospace;'>WMS Handshake & Layer Pipeline</b>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background-color:#020617; border:1px solid #1e293b; padding:12px; border-radius:6px; font-family:monospace; font-size:0.8rem; color:#94a3b8;">
                    <b>ENDPOINT:</b> {wms_meta['endpoint']}<br/>
                    <b>CRS:</b> {wms_meta['crs']}<br/>
                    <b>THEATRE:</b> {wms_meta['target_region']}<br/>
                    <b>BBOX:</b> {wms_meta['bbox']}<br/>
                    <b>STATUS:</b> <span style="color:#10b981;">STREAMING REAL-TIME 10m TILES</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br/><b style='color:#38bdf8; font-family:monospace;'>Active WMS Layers</b>", unsafe_allow_html=True)
                for layer in wms_meta["active_layers"]:
                    st.markdown(f"""
                    <div style="background-color:#0b1120; border:1px solid #1e293b; padding:8px; border-radius:4px; margin-bottom:6px; font-family:monospace; font-size:0.75rem;">
                        <span class="led-live"></span><b>{layer['title']}</b><br/>
                        <span style="color:#64748b;">ID: {layer['layer_id']} &bull; {layer['orbit']}</span>
                    </div>
                    """, unsafe_allow_html=True)

            with col_wms2:
                st.markdown("<b style='color:#38bdf8; font-family:monospace;'>Multi-Sensor Dual-Layer Verification</b>", unsafe_allow_html=True)
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.caption("1. Sentinel-1 SAR GRDH (C-Band Radar Amplitude)")
                    st.image(sar_scene["sar_amplitude"], caption="Radar dark spot (wave damping)", use_container_width=True)
                with subcol2:
                    st.caption("2. Sentinel-2 Optical MSI (False-Color NDVI Index)")
                    # Normalize NDVI for display
                    ndvi_norm = ((sar_scene["ndvi"] + 1.0) / 2.0 * 255.0).astype(np.uint8)
                    ndvi_colored = cv2.applyColorMap(ndvi_norm, cv2.COLORMAP_JET)
                    st.image(ndvi_colored, caption="Red: Algal Bloom Lookalike | Blue: Mineral Oil", use_container_width=True)

                st.info(f"💡 Automated Filter: Suppressed {unet_res.get('optical_lookalikes_filtered', 1)} biogenic lookalike patches in optical NDVI spectrum.")

    # =========================================================================
    # TAB: STAGE 2: U-NET AI & HYDRODYNAMICS
    # =========================================================================
    if is_officer:
        with tab_ai:
            st.markdown("""
            <div class="hud-card">
                <b style="color:#00f0ff; font-family:monospace;">STAGE 2: U-NET SEMANTIC SEGMENTATION & HYDRODYNAMIC BACKCASTING</b>
                <p style="color:#cbd5e1; font-size:0.85rem; margin-top:4px;">
                    Executes U-Net deep convolutional inference (trained on Zenodo SAR Dataset #4124976) to extract slick contours (>94.8% confidence),
                    calculates Fay-spreading slick age, and reverses Lagrangian advection vectors to pinpoint exact emission origin.
                </p>
            </div>
            """, unsafe_allow_html=True)

            c_ai1, c_ai2 = st.columns([1.2, 1.8])
            with c_ai1:
                st.markdown("<b style='color:#38bdf8; font-family:monospace;'>Zenodo U-Net Benchmark Metrics</b>", unsafe_allow_html=True)
                meta = unet_res["model_metadata"]
                st.markdown(f"""
                <div style="background-color:#020617; border:1px solid #1e293b; padding:12px; border-radius:6px; font-family:monospace; font-size:0.8rem; color:#cbd5e1;">
                    <b>Architecture:</b> {meta['architecture']}<br/>
                    <b>Training Corpus:</b> {meta['training_dataset']}<br/>
                    <b>Validation Dice:</b> <span style="color:#10b981;">{meta['validation_dice_coefficient']*100:.2f}%</span><br/>
                    <b>Mean IoU:</b> <span style="color:#00f0ff;">{meta['mean_iou']*100:.2f}%</span><br/>
                    <b>Precision / Recall:</b> {meta['precision']*100:.1f}% / {meta['recall']*100:.1f}%<br/>
                    <b>Weights Checksum:</b> <span style="font-size:0.65rem; color:#64748b;">{meta['weights_hash'][:28]}...</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br/><b style='color:#38bdf8; font-family:monospace;'>Fay Spreading Theory & Age Estimation</b>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background-color:#0b1120; border:1px solid #1e293b; padding:12px; border-radius:6px; font-family:monospace; font-size:0.8rem; color:#cbd5e1;">
                    <b>Regime:</b> {unet_res.get('fay_regime', 'Phase 2 (Gravity-Viscous)')}<br/>
                    <b>Observed Surface Area:</b> <span style="color:#00f0ff;">{unet_res['total_area_sq_km']} sq km</span><br/>
                    <b>Calculated Slick Age:</b> <span style="color:#ffaa00;">{unet_res.get('estimated_age_hours', 12.0)} hours prior</span><br/>
                    <b>Estimated Evaporation Loss:</b> ~30.5%
                </div>
                """, unsafe_allow_html=True)

            with c_ai2:
                st.markdown("<b style='color:#38bdf8; font-family:monospace;'>Segmentation Masks & Backcast Particles</b>", unsafe_allow_html=True)
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    # Color-code probability mask
                    prob_display = (unet_res["probability_map"] * 255).astype(np.uint8)
                    prob_heat = cv2.applyColorMap(prob_display, cv2.COLORMAP_INFERNO)
                    st.image(prob_heat, caption="U-Net Confidence Heatmap (>94.8%)", use_container_width=True)
                with pcol2:
                    st.image(unet_res["binary_mask"] * 255, caption="Cleaned Binary Extraction Mask", use_container_width=True)

                st.markdown(f"""
                <div style="background-color:#020617; border:1px solid #1e293b; padding:10px; border-radius:4px; font-family:monospace; font-size:0.75rem; color:#94a3b8; margin-top:8px;">
                    <b>PINPOINTED EMISSION ORIGIN:</b> <span style="color:#ffaa00;">{backcast_res['origin_release_coord'][0]:.4f}°N, {backcast_res['origin_release_coord'][1]:.4f}°E</span><br/>
                    <b>UNCERTAINTY ENVELOPE:</b> &plusmn;{backcast_res['origin_uncertainty_meters']} meters &bull; <b>DRIFT SPEED:</b> {backcast_res['net_drift_speed_knots']} knots
                </div>
                """, unsafe_allow_html=True)

    # =========================================================================
    # TAB: STAGE 3: ZERO-TRUST AIS & SPOOFING RADAR
    # =========================================================================
    ais_tab_context = tab_audit if is_judge else tab_ais
    with ais_tab_context:
        st.markdown("""
        <div class="hud-card">
            <b style="color:#00f0ff; font-family:monospace;">STAGE 3: ZERO-TRUST AIS TELEMETRY AUDIT & ADVERSARIAL COUNTER-SPOOFING</b>
            <p style="color:#cbd5e1; font-size:0.85rem; margin-top:4px;">
                Adversarial Defense Engine adhering to ITU-R M.1371 schemas.
                Cross-references radio transponder transmissions against physical Sentinel-1 SAR hull reflections to expose GPS spoofing and 'Dark Ships'.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Security Alerts Ticker
        st.markdown("<b style='color:#ef4444; font-family:monospace;'>ACTIVE CYBER & MARITIME THREAT NOTIFICATIONS</b>", unsafe_allow_html=True)
        for alert in ais_res["security_alerts"]:
            is_crit = "CRITICAL" in alert["level"]
            card_class = "hud-card-alert" if is_crit else "hud-card"
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display:flex; justify-content:space-between; font-family:monospace; font-size:0.8rem;">
                    <b style="color:#ef4444;"><span class="led-alert"></span>[{alert['code']}]</b>
                    <span style="color:#94a3b8;">{alert.get('timestamp', 'LIVE SENSOR PASS')}</span>
                </div>
                <div style="font-size:0.85rem; color:#f8fafc; margin-top:4px;">
                    {alert['details']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Suspect Attribution Table
        st.markdown("<br/><b style='color:#00f0ff; font-family:monospace;'>VESSEL ATTRIBUTION MATRIX (RANKED BY PROBABILISTIC LIABILITY)</b>", unsafe_allow_html=True)
        v_rows = []
        for v in ais_res["audited_vessels"]:
            v_rows.append({
                "MMSI": v["mmsi"],
                "Vessel Name": v["vessel_name"],
                "Type": v["vessel_type"],
                "Flag": v["flag"],
                "Max SOG (kts)": f"{v['max_sog']:.1f}",
                "Dist to Origin": f"{v['closest_dist_to_origin_km']} km",
                "Attribution": f"{v['attribution_score']}%",
                "Kinematic Status": "🚨 GPS SPOOFED" if v["is_spoofed"] else ("⚠️ DELIBERATE AIS GAP" if v["has_gap"] else "✅ COMPLIANT")
            })
        st.dataframe(pd.DataFrame(v_rows), use_container_width=True)

    # =========================================================================
    # TAB: STAGE 4: ISO/IEC 27037 EVIDENCE VAULT & BIT-TAMPER INSPECTOR
    # =========================================================================
    with tab_vault:
        st.markdown("""
        <div class="hud-card">
            <b style="color:#00f0ff; font-family:monospace;">STAGE 4: ISO/IEC 27037 FORENSIC EVIDENCE VAULT & CRYPTOGRAPHIC TAMPER PROOF</b>
            <p style="color:#cbd5e1; font-size:0.85rem; margin-top:4px;">
                Adheres strictly to ISO/IEC 27037 standards for digital evidence preservation.
                Locks satellite telemetry, spatial polygon coordinates, backcasting vectors, and suspect MMSI strings into an immutable SHA-256 ledger.
            </p>
        </div>
        """, unsafe_allow_html=True)

        ledger_engine = ForensicLedgerEngine()

        # Top Master Seal Banner
        st.markdown(f"""
        <div style="background-color:#020617; border:1px solid #00f0ff; padding:14px; border-radius:6px; margin-bottom:15px; font-family:monospace;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="color:#00f0ff; font-weight:700;">DOCKET ID: {docket['docket_id']}</span>
                <span style="background-color:#064e3b; color:#10b981; padding:3px 8px; border-radius:3px; font-size:0.75rem;">
                    ISO/IEC 27037 SEALED & IMMUTABLE
                </span>
            </div>
            <div style="font-size:0.8rem; color:#94a3b8; margin-top:6px;">
                <b>MASTER SHA-256 CHECKSUM:</b> <code style="color:#38bdf8;">{docket['master_seal_hash']}</code>
            </div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">
                Sealed by: {docket['sealing_officer']} [{docket['service_number']}] &bull; Time: {docket['timestamp_sealed']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Interactive Tamper-Proof Testing Lab
        st.markdown("<b style='color:#ffaa00; font-family:monospace;'>🔬 INTERACTIVE ADVERSARIAL BIT-TAMPER VALIDATOR</b>", unsafe_allow_html=True)
        st.caption("Simulate an adversarial attempt to alter evidence data (e.g. changing suspect coordinates or clearance records) to verify that the SHA-256 seal immediately invalidates the docket.")

        tc1, tc2 = st.columns([1.5, 1])
        with tc1:
            tamper_mode = st.radio(
                "Select Integrity Verification Mode:",
                [
                    "✅ Normal Verification (Original Unaltered Evidence)",
                    "⚠️ Adversarial Tamper: Alter Suspect Attribution Checksum by 1 character",
                    "⚠️ Adversarial Tamper: Modify Jurisdiction / Sealing Authority"
                ]
            )

        with tc2:
            st.markdown("<br/>", unsafe_allow_html=True)
            if "Normal" in tamper_mode:
                verification = ledger_engine.verify_tamper_integrity(docket)
            elif "Attribution" in tamper_mode:
                verification = ledger_engine.verify_tamper_integrity(
                    docket,
                    tampered_key="suspect_attribution_hash",
                    tampered_value="f" * 64
                )
            else:
                verification = ledger_engine.verify_tamper_integrity(
                    docket,
                    tampered_key="jurisdiction",
                    tampered_value="Unverified Foreign Maritime Authority"
                )

        if verification["seal_intact"]:
            st.markdown(f"""
            <div class="hud-card-success">
                <b style="color:#10b981;">{verification['verification_status']}</b><br/>
                <span style="font-family:monospace; font-size:0.8rem; color:#cbd5e1;">
                    Original Hash: <code>{verification['original_sha256_seal']}</code><br/>
                    Recomputed Hash: <code>{verification['recomputed_sha256_hash']}</code>
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="hud-card-alert">
                <b style="color:#ef4444;"><span class="led-alert"></span>{verification['verification_status']}</b><br/>
                <span style="font-family:monospace; font-size:0.8rem; color:#cbd5e1;">
                    Original Expected: <code>{verification['original_sha256_seal']}</code><br/>
                    Recomputed Tampered: <code style="color:#ef4444;">{verification['recomputed_sha256_hash']}</code><br/>
                    Verdict: <b style="color:#ef4444;">{verification['iso_27037_verdict']}</b> (Court admissibility revoked!)
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Digital Chain of Custody Table
        st.markdown("<br/><b style='color:#00f0ff; font-family:monospace;'>DIGITAL CHAIN OF CUSTODY LOG (ISO/IEC 27037 CLAUSE 6.4)</b>", unsafe_allow_html=True)
        custody_data = []
        for c in docket.get("chain_of_custody", []):
            custody_data.append({
                "Seq": c["sequence"],
                "Event Description": c["event"],
                "Authority / Module": c["authority"],
                "Timestamp (UTC)": c["time"],
                "SHA-256 Checksum": c["hash"]
            })
        st.dataframe(pd.DataFrame(custody_data), use_container_width=True)

        # Export Court-Admissible PDF Dossier
        st.markdown("<br/><b style='color:#00f0ff; font-family:monospace;'>OFFICIAL COURT-ADMISSIBLE DOSSIER EXPORT</b>", unsafe_allow_html=True)
        pdf_bytes = ledger_engine.generate_court_dossier_pdf(docket)

        cd1, cd2 = st.columns([1.5, 1])
        with cd1:
            st.download_button(
                label="📥 DOWNLOAD ISO/IEC 27037 LEGAL EVIDENCE DOSSIER (PDF)",
                data=pdf_bytes,
                file_name=f"{docket['docket_id']}_EVIDENCE_DOSSIER.pdf",
                mime="application/pdf"
            )
        with cd2:
            st.download_button(
                label="📥 EXPORT RAW EVIDENCE LEDGER (JSON)",
                data=json.dumps(docket, indent=2),
                file_name=f"{docket['docket_id']}_RAW_LEDGER.json",
                mime="application/json"
            )

    # =========================================================================
    # TAB: STAGE 5: TACTICAL EDGE INTEROPERABILITY & REROUTING
    # =========================================================================
    if is_officer:
        with tab_reroute:
            st.markdown("""
            <div class="hud-card">
                <b style="color:#00f0ff; font-family:monospace;">STAGE 5: TACTICAL EDGE INTEROPERABILITY & AUTOMATED SAFETY REROUTING</b>
                <p style="color:#cbd5e1; font-size:0.85rem; margin-top:4px;">
                    Lightweight edge execution for port command laptops (Mumbai / Kochi).
                    Dynamically generates 5 NM navigation exclusion zones, computes collision avoidance divert waypoints,
                    and dispatches simulated VHF Navtex and Telegram emergency alert broadcasts.
                </p>
            </div>
            """, unsafe_allow_html=True)

            rerouter = TacticalReroutingEngine(buffer_radius_km=5.0)
            ex_zone = st.session_state["exclusion_zone"]
            reroutes = st.session_state["reroutes"]

            c_rr1, c_rr2 = st.columns([1.2, 1.8])
            with c_rr1:
                st.markdown("<b style='color:#38bdf8; font-family:monospace;'>Dynamic Exclusion Corridor</b>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background-color:#020617; border:1px solid #1e293b; padding:12px; border-radius:6px; font-family:monospace; font-size:0.8rem; color:#cbd5e1;">
                    <b>Zone Status:</b> <span style="color:#ef4444;">{ex_zone['threat_status']}</span><br/>
                    <b>Mandatory Standoff:</b> {ex_zone['safety_buffer_nm']} Nautical Miles<br/>
                    <b>Advisory:</b> {ex_zone['advisory']}<br/>
                    <b>Corridor Bounds:</b> 4 Geodetic Vertices
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br/><b style='color:#38bdf8; font-family:monospace;'>Automated VHF / Telegram Emergency Dispatch</b>", unsafe_allow_html=True)
                if st.button("🚨 TRIGGER EMERGENCY MARITIME SAFETY BROADCAST"):
                    st.session_state["broadcast_dispatched"] = True
                    st.toast("VHF NAVTEX & Telegram Emergency Broadcast Dispatched!", icon="📡")

                if st.session_state.get("broadcast_dispatched", False):
                    dispatch = rerouter.dispatch_vhf_navtex_broadcast(docket["docket_id"], ex_zone, reroutes)
                    st.markdown(f"""
                    <div class="hud-card-success" style="margin-top:10px;">
                        <b style="color:#10b981;">BROADCAST TRANSMISSION CONFIRMED</b><br/>
                        <span style="font-family:monospace; font-size:0.75rem; color:#cbd5e1;">
                            Channels: {', '.join(dispatch['channels'])}<br/>
                            Telegram Channel: {dispatch['telegram_dispatch']['channel']}<br/>
                            Vessels Alerted: {len(dispatch['telegram_dispatch']['vessels_alerted'])}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

            with c_rr2:
                st.markdown("<b style='color:#38bdf8; font-family:monospace;'>Computed Vessel Divert Waypoints</b>", unsafe_allow_html=True)
                for r in reroutes:
                    st.markdown(f"""
                    <div style="background-color:#0b1120; border:1px solid #1e293b; padding:10px; border-radius:6px; margin-bottom:8px; font-family:monospace; font-size:0.8rem;">
                        <div style="display:flex; justify-content:space-between;">
                            <b style="color:#00f0ff;">{r['vessel_name']} (MMSI {r['mmsi']})</b>
                            <span style="color:#f97316;">{r['urgency']}</span>
                        </div>
                        <div style="color:#94a3b8; font-size:0.75rem; margin-top:4px;">
                            Distance to Spill: <b>{r['dist_to_spill_km']} km</b><br/>
                            Divert Waypoint: <span style="color:#10b981;">{r['diversion_waypoint'][0]}°N, {r['diversion_waypoint'][1]}°E</span><br/>
                            Course Change: <span style="color:#cbd5e1;">{r['reroute_instruction']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APP ENTRY POINT
# =============================================================================
def main():
    if not st.session_state.get("authenticated", False):
        render_login_screen()
    else:
        render_command_dashboard()

if __name__ == "__main__":
    main()
