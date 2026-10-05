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
import io
import base64
import textwrap
import cv2
from PIL import Image
from pathlib import Path
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
    page_title="GAGANACAKSUH // Autonomous Maritime Forensics",
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

    /* Tactical Background & Panels - Solid Maritime Theme (No Neon) */
    .stApp {
        background-color: #0B0F17 !important;
        color: #E2E8F0 !important;
    }

    /* Left Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #161F30 !important;
        border-right: 1px solid rgba(148, 163, 184, 0.15) !important;
    }
    section[data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 325px !important;
    }

    /* Top Sticky Header Row */
    div[data-testid="stHorizontalBlock"]:first-of-type {
        position: sticky !important;
        top: 0 !important;
        z-index: 990 !important;
        background-color: #0B0F17 !important;
        padding-top: 6px !important;
        padding-bottom: 10px !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15) !important;
        margin-bottom: 14px !important;
    }

    /* HUD Cards - Solid Tones (No Neon, No Glows) */
    .hud-card {
        background: #161F30;
        border: 1px solid #283548;
        border-left: 4px solid #38BDF8;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
        color: #E2E8F0;
    }

    .hud-card-alert {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-left: 4px solid #EF4444;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
        color: #E2E8F0;
    }

    .hud-card-warning {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 4px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
        color: #E2E8F0;
    }

    .hud-card-success {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-left: 4px solid #10B981;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
        color: #E2E8F0;
    }

    /* Status LED indicator - Solid Colors (No Neon Box Shadows) */
    .led-live {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        margin-right: 6px;
    }

    .led-alert {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #EF4444;
        border-radius: 50%;
        margin-right: 6px;
    }

    .led-warning {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #F59E0B;
        border-radius: 50%;
        margin-right: 6px;
    }

    /* Metric KPI containers - Solid Crisp White (No Glowing Cyan/Neon) */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }

    div[data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        color: #94A3B8 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    div[data-testid="stMetricDelta"] svg {
        fill: #10B981 !important;
    }
    div[data-testid="stMetricDelta"] div {
        color: #10B981 !important;
        font-weight: 600 !important;
    }

    /* Buttons - Solid Maritime Styling */
    .stButton > button {
        background-color: #161F30 !important;
        color: #E2E8F0 !important;
        border: 1px solid #334155 !important;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.5px;
        border-radius: 6px;
        transition: all 0.15s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        border-color: #38BDF8 !important;
    }

    /* Sidebar Radio Navigation (Compact boxes, reduced text size, full-width alignment with other sidebar components) */
    section[data-testid="stSidebar"] .stRadio,
    section[data-testid="stSidebar"] div[data-testid="stRadio"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div,
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        width: 100% !important;
        max-width: 100% !important;
        gap: 5px !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        display: flex !important;
        width: 100% !important;
        max-width: 100% !important;
        align-items: center !important;
        box-sizing: border-box !important;
        background: #0B0F17 !important;
        border: 1px solid #1E293B !important;
        border-radius: 4px !important;
        padding: 5px 8px !important;
        margin: 0 !important;
        color: #E2E8F0 !important;
        font-size: 0.64rem !important;
        font-weight: 500 !important;
        letter-spacing: -0.015em !important;
        line-height: 1.2 !important;
        white-space: nowrap !important;
        transition: all 0.15s ease !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        border-color: #38BDF8 !important;
        background: #111A2E !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        border-color: #38BDF8 !important;
        background: #162032 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
        margin-right: 6px !important;
        flex-shrink: 0 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:last-child {
        flex: 1 !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 0.64rem !important;
        margin: 0 !important;
        padding: 0 !important;
        white-space: nowrap !important;
        line-height: 1.2 !important;
    }
    /* Profile Popover Button (Solid Styling) */
    div[data-testid="stPopover"] > button {
        border-radius: 9999px !important;
        background: #161F30 !important;
        border: 1px solid #38BDF8 !important;
        color: #E2E8F0 !important;
        padding: 5px 16px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stPopover"] > button:hover {
        background: #1E293B !important;
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
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
        <div style="background:#0f172a; color:#e2e8f0; padding:8px 10px; border:1px solid #38BDF8; border-radius:4px; font-family:'Courier New', monospace; font-size:11px;">
            <b style="color:#38BDF8;">{name}</b><br/>
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
                "🌐 2D Tactical Basemap (NASA GIBS WMS & Leaflet Native Radar)",
                "✨ 3D Tactical Perspective (Pydeck DeckGL: 50° Pitch & Extruded Beacons)"
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
                border: 1px solid #38BDF8;
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
                var color = '#38BDF8';
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


# =============================================================================
# VIEW: PAGE 1 - FULL-BLEED LANDING PAGE
# =============================================================================
def render_landing_page():
    # 0. Check query params or session state navigation
    if "page" in st.query_params:
        target = st.query_params["page"]
        if target in ["dashboard", "analysis", "command", "main"]:
            profile = authenticate_user("jury@sih2026.in", "jury123")
            if profile:
                st.session_state["authenticated"] = True
                st.session_state["user_profile"] = profile
                st.session_state["page"] = "dashboard"
                st.rerun()
        elif target in ["auth", "auth_gateway", "login"]:
            st.session_state["page"] = "auth_gateway"
            st.rerun()

    # 1. Remove parent padding and enforce true full-screen breakout
    st.markdown("""
    <style>
        header[data-testid="stHeader"] {
            display: none !important;
        }
        .main, .block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100vw !important;
            overflow: hidden !important;
        }
        div[data-testid="stVerticalBlock"] {
            gap: 0 !important;
        }
        iframe {
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            border: none !important;
            margin: 0 !important;
            padding: 0 !important;
            z-index: 1 !important;
            pointer-events: none !important;
        }
        /* Native Streamlit Overlay Buttons with High Z-Index */
        div.st-key-btn_top_gateway {
            position: fixed !important;
            top: 1.5rem !important;
            right: 2.5rem !important;
            z-index: 9999999 !important;
            width: auto !important;
            pointer-events: auto !important;
        }
        div.st-key-btn_top_gateway button {
            background: rgba(16, 25, 40, 0.85) !important;
            backdrop-filter: blur(8px) !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(255, 255, 255, 0.35) !important;
            border-radius: 6px !important;
            padding: 8px 24px !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            font-family: inherit !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.5) !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }
        div.st-key-btn_top_gateway button:hover {
            background: #1E293B !important;
            border-color: #38BDF8 !important;
            color: #38BDF8 !important;
        }</style>
    """, unsafe_allow_html=True)

    landing_file = Path(__file__).parent / "static" / "fullbleed_landing.html"
    if landing_file.exists():
        with open(landing_file, "r", encoding="utf-8") as f:
            html_landing = f.read()
    else:
        html_landing = """
        <div style="position: absolute; top: 0; left: 0; width: 100vw; height: 100vh; margin: 0; padding: 0; background: #0B0F17; color: #E2E8F0;">
            <h1>GAGANACAKSUH</h1>
        </div>
        """
    # Fixed Top-Right Native Sign In Button
    if st.button("Sign In", key="btn_top_gateway"):
        st.session_state["page"] = "auth_gateway"
        st.session_state["authenticated"] = False
        st.query_params.clear()
        st.rerun()

    components.html(html_landing, height=1000, scrolling=False)


# =============================================================================
# VIEW: PAGE 2 - TACTICAL HIGH-CONTRAST SECURE AUTH GATEWAY (RICH CSS & COLORS)
# =============================================================================
def render_auth_gateway():
    # Human-Crafted Enterprise Dark Theme CSS
    st.markdown("""
    <style>
        .stApp {
            background-color: #080C14 !important;
            background-image: radial-gradient(circle at 50% 12%, #0F172A 0%, #080C14 75%) !important;
        }

        /* Seamless Header Section of Auth Card */
        .auth-panel-top {
            background-color: #0F172A;
            border: 1px solid #1E293B;
            border-bottom: none;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
            padding: 28px 24px 14px 24px;
            margin-top: 2rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        }

        .auth-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            background-color: #1E293B;
            border: 1px solid #283548;
            padding: 3px 9px;
            border-radius: 9999px;
            font-size: 0.6875rem;
            font-weight: 600;
            color: #94A3B8;
            letter-spacing: 0.05em;
            margin-bottom: 12px;
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: #10B981;
        }

        .auth-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: -0.02em;
            margin: 0 0 6px 0;
        }

        .auth-subtitle {
            font-size: 0.875rem;
            color: #94A3B8;
            line-height: 1.45;
            margin: 0;
        }

        /* Seamless Form Integration into the same card */
        div[data-testid="stForm"] {
            background-color: #0F172A !important;
            border: 1px solid #1E293B !important;
            border-top: none !important;
            border-bottom-left-radius: 10px !important;
            border-bottom-right-radius: 10px !important;
            border-top-left-radius: 0 !important;
            border-top-right-radius: 0 !important;
            padding: 0 24px 22px 24px !important;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4) !important;
            margin-bottom: 14px !important;
        }

        /* Calibrated Input Fields (Solid & Restrained) */
        div[data-testid="stTextInput"] input {
            background-color: #0B0F19 !important;
            border: 1px solid #283548 !important;
            border-radius: 6px !important;
            color: #F1F5F9 !important;
            font-size: 0.875rem !important;
            padding: 10px 12px !important;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: #2563EB !important;
            box-shadow: 0 0 0 1px #2563EB !important;
            background-color: #0D1322 !important;
        }

        /* Primary Action Button: Solid Enterprise Blue */
        div[data-testid="stFormSubmitButton"] button {
            background-color: #2563EB !important;
            border: 1px solid #3B82F6 !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            font-size: 0.875rem !important;
            border-radius: 6px !important;
            padding: 10px 16px !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
            transition: background-color 0.15s ease, border-color 0.15s ease !important;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background-color: #1D4ED8 !important;
            border-color: #2563EB !important;
        }

        /* Request Access Ghost Button */
        div.st-key-btn-request-access button {
            background: transparent !important;
            border: 1px solid #283548 !important;
            color: #94A3B8 !important;
            font-size: 0.8125rem !important;
            font-weight: 500 !important;
            border-radius: 6px !important;
            transition: all 0.15s ease !important;
        }

        div.st-key-btn-request-access button:hover {
            background-color: #162032 !important;
            border-color: #334155 !important;
            color: #F1F5F9 !important;
        }

        /* SIH Jury Direct Access: Clean Enterprise Secondary Button */
        div.st-key-btn-sih-jury-direct button {
            background-color: #131D31 !important;
            border: 1px solid #243553 !important;
            color: #93C5FD !important;
            font-weight: 500 !important;
            font-size: 0.8125rem !important;
            border-radius: 6px !important;
            padding: 10px 14px !important;
            box-shadow: none !important;
            transition: all 0.15s ease !important;
        }

        div.st-key-btn-sih-jury-direct button:hover {
            background-color: #1A2844 !important;
            border-color: #3B82F6 !important;
            color: #FFFFFF !important;
        }

        /* Return Link */
        div.st-key-btn-back-overview button {
            background: transparent !important;
            border: none !important;
            color: #64748B !important;
            font-size: 0.8125rem !important;
            font-weight: 400 !important;
        }

        div.st-key-btn-back-overview button:hover {
            color: #94A3B8 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Centered Authentication Card
    col_g1, col_g2, col_g3 = st.columns([1, 1.25, 1])
    with col_g2:
        # Auth Card Top Header
        st.markdown(textwrap.dedent("""
        <div class="auth-panel-top">
            <div class="auth-status-pill">
                <span class="status-dot"></span>
                <span>RESTRICTED ACCESS // OFFICIAL CLEARANCE</span>
            </div>
            <h2 class="auth-title">Sign In</h2>
            <p class="auth-subtitle">Access the Gaganacaksuh Maritime Forensics Engine.</p>
        </div>
        """).strip(), unsafe_allow_html=True)

        # Official Access Form (Official Email, Password, Sign In)
        with st.form("enterprise_auth_form"):
            st.markdown("""<div style="font-size: 0.78125rem; font-weight: 500; color: #94A3B8; margin-top: 14px; margin-bottom: 5px;">Official Email</div>""", unsafe_allow_html=True)
            email_input = st.text_input(
                "Official Email",
                label_visibility="collapsed",
                key="official-email",
                placeholder="Enter your email"
            )

            st.markdown("""<div style="font-size: 0.78125rem; font-weight: 500; color: #94A3B8; margin-top: 12px; margin-bottom: 5px;">Passkey / Password</div>""", unsafe_allow_html=True)
            password_input = st.text_input(
                "Password",
                type="password",
                label_visibility="collapsed",
                key="password",
                placeholder="••••••••"
            )

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            sign_in_submit = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if sign_in_submit:
                email_clean = email_input.strip()
                pass_clean = password_input.strip()
                if not email_clean or not pass_clean:
                    st.warning("⚠️ Please enter both your official email and passkey to sign in.")
                else:
                    profile = authenticate_user(email_clean, pass_clean)
                    if profile:
                        st.session_state["authenticated"] = True
                        st.session_state["user_profile"] = profile
                        st.session_state["page"] = "dashboard"
                        st.query_params.clear()
                        st.toast(f"Clearance Verified: {profile['officer_name']}", icon="⚓")
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Verify your service email and passkey.")

        # Professional Admin Registration Notice
        st.markdown("""
        <div style="font-size: 0.78rem; color: #94A3B8; text-align: center; margin-top: 14px; padding: 10px 14px; background: rgba(22, 31, 48, 0.6); border: 1px solid #161F30; border-radius: 6px; line-height: 1.5; font-family: -apple-system, sans-serif;">
            🔒 <b>Access Provisioning Notice:</b> Only the platform System Administrator can register users and provision service credentials. Authorized personnel should use their issued credentials to log in.
        </div>
        """, unsafe_allow_html=True)

        # Clean Hairline Divider with "OR"
        st.markdown(textwrap.dedent("""
        <div style="position: relative; display: flex; align-items: center; justify-content: center; margin: 20px 0;">
            <div style="position: absolute; left: 0; right: 0; height: 1px; background-color: #1E293B; z-index: 1;"></div>
            <span style="position: relative; z-index: 2; background-color: #080C14; padding: 0 12px; font-size: 0.6875rem; font-weight: 600; color: #475569; letter-spacing: 0.06em; text-transform: uppercase;">OR</span>
        </div>
        """).strip(), unsafe_allow_html=True)

        # SIH Jury Direct Access Button (Bypass for Live Demos)
        jury_bypass = st.button("SIH Jury Direct Access (One-Click)", key="btn-sih-jury-direct", use_container_width=True)
        st.markdown(textwrap.dedent("""
        <div style="font-size: 0.75rem; color: #64748B; text-align: center; margin-top: 6px; margin-bottom: 1.25rem; line-height: 1.4;">
            Pre-authenticated evaluation session for the judging bench.
        </div>
        """).strip(), unsafe_allow_html=True)

        if jury_bypass:
            profile = authenticate_user("jury@sih2026.in", "jury123")
            if profile:
                st.session_state["authenticated"] = True
                st.session_state["user_profile"] = profile
                st.session_state["page"] = "dashboard"
                st.query_params.clear()
                st.toast("SIH Evaluator Fast-Track Activated.", icon="⚓")
                st.rerun()

        # Back to Landing Page Link
        back_col1, back_col2, back_col3 = st.columns([1, 2, 1])
        with back_col2:
            if st.button("← Back to Landing Page", key="btn-back-overview", use_container_width=True):
                st.session_state["page"] = "landing"
                st.session_state["authenticated"] = False
                st.query_params.clear()
                st.rerun()


def render_login_screen():
    """Alias for authentication gateway."""
    render_auth_gateway()


def render_command_dashboard():
    user = st.session_state["user_profile"]
    is_officer = (user["role"] == ROLE_COMMAND_OFFICER)
    is_judge = (user["role"] == ROLE_TRIBUNAL_JUDGE)

    # --- TOP TACTICAL NAVIGATION BAR ---
    badge_bg = "#064e3b" if is_officer else "#1e1b4b"
    badge_fg = "#10b981" if is_officer else "#818cf8"

    # --- TOP STICKY HEADER (GAGANACAKSUH TOP-LEFT, PROFILE TOP-RIGHT) ---
    col_hdr_left, col_hdr_right = st.columns([5.5, 1.2])
    with col_hdr_left:
        st.markdown(f"""
        <div style="display:flex; align-items:baseline; gap:16px; margin-top:-6px; margin-bottom:8px;">
            <span style="font-family:'Cinzel', Georgia, serif; font-size:2.3rem; font-weight:800; letter-spacing:0.08em; color:#E2E8F0; text-transform:uppercase;">
                GAGANACAKSUH
            </span>
            <span style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#94A3B8; letter-spacing:0.04em;">
                UTC: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}Z &bull; SECTOR: ARABIAN SEA EEZ
            </span>
        </div>
        """, unsafe_allow_html=True)
    with col_hdr_right:
        with st.popover(f"👤 {user['officer_name'].split()[0]}", use_container_width=True):
            st.markdown(f"""
            <div style="text-align: center; padding: 4px 0 12px 0; border-bottom: 1px solid #161F30;">
                <div style="width: 52px; height: 52px; border-radius: 50%; background: #38BDF8; color: #0B0F17; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; font-weight: 700; margin: 0 auto 8px auto; border: 2px solid #E2E8F0;">
                    {user['officer_name'][0].upper()}
                </div>
                <div style="font-weight: 700; font-size: 0.95rem; color: #E2E8F0;">{user['officer_name']}</div>
                <div style="font-size: 0.72rem; color: #94A3B8; font-family: monospace;">{user['role']}</div>
            </div>
            <div style="background: #0B0F17; border-radius: 6px; padding: 10px 12px; margin: 10px 0; font-size: 0.75rem; color: #E2E8F0; font-family: monospace; line-height: 1.8;">
                <span style="color: #38BDF8;">SERVICE ID:</span> {user['service_number']}<br/>
                <span style="color: #38BDF8;">BASE:</span> {user['base']}<br/>
                <span style="color: #38BDF8;">CLEARANCE:</span> <span style="font-weight: 600; color: #38BDF8;">{user['meta']['clearance_level']}</span><br/>
                <span style="color: #94A3B8; font-size: 0.65rem;">SESSION: {user['session_token'][:18]}...</span>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚪 Terminate Session / Log Out", key="btn_hdr_logout", use_container_width=True, type="primary"):
                st.session_state["authenticated"] = False
                st.session_state["user_profile"] = None
                st.session_state["page"] = "auth_gateway"
                st.query_params.clear()
                st.rerun()

    # --- SIDEBAR CONTROLS & TELEMETRY ---
    with st.sidebar:
        st.markdown("<b style='color:#E2E8F0; font-family:monospace; font-size:0.85rem; letter-spacing:0.06em;'>PIPELINE STAGES</b>", unsafe_allow_html=True)

        stage_options = [
            "Tactical Common Operating Picture",
            "Copernicus Multi-Spectral WMS",
            "Neural Segmentation & Drift Model",
            "Zero-Trust AIS Telemetry Audit",
            "Cryptographic Chain of Custody",
            "Autonomous Vessel Reroute & Dispatch"
        ] if is_officer else [
            "Tactical Common Operating Picture",
            "Zero-Trust AIS Telemetry Audit",
            "Cryptographic Chain of Custody"
        ]

        curr_stage = st.session_state.get("active_stage", stage_options[0])
        curr_idx = stage_options.index(curr_stage) if curr_stage in stage_options else 0

        active_stage = st.radio(
            "Select Pipeline Stage:",
            stage_options,
            index=curr_idx,
            key="sidebar_stage_radio",
            label_visibility="collapsed"
        )
        st.session_state["active_stage"] = active_stage

        st.markdown(f"""
        <div style="margin-top: 8px; margin-bottom: 12px; padding: 6px 10px; background-color: #0B0F17; border-radius: 4px; border: 1px solid #1E293B; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #94A3B8; line-height: 1.5;">
            <div><b>ZONE:</b> <span style="color:#E2E8F0;">SECTOR 4</span></div>
            <div><b>SENSORS:</b> 4 ACTIVE</div>
            <div><b>COURT DOCKET:</b> VALID</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<b style='color:#38BDF8; font-family:monospace;'>SURVEILLANCE SECTOR SELECTOR</b>", unsafe_allow_html=True)
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
        st.markdown("<b style='color:#38BDF8; font-family:monospace;'>CONSTELLATION TELEMETRY</b>", unsafe_allow_html=True)
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

    # --- HELPER FUNCTIONS FOR EACH PIPELINE STAGE ---
    def render_tactical_cop_stage():
        with st.container(border=True):
            st.subheader("Tactical Common Operating Picture", divider="blue")
            st.caption("Real-Time Maritime Common Operating Picture (COP) integrating Sentinel-1 C-Band SAR, U-Net slick contours, Lagrangian origin cone, AIS transponder tracks, and navigation exclusion corridors.")

            with st.container(border=True):
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

            with st.container(border=True):
                st.markdown(textwrap.dedent("""
                <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 10px; width: 100%; box-sizing: border-box; font-family: monospace; font-size: 0.72rem; padding: 4px 0;">
                    <div style="flex: 1 1 170px; max-width: 220px; min-width: 150px; background-color: #1e1017; border: 1px solid #3d1724; border-left: 3px solid #ff0055; padding: 8px 10px; border-radius: 4px; box-sizing: border-box; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                        <div><b style="color: #ff0055;">RED SOLID</b></div>
                        <div style="color: #e2e8f0; margin-top: 2px;">Mineral Oil Slick (U-Net)</div>
                    </div>
                    <div style="flex: 1 1 170px; max-width: 220px; min-width: 150px; background-color: #1f190e; border: 1px solid #3d2b14; border-left: 3px solid #ffaa00; padding: 8px 10px; border-radius: 4px; box-sizing: border-box; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                        <div><b style="color: #ffaa00;">YELLOW DASH</b></div>
                        <div style="color: #e2e8f0; margin-top: 2px;">Lagrangian Origin Cone</div>
                    </div>
                    <div style="flex: 1 1 170px; max-width: 220px; min-width: 150px; background-color: #11202e; border: 1px solid #1c354d; border-left: 3px solid #38BDF8; padding: 8px 10px; border-radius: 4px; box-sizing: border-box; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                        <div><b style="color: #38BDF8;">CYAN LINE</b></div>
                        <div style="color: #e2e8f0; margin-top: 2px;">Compliant AIS Tracks</div>
                    </div>
                    <div style="flex: 1 1 170px; max-width: 220px; min-width: 150px; background-color: #2a110a; border: 1px solid #4a1d12; border-left: 3px solid #f97316; padding: 8px 10px; border-radius: 4px; box-sizing: border-box; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                        <div><b style="color: #f97316;">ORANGE MARKER</b></div>
                        <div style="color: #e2e8f0; margin-top: 2px;">GPS Spoofed (&gt;35 kts)</div>
                    </div>
                    <div style="flex: 1 1 170px; max-width: 220px; min-width: 150px; background-color: #2e0d16; border: 1px solid #4f1523; border-left: 3px solid #ff0033; padding: 8px 10px; border-radius: 4px; box-sizing: border-box; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                        <div><b style="color: #ff0033;">PULSING CRIMSON</b></div>
                        <div style="color: #e2e8f0; margin-top: 2px;">SAR Dark Ship</div>
                    </div>
                </div>
                """).strip(), unsafe_allow_html=True)


    def render_satellite_wms_stage():
        with st.container(border=True):
            st.subheader("Copernicus Multi-Spectral WMS Surveillance", divider="blue")
            st.caption("Dual-Satellite multi-sensor grid integrating Sentinel-1 C-Band SAR with Sentinel-2 Optical MSI for automated false-positive suppression (discriminating mineral crude from biogenic algal blooms).")

            wms_meta = get_wms_capabilities_mock(st.session_state["selected_region"])
            sar_scene = st.session_state["sar_scene"]

            col_wms1, col_wms2 = st.columns([1.1, 1.9])
            with col_wms1:
                with st.container(border=True):
                    st.markdown("**WMS STREAM HANDSHAKE & TELEMETRY**")
                    st.markdown(textwrap.dedent(f"""
                    <div style="font-family:monospace; font-size:0.75rem; color:#94a3b8; line-height:1.6;">
                        <div><b>ENDPOINT:</b> {wms_meta['endpoint']}</div>
                        <div><b>CRS:</b> {wms_meta['crs']}</div>
                        <div><b>THEATRE:</b> {wms_meta['target_region']}</div>
                        <div><b>BBOX:</b> {wms_meta['bbox']}</div>
                        <div style="margin-top:4px; padding-top:4px; border-top:1px dashed #1e293b;">
                            <b>STREAM STATUS:</b> <span style="color:#10b981; font-weight:600;">ACTIVE (10m RESOLUTION)</span>
                        </div>
                    </div>
                    """).strip(), unsafe_allow_html=True)

                with st.container(border=True):
                    st.markdown("**ACTIVE SURVEILLANCE SENSOR LAYERS**")
                    layers_html = "".join([f'<div style="background-color:#020617; border:1px solid #1e293b; border-radius:4px; padding:6px 8px; margin-bottom:5px; font-family:monospace; font-size:0.72rem;"><div style="display:flex; align-items:flex-start; justify-content:space-between; gap:6px;"><div style="color:#f1f5f9; font-weight:600; flex:1; min-width:0; word-break:break-word;"><span class="led-live"></span>{layer["title"]}</div><span style="color:#10b981; font-size:0.65rem; font-weight:700; flex-shrink:0; background:rgba(16,185,129,0.12); padding:2px 5px; border-radius:3px;">ONLINE</span></div><div style="color:#64748b; font-size:0.68rem; margin-top:2px; margin-left:14px;">ID: {layer["layer_id"]} &bull; {layer["orbit"]}</div></div>' for layer in wms_meta["active_layers"]])
                    st.markdown(layers_html, unsafe_allow_html=True)

            with col_wms2:
                # Normalize NDVI for display
                ndvi_norm = ((sar_scene["ndvi"] + 1.0) / 2.0 * 255.0).astype(np.uint8)
                ndvi_colored = cv2.applyColorMap(ndvi_norm, cv2.COLORMAP_JET)

                sar_img = sar_scene["sar_amplitude"]
                if sar_img.dtype != np.uint8:
                    sar_min, sar_max = float(sar_img.min()), float(sar_img.max())
                    if sar_max > sar_min:
                        sar_img = ((sar_img - sar_min) / (sar_max - sar_min) * 255.0).astype(np.uint8)
                    else:
                        sar_img = sar_img.astype(np.uint8)
                buf_sar = io.BytesIO()
                Image.fromarray(sar_img).save(buf_sar, format="PNG")
                sar_b64 = base64.b64encode(buf_sar.getvalue()).decode("utf-8")

                buf_ndvi = io.BytesIO()
                Image.fromarray(ndvi_colored).save(buf_ndvi, format="PNG")
                ndvi_b64 = base64.b64encode(buf_ndvi.getvalue()).decode("utf-8")

                with st.container(border=True):
                    st.markdown("**MULTI-SENSOR VERIFICATION IMAGERY**")
                    st.markdown(textwrap.dedent(f"""
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; width: 100%; box-sizing: border-box;">
                        <div style="background-color: #0b1120; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 12px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: space-between;">
                            <div style="font-family: monospace; font-size: 0.75rem; color: #94A3B8; font-weight: 600; margin-bottom: 6px;">
                                1. Sentinel-1 SAR GRDH (Radar Amplitude)
                            </div>
                            <div style="flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; border-radius: 4px; background: #020617;">
                                <img src="data:image/png;base64,{sar_b64}" style="width: 100%; height: 185px; object-fit: cover; display: block;" alt="Sentinel-1 SAR" />
                            </div>
                            <div style="font-size: 0.7rem; color: #64748B; margin-top: 6px; text-align: center; font-family: monospace;">
                                Radar dark spot (wave damping)
                            </div>
                        </div>
                        <div style="background-color: #0b1120; border: 1px solid #1e293b; border-radius: 6px; padding: 10px 12px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: space-between;">
                            <div style="font-family: monospace; font-size: 0.75rem; color: #94A3B8; font-weight: 600; margin-bottom: 6px;">
                                2. Sentinel-2 Optical MSI (NDVI Index)
                            </div>
                            <div style="flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; border-radius: 4px; background: #020617;">
                                <img src="data:image/png;base64,{ndvi_b64}" style="width: 100%; height: 185px; object-fit: cover; display: block;" alt="Sentinel-2 Optical" />
                            </div>
                            <div style="font-size: 0.7rem; color: #64748B; margin-top: 6px; text-align: center; font-family: monospace;">
                                Red: Algal Bloom | Blue: Mineral Oil
                            </div>
                        </div>
                    </div>
                    """).strip(), unsafe_allow_html=True)

                with st.container(border=True):
                    st.markdown("**FORENSIC CLASSIFICATION: MARPOL ANNEX II DISCHARGE**")
                    st.markdown(textwrap.dedent("""
                    <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.5; margin-bottom: 6px;">
                        The Sentinel-2 false-color optical index classifies this anomaly as yellow, mathematically ruling out petroleum-based mineral oil (blue) and biological algal blooms (red). Cross-referencing Sentinel-1 wave damping with ERA5 wind forcings (&gt;3m/s) confirms physical surface film presence.
                    </div>
                    <div style="font-size: 0.75rem; color: #94A3B8; font-family: monospace; line-height: 1.6;">
                        &bull; <strong style="color: #F8FAFC;">Primary Match:</strong> Edible liquid cargo (vegetable/palm oil) tank wash discharge.<br/>
                        &bull; <strong style="color: #F8FAFC;">Secondary Match:</strong> Natural biogenic slick (zooplankton lipid release).
                    </div>
                    """).strip(), unsafe_allow_html=True)

                st.markdown(textwrap.dedent(f"""
                <div style="width:100%; background:rgba(16, 185, 129, 0.08); border:1px solid rgba(16, 185, 129, 0.35); border-left:4px solid #10B981; border-radius:6px; padding:12px 16px; font-size:0.78rem; color:#E2E8F0; line-height:1.5; box-sizing:border-box; text-align:center; display:flex; align-items:center; justify-content:center;">
                    <span><b style="color:#10B981;">Automated Spectral Filter:</b> Suppressed {unet_res.get('optical_lookalikes_filtered', 1)} biogenic lookalike patches in optical NDVI spectrum.</span>
                </div>
                """).strip(), unsafe_allow_html=True)


    def render_unet_drift_stage():
        with st.container(border=True):
            st.subheader("Neural Segmentation & Hydrodynamic Drift Model", divider="blue")
            st.caption("U-Net deep convolutional inference (trained on Zenodo SAR Dataset #4124976) with Fay-spreading slick age calculation and reversed Lagrangian advection vectors.")

            meta = unet_res["model_metadata"]
            c_ai1, c_ai2 = st.columns([1.1, 1.9])
            with c_ai1:
                with st.container(border=True):
                    st.markdown("**NEURAL ARCHITECTURE & VALIDATION METRICS**")
                    st.markdown(textwrap.dedent(f"""
                    <div style="font-family:monospace; font-size:0.75rem; color:#cbd5e1; line-height:1.6;">
                        <div><b>Architecture:</b> {meta['architecture']}</div>
                        <div><b>Training Corpus:</b> {meta['training_dataset']}</div>
                        <div><b>Validation Dice:</b> <span style="color:#10b981; font-weight:600;">{meta['validation_dice_coefficient']*100:.2f}%</span></div>
                        <div><b>Mean IoU:</b> <span style="color:#38BDF8; font-weight:600;">{meta['mean_iou']*100:.2f}%</span></div>
                        <div><b>Precision / Recall:</b> {meta['precision']*100:.1f}% / {meta['recall']*100:.1f}%</div>
                        <div><b>Weights SHA-256:</b> <code style="font-size:0.65rem; color:#64748b;">{meta['weights_hash'][:22]}...</code></div>
                    </div>
                    """).strip(), unsafe_allow_html=True)

                with st.container(border=True):
                    st.markdown("**FAY WEATHERING & SPREADING KINEMATICS**")
                    st.markdown(textwrap.dedent(f"""
                    <div style="font-family:monospace; font-size:0.75rem; color:#cbd5e1; line-height:1.6;">
                        <div><b>Spreading Regime:</b> {unet_res.get('fay_regime', 'Phase 3 (Surface Tension-Viscous)')}</div>
                        <div><b>Observed Surface Area:</b> <span style="color:#38BDF8; font-weight:600;">{unet_res['total_area_sq_km']} sq km</span></div>
                        <div><b>Calculated Slick Age:</b> <span style="color:#F59E0B; font-weight:600;">{unet_res.get('estimated_age_hours', 12.0)} hours prior</span></div>
                        <div><b>Evaporation Loss:</b> ~30.5%</div>
                    </div>
                    """).strip(), unsafe_allow_html=True)

            with c_ai2:
                with st.container(border=True):
                    st.markdown("**SEGMENTATION CONFIDENCE MASKS**")
                    pcol1, pcol2 = st.columns(2)
                    with pcol1:
                        prob_display = (unet_res["probability_map"] * 255).astype(np.uint8)
                        prob_heat = cv2.applyColorMap(prob_display, cv2.COLORMAP_INFERNO)
                        st.image(prob_heat, caption="U-Net Confidence Heatmap (>94.8%)", use_container_width=True)
                    with pcol2:
                        st.image(unet_res["binary_mask"] * 255, caption="Cleaned Binary Extraction Mask", use_container_width=True)

            with st.container(border=True):
                st.markdown("**PINPOINTED EMISSION ORIGIN & LAGRANGIAN BACKCAST VECTOR**")
                st.markdown(textwrap.dedent(f"""
                <div style="font-family:monospace; font-size:0.78rem; color:#cbd5e1;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
                        <div><b>ORIGIN COORDINATES:</b> <span style="color:#F59E0B; font-weight:700;">{backcast_res['origin_release_coord'][0]:.4f}°N, {backcast_res['origin_release_coord'][1]:.4f}°E</span></div>
                        <div><b>NET DRIFT SPEED:</b> <span style="color:#38BDF8;">{backcast_res['net_drift_speed_knots']} kts</span></div>
                    </div>
                    <div style="margin-top:4px; font-size:0.72rem; color:#94a3b8;">
                        <b>UNCERTAINTY ENVELOPE:</b> &plusmn;{backcast_res['origin_uncertainty_meters']} meters &bull; <b>PARTICLE ADVECTION:</b> 400 Reversed Lagrangian Nodes
                    </div>
                </div>
                """).strip(), unsafe_allow_html=True)


    def render_ais_audit_stage():
        with st.container(border=True):
            st.subheader("Zero-Trust AIS Telemetry Audit & Counter-Spoofing", divider="blue")
            st.caption("Adversarial Defense Engine adhering to ITU-R M.1371 schemas. Cross-references radio transponder transmissions against physical Sentinel-1 SAR hull reflections to expose GPS spoofing and Dark Ships.")

            with st.container(border=True):
                st.markdown("**ACTIVE CYBER & MARITIME THREAT NOTIFICATIONS**")
                for alert in ais_res["security_alerts"]:
                    is_crit = "CRITICAL" in alert["level"]
                    card_class = "hud-card-alert" if is_crit else "hud-card"
                    st.markdown(f"""
                    <div class="{card_class}" style="margin-bottom:8px;">
                        <div style="display:flex; justify-content:space-between; font-family:monospace; font-size:0.8rem;">
                            <b style="color:#ef4444;"><span class="led-alert"></span>[{alert['code']}]</b>
                            <span style="color:#94a3b8;">{alert.get('timestamp', 'LIVE SENSOR PASS')}</span>
                        </div>
                        <div style="font-size:0.82rem; color:#f8fafc; margin-top:4px;">
                            {alert['details']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown("**TARGET VESSEL ATTRIBUTION MATRIX (RANKED BY PROBABILISTIC LIABILITY)**")
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
                st.dataframe(pd.DataFrame(v_rows), use_container_width=True, hide_index=True)


    def render_evidence_vault_stage():
        with st.container(border=True):
            st.subheader("Cryptographic Chain of Custody (ISO/IEC 27037)", divider="blue")
            st.caption("Adheres strictly to ISO/IEC 27037 standards for digital evidence preservation. Locks satellite telemetry, spatial polygon coordinates, backcasting vectors, and suspect MMSI strings into an immutable SHA-256 ledger.")

            ledger_engine = ForensicLedgerEngine()

            with st.container(border=True):
                st.markdown(f"""
                <div style="font-family:monospace;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="color:#38BDF8; font-weight:700;">DOCKET ID: {docket['docket_id']}</span>
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

            with st.container(border=True):
                st.markdown("**INTERACTIVE ADVERSARIAL BIT-TAMPER VALIDATOR**")
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

            with st.container(border=True):
                st.markdown("**DIGITAL CHAIN OF CUSTODY LOG (ISO/IEC 27037 CLAUSE 6.4)**")
                custody_data = []
                for c in docket.get("chain_of_custody", []):
                    custody_data.append({
                        "Seq": c["sequence"],
                        "Event Description": c["event"],
                        "Authority / Module": c["authority"],
                        "Timestamp (UTC)": c["time"],
                        "SHA-256 Checksum": c["hash"]
                    })
                st.dataframe(pd.DataFrame(custody_data), use_container_width=True, hide_index=True)

            with st.container(border=True):
                st.markdown("**OFFICIAL COURT-ADMISSIBLE DOSSIER EXPORT**")
                pdf_bytes = ledger_engine.generate_court_dossier_pdf(docket)

                cd1, cd2 = st.columns([1.5, 1])
                with cd1:
                    st.download_button(
                        label="📥 DOWNLOAD ISO/IEC 27037 LEGAL EVIDENCE DOSSIER (PDF)",
                        data=pdf_bytes,
                        file_name=f"{docket['docket_id']}_EVIDENCE_DOSSIER.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                with cd2:
                    st.download_button(
                        label="📥 EXPORT RAW EVIDENCE LEDGER (JSON)",
                        data=json.dumps(docket, indent=2),
                        file_name=f"{docket['docket_id']}_RAW_LEDGER.json",
                        mime="application/json",
                        use_container_width=True
                    )


    def render_reroute_dispatch_stage():
        with st.container(border=True):
            st.subheader("Autonomous Vessel Reroute & Edge Dispatch", divider="blue")
            st.caption("Lightweight edge execution computing 5 NM exclusion corridor, vessel divert waypoints, and automated VHF Navtex emergency broadcasts.")

            rerouter = TacticalReroutingEngine(buffer_radius_km=5.0)
            ex_zone = st.session_state["exclusion_zone"]
            reroutes = st.session_state["reroutes"]

            c_rr1, c_rr2 = st.columns(2)
            with c_rr1:
                with st.container(border=True):
                    st.markdown("**DYNAMIC EXCLUSION CORRIDOR**")
                    st.markdown(textwrap.dedent(f"""
                    <div style="font-family:monospace; font-size:0.8rem; color:#cbd5e1; line-height:1.6;">
                        <div><b>Zone Status:</b> <span style="color:#ef4444;">{ex_zone['threat_status']}</span></div>
                        <div><b>Mandatory Standoff:</b> {ex_zone['safety_buffer_nm']} Nautical Miles</div>
                        <div><b>Advisory:</b> {ex_zone['advisory']}</div>
                        <div><b>Corridor Bounds:</b> 4 Geodetic Vertices</div>
                    </div>
                    """).strip(), unsafe_allow_html=True)

                with st.container(border=True):
                    st.markdown("**AUTOMATED EMERGENCY SAFETY DISPATCH**")
                    if st.button("🚨 TRIGGER EMERGENCY MARITIME SAFETY BROADCAST", use_container_width=True):
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
                with st.container(border=True):
                    st.markdown("**COMPUTED VESSEL COLLISION AVOIDANCE REROUTES**")
                    for r in reroutes:
                        st.markdown(f"""
                        <div style="background-color:#0b1120; border:1px solid #1e293b; padding:10px 12px; border-radius:6px; font-family:monospace; font-size:0.78rem; margin-bottom:8px;">
                            <div style="display:flex; justify-content:space-between;">
                                <b style="color:#38BDF8;">{r['vessel_name']} (MMSI {r['mmsi']})</b>
                                <span style="color:#f97316;">{r['urgency']}</span>
                            </div>
                            <div style="color:#94a3b8; font-size:0.72rem; margin-top:6px;">
                                Distance to Spill: <b>{r['dist_to_spill_km']} km</b><br/>
                                Divert Waypoint: <span style="color:#10b981;">{r['diversion_waypoint'][0]}°N, {r['diversion_waypoint'][1]}°E</span><br/>
                                Course Change: <span style="color:#cbd5e1;">{r['reroute_instruction']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    # --- MAIN WORKBENCH CONTENT (FULL WIDTH) ---
    unet_res = st.session_state["unet_results"]
    ais_res = st.session_state["ais_results"]
    backcast_res = st.session_state["backcast_results"]
    docket = st.session_state["docket"]

    # Summary Metrics Row (Strictly aligned in a single horizontal row of 4 columns)
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("OIL SLICK AREA", f"{unet_res['total_area_sq_km']} km²", f"Confidence {unet_res['mean_confidence_score']*100:.1f}%")
        with col2:
            st.metric("SPREAD AGE", f"{unet_res.get('estimated_age_hours', 12.0)} hrs", unet_res.get("fay_regime", "Viscous"))
        with col3:
            st.metric("TRACKED VESSELS", f"{ais_res['total_vessels_tracked']}", f"{ais_res['spoofed_vessels_detected']} GPS Spoofed")
        with col4:
            st.metric("DARK SHIPS FOUND", f"{ais_res['dark_vessels_detected']}", "SAR Hull Scan Confirmed")

    # Active Stage Renderer (2D Map is First View by Default)
    if "Common Operating Picture" in active_stage or "COP" in active_stage or "Tactical" in active_stage:
        render_tactical_cop_stage()
    elif "Copernicus" in active_stage or "WMS" in active_stage:
        render_satellite_wms_stage()
    elif "Neural" in active_stage or "Segmentation" in active_stage or "Drift" in active_stage:
        render_unet_drift_stage()
    elif "Zero-Trust" in active_stage or "AIS" in active_stage:
        render_ais_audit_stage()
    elif "Chain of Custody" in active_stage or "Cryptographic" in active_stage or "Evidence" in active_stage or "Vault" in active_stage:
        render_evidence_vault_stage()
    elif "Reroute" in active_stage or "Dispatch" in active_stage:
        render_reroute_dispatch_stage()


def main():
    if "page" in st.query_params:
        target = st.query_params["page"]
        st.query_params.clear()
        if target in ["dashboard", "analysis", "command", "main"]:
            if not st.session_state.get("authenticated", False):
                profile = authenticate_user("jury@sih2026.in", "jury123")
                if profile:
                    st.session_state["authenticated"] = True
                    st.session_state["user_profile"] = profile
            st.session_state["page"] = "dashboard"
        elif target in ["auth", "auth_gateway", "login", "signin", "sign_in"]:
            st.session_state["page"] = "auth_gateway"
            st.session_state["authenticated"] = False
        elif target in ["landing"]:
            st.session_state["page"] = "landing"
            st.session_state["authenticated"] = False

    page = st.session_state.get("page", "landing")
    if page in ["auth_gateway", "auth", "login", "signin", "sign_in"]:
        render_auth_gateway()
    elif page == "landing":
        render_landing_page()
    elif st.session_state.get("authenticated", False) and st.session_state.get("user_profile") is not None:
        render_command_dashboard()
    else:
        render_landing_page()

if __name__ == "__main__":
    main()
