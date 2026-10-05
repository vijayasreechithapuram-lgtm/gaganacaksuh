import numpy as np
import cv2
import os

def create_benchmark_slick_images():
    os.makedirs('static/img', exist_ok=True)
    width, height = 640, 640
    np.random.seed(42)

    # Common slick geometry centered in the image
    y, x = np.ogrid[:height, :width]
    cx, cy = 340, 310
    rad = np.radians(32.0)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    dx = x - cx
    dy = y - cy
    rx = dx * cos_a + dy * sin_a
    ry = -dx * sin_a + dy * cos_a

    # Complex slick shape: main body + tail streak + filaments
    theta = np.arctan2(ry, rx)
    dist_r = np.sqrt((rx / 140.0)**2 + (ry / 48.0)**2)
    edge_noise = 0.22 * np.sin(4 * theta) + 0.15 * np.cos(7 * theta)
    main_slick = (dist_r + edge_noise) <= 1.0

    tail_mask = ((rx > 0) & (rx < 230) & (np.abs(ry) < (36.0 * (1 - rx / 250.0))))
    slick_mask = np.logical_or(main_slick, tail_mask)

    # Dilate/smooth slightly for natural fluid dispersion
    slick_uint = (slick_mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    slick_soft = cv2.GaussianBlur(slick_uint, (15, 15), 0) / 255.0

    # Vessel coordinates (at origin of spill)
    vx, vy = int(cx + 175 * cos_a), int(cy + 175 * sin_a)

    # -------------------------------------------------------------
    # 1. Sentinel-1 SAR (Wave Damping - Radar Amplitude in Grayscale)
    # -------------------------------------------------------------
    # Sea surface Rayleigh / Gamma speckle noise
    sea_clutter = np.random.gamma(9.0, 13.5, (height, width))
    sea_clutter = np.clip(sea_clutter, 25, 230).astype(np.float32)

    # Capillary wave damping: severely dampens backscatter inside the slick
    sar_img = sea_clutter * (1.0 - 0.78 * slick_soft)
    sar_img = np.clip(sar_img, 0, 255).astype(np.uint8)

    # Add metallic vessel radar return (bright metallic reflection)
    cv2.circle(sar_img, (vx, vy), 4, 255, -1)
    cv2.line(sar_img, (vx - 6, vy - 3), (vx + 6, vy + 3), 250, 2)

    # Add forensic telemetry banner & annotations
    sar_bgr = cv2.cvtColor(sar_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(sar_bgr, "SENTINEL-1 C-SAR GRDH [VV]", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230, 230, 230), 2)
    cv2.putText(sar_bgr, "PASS: DESCENDING #117 | RES: 10m", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.putText(sar_bgr, "WAVE DAMPING DARK SPOT [CONFIRMED]", (20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    cv2.putText(sar_bgr, "METALLIC HULL RETURN", (vx - 140, vy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.arrowedLine(sar_bgr, (vx - 20, vy - 15), (vx - 6, vy - 6), (0, 255, 0), 1, tipLength=0.3)

    # -------------------------------------------------------------
    # 2. ISRO EOS-04 Polarimetric (Pauli Decomposition RGB)
    # -------------------------------------------------------------
    # Pauli decomposition: Red = HH-VV (double bounce hull), Green = HV (volume), Blue = HH+VV (surface)
    eos_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    # Ocean background: strong surface scattering (Cyan/Blue)
    eos_blue = np.clip(sea_clutter * 0.95 * (1.0 - 0.72 * slick_soft), 10, 240).astype(np.uint8)
    eos_green = np.clip(sea_clutter * 0.35 * (1.0 - 0.65 * slick_soft), 5, 120).astype(np.uint8)
    eos_red = np.clip(sea_clutter * 0.15 * (1.0 - 0.50 * slick_soft), 0, 80).astype(np.uint8)
    eos_bgr[:, :, 0] = eos_blue
    eos_bgr[:, :, 1] = eos_green
    eos_bgr[:, :, 2] = eos_red

    # Vessel target double bounce (intense red/yellow specular reflection)
    cv2.circle(eos_bgr, (vx, vy), 5, (0, 60, 255), -1)
    cv2.circle(eos_bgr, (vx, vy), 3, (50, 220, 255), -1)

    cv2.putText(eos_bgr, "ISRO EOS-04 POLARIMETRIC [PAULI RGB]", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (230, 230, 230), 2)
    cv2.putText(eos_bgr, "HYBRID QUAD-POL C-BAND | RES: 3m", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.putText(eos_bgr, "STRUCTURAL CONFIRMATION [DOUBLE-BOUNCE]", (20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    cv2.putText(eos_bgr, "STEEL HULL TARGET", (vx - 130, vy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
    cv2.arrowedLine(eos_bgr, (vx - 20, vy - 15), (vx - 6, vy - 6), (0, 220, 255), 1, tipLength=0.3)

    # -------------------------------------------------------------
    # 3. Sentinel-2 Optical MSI (False Color / Hydrocarbon Index)
    # -------------------------------------------------------------
    # Optical false-color: deep sea water is dark navy, hydrocarbon film reflects distinctly in sunglint/SWIR
    s2_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    # Ocean base
    s2_bgr[:, :, 0] = np.clip(70 + sea_clutter * 0.15, 0, 180).astype(np.uint8)  # Blue
    s2_bgr[:, :, 1] = np.clip(45 + sea_clutter * 0.10, 0, 140).astype(np.uint8)  # Green
    s2_bgr[:, :, 2] = np.clip(25 + sea_clutter * 0.05, 0, 90).astype(np.uint8)   # Red

    # Mineral oil sheen (distinct hydrocarbon refraction & false-color spectral index)
    s2_bgr[:, :, 0] = np.clip(s2_bgr[:, :, 0] * (1.0 - slick_soft * 0.4) + slick_soft * 30, 0, 255).astype(np.uint8)
    s2_bgr[:, :, 1] = np.clip(s2_bgr[:, :, 1] * (1.0 - slick_soft * 0.2) + slick_soft * 170, 0, 255).astype(np.uint8)
    s2_bgr[:, :, 2] = np.clip(s2_bgr[:, :, 2] * (1.0 - slick_soft * 0.1) + slick_soft * 220, 0, 255).astype(np.uint8)

    # Biogenic lookalike patch nearby for comparison (high chlorophyll / NDVI > 0.45)
    lx, ly = 160, 160
    lr = np.sqrt(((x - lx) / 50.0)**2 + ((y - ly) / 36.0)**2)
    algae_soft = np.clip(1.0 - lr, 0.0, 1.0)
    s2_bgr[:, :, 1] = np.clip(s2_bgr[:, :, 1] + algae_soft * 180, 0, 255).astype(np.uint8)
    s2_bgr[:, :, 0] = np.clip(s2_bgr[:, :, 0] * (1.0 - algae_soft * 0.5), 0, 255).astype(np.uint8)

    cv2.putText(s2_bgr, "SENTINEL-2 OPTICAL MSI [HYDROCARBON INDEX]", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (230, 230, 230), 2)
    cv2.putText(s2_bgr, "B02/B04/B08 COMPOSITE | RES: 10m", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.putText(s2_bgr, "ALGAL BLOOM (RULED OUT)", (lx - 80, ly - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
    cv2.putText(s2_bgr, "CRUDE OIL SHEEN (NDVI < 0)", (cx - 100, cy + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
    cv2.putText(s2_bgr, "MINERAL OIL VERIFIED [NO CHLOROPHYLL]", (20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    # -------------------------------------------------------------
    # 4. Suomi NPP / VIIRS Thermal (SST & Thermal Radiance Anomaly)
    # -------------------------------------------------------------
    # Thermal infrared gradient (Inferno/Magma colormap for sea surface temperature)
    temp_grid = 28.5 + (sea_clutter - 120.0) * 0.008  # ~28.5°C ocean
    # Oil film thermal contrast anomaly: cooler surface skin by -1.2°C due to reduced evaporation/albedo
    temp_grid -= slick_soft * 1.35
    # Ship engine exhaust hot-spot: +8.5°C
    v_dist = np.sqrt(((x - vx)/10.0)**2 + ((y - vy)/10.0)**2)
    v_heat = np.clip(1.0 - v_dist, 0.0, 1.0)
    temp_grid += v_heat * 8.5

    # Normalize temperature to 0-255 for Inferno colormap
    temp_norm = np.clip((temp_grid - 26.5) / 4.0 * 255.0, 0, 255).astype(np.uint8)
    viirs_bgr = cv2.applyColorMap(temp_norm, cv2.COLORMAP_INFERNO)

    cv2.putText(viirs_bgr, "SUOMI NPP / VIIRS THERMAL IR [DNB]", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2)
    cv2.putText(viirs_bgr, "CH M15/M16 11um SST CONTRAST | RES: 375m", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(viirs_bgr, "ENGINE THERMAL PLUME", (vx - 145, vy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.arrowedLine(viirs_bgr, (vx - 20, vy - 15), (vx - 6, vy - 6), (255, 255, 255), 1, tipLength=0.3)
    cv2.putText(viirs_bgr, "THERMAL RADIANCE ANOMALY [CONFIRMED]", (20, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    # Save to both static/img/ and root
    cv2.imwrite("static/img/sar_slick.png", sar_bgr)
    cv2.imwrite("static/img/eos_slick.png", eos_bgr)
    cv2.imwrite("static/img/s2_slick.png", s2_bgr)
    cv2.imwrite("static/img/viirs_slick.png", viirs_bgr)

    cv2.imwrite("sar_slick.png", sar_bgr)
    cv2.imwrite("eos_slick.png", eos_bgr)
    cv2.imwrite("s2_slick.png", s2_bgr)
    cv2.imwrite("viirs_slick.png", viirs_bgr)

    print("Successfully generated all 4 benchmark oil slick satellite images!")

if __name__ == "__main__":
    create_benchmark_slick_images()
