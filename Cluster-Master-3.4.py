
import pandas as pd
import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from sklearn.cluster import DBSCAN

st.set_page_config(page_title="AT2 Spatial Analysis", page_icon="🔬", layout="wide")
st.title("🔬 AT2 Spatial Analysis")
st.markdown("""
**Surfactant-C+ AT2-Zellen – räumliche Analyse**

Jede Kombination aus **Image + ROI_ID** wird zunächst als eigene Analyseeinheit behandelt.
""")

st.sidebar.header("⚙️ Analyseparameter")
st.sidebar.subheader("📏 Kalibrierung")
calibration_mode = st.sidebar.radio("Koordinaten/Kalibrierung", ["Automatisch aus QuPath", "Manuell"], index=0)
manual_pixel_um = st.sidebar.number_input("Pixelgröße (µm / Pixel)", min_value=0.0001, max_value=10.0, value=0.2128, step=0.001, format="%.4f", help="Nur relevant, wenn keine QuPath-Kalibrierung verwendet wird.")

st.sidebar.markdown("---")
st.sidebar.subheader("🔵 Clusterdefinition")
eps_um = st.sidebar.number_input("Clusterabstand eps (µm)", min_value=1.0, max_value=1000.0, value=10.0, step=1.0, help="Maximaler Abstand zwischen AT2-Zellen innerhalb eines DBSCAN-Clusters.")
min_samples = st.sidebar.number_input("Minimale AT2-Zellen pro Cluster", min_value=2, max_value=50, value=3, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🔷 Voronoi")
voronoi_mode = st.sidebar.radio(
    "Maximale Voronoi-Fläche",
    ["Automatisch: Mittelwert + 2 SD", "Benutzerdefiniert"],
    index=0,
    help="Die Grenze betrifft ausschließlich die Voronoi-Auswertung. Keine AT2-Zelle wird aus Zählung oder DBSCAN entfernt."
)
manual_voronoi_area_um2 = st.sidebar.number_input(
    "Eigenes Voronoi-Maximum (µm²)",
    min_value=1.0,
    max_value=1e12,
    value=2000.0,
    step=100.0,
    format="%.0f",
    disabled=(voronoi_mode != "Benutzerdefiniert"),
    help="Nur für die Voronoi-Auswertung. Voronoi-Flächen oberhalb dieses Wertes werden nicht verwendet. Die AT2-Zellen bleiben vollständig erhalten."
)

st.markdown("---")
uploaded_file = st.file_uploader("📂 QuPath MASTER-CSV laden", type=["csv"])
if uploaded_file is None:
    st.info("Bitte deine **Positive_Centroids_MASTER.csv** aus QuPath laden.\n\nDie Datei kann beliebig viele Bilder enthalten.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
except Exception as e:
    st.error(f"CSV konnte nicht gelesen werden:\n{e}")
    st.stop()

df.columns = df.columns.astype(str).str.strip()
required_basic = ["Image", "ROI_ID", "ROI_Area_mm2"]
missing_basic = [col for col in required_basic if col not in df.columns]
if missing_basic:
    st.error("Diese QuPath-Spalten fehlen:\n\n" + "\n".join(missing_basic))
    st.write("Gefundene Spalten:")
    st.write(list(df.columns))
    st.stop()

numeric_columns = ["ROI_Area_pixel2", "ROI_Area_um2", "ROI_Area_mm2", "PixelWidth_um", "PixelHeight_um", "Positive_Count", "X_pixel", "Y_pixel", "X_um", "Y_um"]
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if calibration_mode == "Automatisch aus QuPath":
    if "PixelWidth_um" in df.columns and "PixelHeight_um" in df.columns:
        valid_calibration = (df["PixelWidth_um"].notna() & df["PixelHeight_um"].notna() & (df["PixelWidth_um"] > 0) & (df["PixelHeight_um"] > 0))
        if valid_calibration.any():
            median_px_width = float(df.loc[valid_calibration, "PixelWidth_um"].median())
            median_px_height = float(df.loc[valid_calibration, "PixelHeight_um"].median())
            st.success(f"✓ QuPath-Kalibrierung erkannt: {median_px_width:.4f} × {median_px_height:.4f} µm/Pixel")
        else:
            median_px_width = manual_pixel_um
            median_px_height = manual_pixel_um
            st.warning("Keine gültige QuPath-Kalibrierung. Manueller Wert wird verwendet.")
    else:
        median_px_width = manual_pixel_um
        median_px_height = manual_pixel_um
        st.warning("Keine Pixelkalibrierung in CSV gefunden. Manueller Wert wird verwendet.")
else:
    median_px_width = manual_pixel_um
    median_px_height = manual_pixel_um
    st.info(f"Manuelle Kalibrierung: {manual_pixel_um:.4f} µm/Pixel")


def empty_result(image_name, roi_id, roi_area_mm2, n_cells, at2_per_mm2, pixel_width_um, pixel_height_um):
    return {
        "Image": image_name, "ROI_ID": roi_id, "ROI_Area_mm2": roi_area_mm2,
        "AT2_Count": n_cells, "AT2_per_mm2": at2_per_mm2,
        "Clustered_AT2_percent": np.nan, "Clusters_per_mm2": np.nan,
        "Median_AT2_per_Cluster": np.nan, "Median_Cluster_Area_um2": np.nan,
        "Median_Voronoi_Area_um2": np.nan, "Voronoi_CV": np.nan,
        "Voronoi_Mean_um2": np.nan, "Voronoi_SD_um2": np.nan,
        "Voronoi_Cutoff_um2": np.nan, "Voronoi_N_Total": 0,
        "Voronoi_N_Used": 0, "Voronoi_N_Excluded": 0,
        "Voronoi_Excluded_percent": np.nan, "Cluster_Count": 0,
        "PixelWidth_um": pixel_width_um, "PixelHeight_um": pixel_height_um
    }


def analyze_at2(image_df, eps_um, min_samples, calibration_mode, manual_pixel_um, voronoi_mode, manual_voronoi_area_um2):
    image_name = str(image_df["Image"].iloc[0])
    roi_id = str(image_df["ROI_ID"].iloc[0])
    n_cells = len(image_df)
    roi_area_mm2 = float(image_df["ROI_Area_mm2"].iloc[0])

    if calibration_mode == "Automatisch aus QuPath" and "PixelWidth_um" in image_df.columns and "PixelHeight_um" in image_df.columns:
        pw = pd.to_numeric(image_df["PixelWidth_um"], errors="coerce").dropna()
        ph = pd.to_numeric(image_df["PixelHeight_um"], errors="coerce").dropna()
        if len(pw) > 0 and len(ph) > 0 and pw.iloc[0] > 0 and ph.iloc[0] > 0:
            pixel_width_um = float(pw.iloc[0])
            pixel_height_um = float(ph.iloc[0])
        else:
            pixel_width_um = manual_pixel_um
            pixel_height_um = manual_pixel_um
    else:
        pixel_width_um = manual_pixel_um
        pixel_height_um = manual_pixel_um

    at2_per_mm2 = n_cells / roi_area_mm2 if roi_area_mm2 > 0 else np.nan

    if "X_um" in image_df.columns and "Y_um" in image_df.columns and image_df[["X_um", "Y_um"]].notna().all(axis=1).any():
        coordinate_df = image_df[["X_um", "Y_um"]].dropna()
        xy_um = coordinate_df.to_numpy(dtype=float)
    elif "X_pixel" in image_df.columns and "Y_pixel" in image_df.columns:
        coordinate_df = image_df[["X_pixel", "Y_pixel"]].dropna()
        xy_pixel = coordinate_df.to_numpy(dtype=float)
        xy_um = np.column_stack([xy_pixel[:, 0] * pixel_width_um, xy_pixel[:, 1] * pixel_height_um])
    else:
        return empty_result(image_name, roi_id, roi_area_mm2, n_cells, at2_per_mm2, pixel_width_um, pixel_height_um)

    n_coordinates = len(xy_um)
    if n_coordinates == 0:
        return empty_result(image_name, roi_id, roi_area_mm2, n_cells, at2_per_mm2, pixel_width_um, pixel_height_um)

    if n_coordinates < 3:
        return {
            "Image": image_name, "ROI_ID": roi_id, "ROI_Area_mm2": roi_area_mm2,
            "AT2_Count": n_cells, "AT2_per_mm2": at2_per_mm2,
            "Clustered_AT2_percent": 0, "Clusters_per_mm2": 0,
            "Median_AT2_per_Cluster": np.nan, "Median_Cluster_Area_um2": np.nan,
            "Median_Voronoi_Area_um2": np.nan, "Voronoi_CV": np.nan,
            "Voronoi_Mean_um2": np.nan, "Voronoi_SD_um2": np.nan,
            "Voronoi_Cutoff_um2": np.nan, "Voronoi_N_Total": n_coordinates,
            "Voronoi_N_Used": 0, "Voronoi_N_Excluded": 0,
            "Voronoi_Excluded_percent": 0.0, "Cluster_Count": 0,
            "PixelWidth_um": pixel_width_um, "PixelHeight_um": pixel_height_um
        }

    # ========================================================
    # DBSCAN — UNVERÄNDERT GEGENÜBER 3.3
    # ========================================================
    dbscan = DBSCAN(eps=float(eps_um), min_samples=int(min_samples))
    labels = dbscan.fit_predict(xy_um)
    cluster_ids = sorted([x for x in np.unique(labels) if x != -1])
    cluster_count = len(cluster_ids)
    clustered_at2 = int((labels != -1).sum())
    clustered_percent = clustered_at2 / n_coordinates * 100 if n_coordinates > 0 else np.nan
    clusters_per_mm2 = cluster_count / roi_area_mm2 if roi_area_mm2 > 0 else np.nan

    cluster_sizes = []
    cluster_areas = []
    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        cluster_size = int(mask.sum())
        cluster_sizes.append(cluster_size)
        cluster_points = xy_um[mask]
        if cluster_size >= 3:
            try:
                area = float(ConvexHull(cluster_points).volume)
            except Exception:
                area = np.nan
        else:
            area = np.nan
        cluster_areas.append(area)

    median_cluster_size = float(np.median(cluster_sizes)) if cluster_sizes else np.nan
    valid_cluster_areas = [x for x in cluster_areas if not np.isnan(x) and x > 0]
    median_cluster_area = float(np.median(valid_cluster_areas)) if valid_cluster_areas else np.nan

    # ========================================================
    # VORONOI — 3.4
    # Nur dieser Analysezweig erhält einen Flächenfilter.
    # AT2_Count, AT2/mm² und DBSCAN bleiben vollständig unangetastet.
    # Automatik: Mean + 2 SD. Manuell: frei gesetztes Maximum.
    # ========================================================
    raw_areas = []
    try:
        vor = Voronoi(xy_um)
        for region_index in vor.point_region:
            region = vor.regions[region_index]
            if len(region) == 0 or -1 in region:
                raw_areas.append(np.nan)
                continue
            vertices = vor.vertices[region]
            if len(vertices) < 3:
                raw_areas.append(np.nan)
                continue
            x = vertices[:, 0]
            y = vertices[:, 1]
            area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            raw_areas.append(float(area) if np.isfinite(area) and area > 0 else np.nan)

        raw_areas = np.asarray(raw_areas, dtype=float)
        finite_areas = raw_areas[np.isfinite(raw_areas) & (raw_areas > 0)]
        voronoi_n_total = len(finite_areas)

        if len(finite_areas) >= 2:
            voronoi_mean = float(np.mean(finite_areas))
            voronoi_sd = float(np.std(finite_areas, ddof=1))
        elif len(finite_areas) == 1:
            voronoi_mean = float(finite_areas[0])
            voronoi_sd = 0.0
        else:
            voronoi_mean = np.nan
            voronoi_sd = np.nan

        if voronoi_mode == "Automatisch: Mittelwert + 2 SD":
            voronoi_cutoff = voronoi_mean + 2.0 * voronoi_sd if np.isfinite(voronoi_mean) and np.isfinite(voronoi_sd) else np.nan
        else:
            voronoi_cutoff = float(manual_voronoi_area_um2)

        if np.isfinite(voronoi_cutoff):
            used_areas = finite_areas[finite_areas <= voronoi_cutoff]
        else:
            used_areas = np.array([], dtype=float)

        voronoi_n_used = len(used_areas)
        voronoi_n_excluded = voronoi_n_total - voronoi_n_used
        voronoi_excluded_percent = voronoi_n_excluded / voronoi_n_total * 100 if voronoi_n_total > 0 else np.nan
        median_voronoi_area = float(np.median(used_areas)) if voronoi_n_used > 0 else np.nan

        if voronoi_n_used >= 3:
            used_mean = float(np.mean(used_areas))
            used_sd = float(np.std(used_areas, ddof=1))
            voronoi_cv = used_sd / used_mean if used_mean > 0 else np.nan
        else:
            voronoi_cv = np.nan

    except Exception:
        median_voronoi_area = np.nan
        voronoi_cv = np.nan
        voronoi_mean = np.nan
        voronoi_sd = np.nan
        voronoi_cutoff = np.nan
        voronoi_n_total = 0
        voronoi_n_used = 0
        voronoi_n_excluded = 0
        voronoi_excluded_percent = np.nan

    return {
        "Image": image_name, "ROI_ID": roi_id, "ROI_Area_mm2": roi_area_mm2,
        "AT2_Count": n_cells, "AT2_per_mm2": at2_per_mm2,
        "Clustered_AT2_percent": clustered_percent, "Clusters_per_mm2": clusters_per_mm2,
        "Median_AT2_per_Cluster": median_cluster_size, "Median_Cluster_Area_um2": median_cluster_area,
        "Median_Voronoi_Area_um2": median_voronoi_area, "Voronoi_CV": voronoi_cv,
        "Voronoi_Mean_um2": voronoi_mean, "Voronoi_SD_um2": voronoi_sd,
        "Voronoi_Cutoff_um2": voronoi_cutoff, "Voronoi_N_Total": voronoi_n_total,
        "Voronoi_N_Used": voronoi_n_used, "Voronoi_N_Excluded": voronoi_n_excluded,
        "Voronoi_Excluded_percent": voronoi_excluded_percent,
        "Cluster_Count": cluster_count, "PixelWidth_um": pixel_width_um, "PixelHeight_um": pixel_height_um
    }


results = []
grouped = df.groupby(["Image", "ROI_ID"], sort=True)
total = len(grouped)
progress = st.progress(0)
for i, (group_key, image_df) in enumerate(grouped):
    results.append(analyze_at2(image_df, eps_um, min_samples, calibration_mode, manual_pixel_um, voronoi_mode, manual_voronoi_area_um2))
    progress.progress(int((i + 1) / total * 100))
progress.empty()

results_df = pd.DataFrame(results)
desired_columns = [
    "Image", "ROI_ID", "ROI_Area_mm2", "AT2_Count", "AT2_per_mm2",
    "Clustered_AT2_percent", "Cluster_Count", "Clusters_per_mm2",
    "Median_AT2_per_Cluster", "Median_Cluster_Area_um2",
    "Median_Voronoi_Area_um2", "Voronoi_CV", "Voronoi_Mean_um2",
    "Voronoi_SD_um2", "Voronoi_Cutoff_um2", "Voronoi_N_Total",
    "Voronoi_N_Used", "Voronoi_N_Excluded", "Voronoi_Excluded_percent",
    "PixelWidth_um", "PixelHeight_um"
]
results_df = results_df[desired_columns]

st.success(f"{len(results_df)} Image/ROI-Einheiten analysiert.")
st.subheader("🔬 AT2-Ergebnisse")
display_df = results_df.copy()
for col in ["ROI_Area_mm2", "AT2_per_mm2", "Clustered_AT2_percent", "Clusters_per_mm2", "Median_AT2_per_Cluster", "Median_Cluster_Area_um2", "Median_Voronoi_Area_um2", "Voronoi_CV", "Voronoi_Mean_um2", "Voronoi_SD_um2", "Voronoi_Cutoff_um2", "Voronoi_Excluded_percent"]:
    display_df[col] = display_df[col].round(3)
st.dataframe(display_df, use_container_width=True, height=650)

st.subheader("📊 Übersicht")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Image / ROI", len(results_df))
c2.metric("AT2 gesamt", int(results_df["AT2_Count"].sum()))
c3.metric("Ø AT2/mm²", f"{results_df['AT2_per_mm2'].mean():.1f}")
c4.metric("Ø Clustered AT2", f"{results_df['Clustered_AT2_percent'].mean():.1f}%")
c5.metric("Ø Cluster/mm²", f"{results_df['Clusters_per_mm2'].mean():.2f}")
c6.metric("Ø Voronoi CV", f"{results_df['Voronoi_CV'].mean():.2f}")

st.markdown("---")
st.subheader("📥 Ergebnisse speichern")
st.caption("Die CSV wird über den Browser gespeichert. Du kannst beim Speichern selbst den Speicherort auswählen.")
csv_output = results_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(label="📥 AT2-Ergebnisse als CSV speichern", data=csv_output, file_name="AT2_spatial_results.csv", mime="text/csv")

st.markdown("---")
st.subheader("ℹ️ Parameter")
st.markdown("""
**AT2/mm²**

Anzahl Surfactant-C-positiver AT2-Zellen pro mm² analysierter ROI-Fläche.

**Clustered AT2 (%)**

Prozentualer Anteil der AT2-Zellen, die DBSCAN einem Cluster zuordnet.

**Cluster Count**

Anzahl der erkannten AT2-Cluster.

**Cluster/mm²**

Anzahl der AT2-Cluster pro mm².

**Median AT2/Cluster**

Median der AT2-Zellzahl innerhalb der erkannten Cluster.

**Median Clusterfläche (µm²)**

Median der Fläche der konvexen Hülle der AT2-Zellzentren.

**Median Voronoi Area (µm²)**

Median der endlichen Voronoi-Flächen, die nach Anwendung der Voronoi-Ausreißergrenze für die Voronoi-Auswertung verwendet werden.

**Voronoi CV**

Variationskoeffizient der für die Voronoi-Auswertung verwendeten Flächen.

**Voronoi-Ausreißergrenze**

Automatisch: **Mittelwert + 2 SD** der endlichen Voronoi-Flächen der jeweiligen Image/ROI-Einheit.

Benutzerdefiniert: frei wählbares maximales Voronoi-Flächenmaß.

**WICHTIG:** Die Voronoi-Ausreißergrenze betrifft ausschließlich die Voronoi-Auswertung. Keine AT2-Zelle wird aus AT2_Count, AT2/mm², DBSCAN, Cluster Count, Clustergröße oder Clusterfläche entfernt.

**Voronoi N Total**

Anzahl der endlichen Voronoi-Flächen vor dem Voronoi-Filter.

**Voronoi N Used**

Anzahl der Voronoi-Flächen, die für Median und CV verwendet werden.

**Voronoi N Excluded**

Anzahl der Voronoi-Flächen, die ausschließlich aus der Voronoi-Auswertung ausgeschlossen wurden.

**Voronoi Excluded %**

Prozentualer Anteil der Voronoi-Flächen, die ausschließlich aus der Voronoi-Auswertung ausgeschlossen wurden.

Offene Randzellen besitzen keine endliche Voronoi-Fläche und werden daher nicht in die Voronoi-Statistik einbezogen. Sie bleiben selbstverständlich in allen übrigen Analysen erhalten.

**Kalibrierung**

Wenn QuPath `X_um/Y_um` liefert, werden diese Koordinaten direkt verwendet. Wenn nur `X_pixel/Y_pixel` vorhanden sind, werden sie mit der Pixelkalibrierung in µm umgerechnet.

Der `eps`-Wert von DBSCAN wird immer in **µm** angegeben.
""")
