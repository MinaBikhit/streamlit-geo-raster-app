from __future__ import annotations

import io
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import streamlit as st
from shapely.geometry import box
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image
import folium
from streamlit_folium import st_folium
from rasterio.warp import transform_bounds


from core.indices import ndvi
from core.zonal import zonal_stats_from_index_raster
from core.utils import ensure_crs

st.set_page_config(page_title="Geo+Raster NDVI Zonal Stats", layout="wide")

st.title("One-click geospatial demo: NDVI + per-polygon mean")
st.caption("Upload GeoJSON polygons + Sentinel-2 B04 (Red) and B08 (NIR) GeoTIFFs, compute NDVI, then compute per-polygon statistics.")

if "results_ready" not in st.session_state:
    st.session_state.results_ready = False
    st.session_state.csv_bytes = None
    st.session_state.geojson_bytes = None
    st.session_state.preview_rgb = None
    st.session_state.last_inputs_sig = None
    st.session_state.ndvi_bounds_4326 = None
    st.session_state.ndvi_png = None
    st.session_state.results_gdf = None


def input_signature(geojson_file, red_file, nir_file):
    # A simple signature so if user uploads different files we reset cached results
    if not (geojson_file and red_file and nir_file):
        return None
    return (
        geojson_file.name, geojson_file.size,
        red_file.name, red_file.size,
        nir_file.name, nir_file.size
    )

with st.sidebar:
    st.header("Inputs")

    geojson_file = st.file_uploader("Upload GeoJSON (polygons preferred)", type=["geojson", "json"])

    st.divider()
    st.subheader("Sentinel-2 bands for NDVI")
    red_file = st.file_uploader("Upload Red band (Sentinel-2 B04) [GeoTIFF]", type=["tif", "tiff"])
    nir_file = st.file_uploader("Upload NIR band (Sentinel-2 B08) [GeoTIFF]", type=["tif", "tiff"])

    compute_btn = st.button("Compute NDVI + zonal stats", type="primary", use_container_width=True)

sig = input_signature(geojson_file, red_file, nir_file)
if sig != st.session_state.last_inputs_sig:
    st.session_state.results_ready = False
    st.session_state.csv_bytes = None
    st.session_state.geojson_bytes = None
    st.session_state.preview_rgb = None
    st.session_state.ndvi_png = None
    st.session_state.ndvi_bounds_4326 = None
    st.session_state.results_gdf = None
    st.session_state.last_inputs_sig = sig


@st.cache_data(show_spinner=False)
def load_geojson(file_bytes: bytes) -> gpd.GeoDataFrame:
    return gpd.read_file(io.BytesIO(file_bytes))


def raster_meta(raster_bytes: bytes):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmp:
        tmp.write(raster_bytes)
        tmp.flush()
        with rasterio.open(tmp.name) as src:
            return {
                "crs": src.crs,
                "count": src.count,
                "width": src.width,
                "height": src.height,
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
                "transform": src.transform,
                "profile": src.profile,
                "bounds": src.bounds,
            }
def validate_same_grid(meta_a: dict, meta_b: dict) -> None:
    # CRS must match
    if str(meta_a["crs"]) != str(meta_b["crs"]):
        raise ValueError(f"CRS mismatch: {meta_a['crs']} vs {meta_b['crs']}")

    # shape must match
    if (meta_a["width"], meta_a["height"]) != (meta_b["width"], meta_b["height"]):
        raise ValueError(
            f"Raster size mismatch: {meta_a['width']}x{meta_a['height']} vs {meta_b['width']}x{meta_b['height']}"
        )

    # transform must match (pixel grid alignment)
    if meta_a["transform"] != meta_b["transform"]:
        raise ValueError("Raster grid mismatch: transforms differ (alignment/resolution differs).")

def compute_ndvi_from_two_tiffs(red_bytes: bytes, nir_bytes: bytes):
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmp_red, \
         tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmp_nir:

        tmp_red.write(red_bytes); tmp_red.flush()
        tmp_nir.write(nir_bytes); tmp_nir.flush()

        with rasterio.open(tmp_red.name) as red_src, rasterio.open(tmp_nir.name) as nir_src:
            red = red_src.read(1).astype(np.float32)
            nir = nir_src.read(1).astype(np.float32)

            # nodata handling (treat nodata as NaN)
            if red_src.nodata is not None:
                red = np.where(red == red_src.nodata, np.nan, red)
            if nir_src.nodata is not None:
                nir = np.where(nir == nir_src.nodata, np.nan, nir)

            index = ndvi(nir=nir, red=red)

            # Use NIR profile as base (either is fine if validated equal)
            profile = nir_src.profile.copy()
            profile.update(dtype="float32", count=1)

            return index, profile

def ndvi_to_rgba_png(ndvi_arr: np.ndarray, cmap_name: str = "RdYlGn") -> bytes:
    """
    Convert NDVI array (-1..1) to an RGBA PNG with NoData/NaN transparent.
    Returns PNG bytes.
    """
    arr = np.clip(ndvi_arr.copy(), -1, 1)
    mask = ~np.isfinite(arr)  # NaN/inf -> transparent

    norm = (arr + 1) / 2.0
    norm = np.clip(norm, 0, 1)

    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm)  # float 0..1 RGBA
    rgba[mask, 3] = 0.0  # transparent where invalid

    img = (rgba * 255).astype(np.uint8)
    pil = Image.fromarray(img, mode="RGBA")
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def raster_bounds_to_wgs84(bounds, src_crs) -> list[list[float]]:
    """
    Return Leaflet bounds: [[south, west], [north, east]] in EPSG:4326.
    """
    if src_crs is None:
        raise ValueError("Raster CRS missing, cannot create basemap overlay.")

    # bounds: left, bottom, right, top in raster CRS
    left, bottom, right, top = bounds
    w, s, e, n = transform_bounds(src_crs, "EPSG:4326", left, bottom, right, top, densify_pts=21)
    return [[s, w], [n, e]]


col1, col2 = st.columns([1, 1])

if geojson_file and red_file and nir_file:
    gdf = load_geojson(geojson_file.getvalue())
    gdf = ensure_crs(gdf)

    red_meta = raster_meta(red_file.getvalue())
    nir_meta = raster_meta(nir_file.getvalue())

    try:
        validate_same_grid(red_meta, nir_meta)
    except ValueError as e:
        st.error(f"Red/NIR rasters are not on the same grid: {e}")
        st.stop()

    # pick one as reference for CRS/bounds etc.
    meta = nir_meta

    with col1:
        st.subheader("Vector (GeoJSON)")
        st.write(f"Features: **{len(gdf)}**")
        st.write(f"CRS: **{gdf.crs}**")
        st.dataframe(gdf.drop(columns="geometry").head(10), use_container_width=True)

    with col2:
        st.subheader("Rasters (GeoTIFF)")
        st.write(f"Red CRS: **{red_meta['crs']}** | NIR CRS: **{nir_meta['crs']}**")
        st.write(f"Size: **{meta['width']} × {meta['height']}**")
        st.write(f"Red nodata: **{red_meta['nodata']}** | NIR nodata: **{nir_meta['nodata']}**")

    if compute_btn:
        if meta["crs"] is None:
            st.error("Raster CRS is missing. Cannot align vector to raster CRS safely.")
            st.stop()

        gdf_proj = gdf.to_crs(meta["crs"])
        raster_bbox = box(*meta["bounds"])

        geom_ok = gdf_proj.geometry.notna() & ~gdf_proj.geometry.is_empty
        intersects = geom_ok & gdf_proj.geometry.intersects(raster_bbox)

        if not bool(intersects.any()):
            st.error(
                "No overlap detected: your GeoJSON geometries do not intersect the raster extent.\n\n"
                "Fix: use a GeoJSON for the same area as the GeoTIFF, or choose the correct raster tile."
            )
            st.stop()

        dropped = int((~intersects).sum())
        kept = int(intersects.sum())
        if dropped > 0:
            st.warning(
                f"{dropped} feature(s) do not overlap the raster extent and will be ignored. Using {kept} feature(s).")

        gdf_proj_use = gdf_proj.loc[intersects].copy()
        st.write(f"Features used for stats: {len(gdf_proj_use)}")

        with st.spinner("Computing NDVI..."):
            index_arr, index_profile = compute_ndvi_from_two_tiffs(
                red_file.getvalue(),
                nir_file.getvalue()
            )

        finite = np.isfinite(index_arr)
        st.write(f"NDVI finite pixels: {int(finite.sum())} / {index_arr.size}")
        if finite.any():
            st.write(f"NDVI range (finite): {float(np.nanmin(index_arr)):.3f} .. {float(np.nanmax(index_arr)):.3f}")
        else:
            st.error(
                "NDVI contains no finite values. Most common causes:\n"
                "- wrong files uploaded (not B04/B08 raw bands)\n"
                "- rasters are all nodata in this area\n"
                "- band files come from different products/tiles\n"
            )
            st.stop()

        with st.spinner("Computing zonal statistics (per polygon)..."):
            stats = zonal_stats_from_index_raster(
                gdf=gdf_proj_use,
                raster_path="in-memory",
                index_array=index_arr,
                index_profile=index_profile,
            )

        stats_df = pd.DataFrame(stats).set_index("feature_index")
        st.write(f"Features with pixels (count>0): {(stats_df['count'] > 0).sum()} / {len(stats_df)}")

        # ✅ build FINAL out_gdf (with stats)
        out_gdf = gdf.loc[intersects.values].copy()
        for c in ["mean", "min", "max", "std", "count"]:
            out_gdf[c] = stats_df[c].values

        out_gdf = out_gdf.reset_index(drop=True)
        out_gdf.insert(0, "feature_id", np.arange(1, len(out_gdf) + 1))

        st.success("Done.")

        # ✅ cache the final table
        st.session_state.results_gdf = out_gdf

        # ✅ cache downloads
        st.session_state.csv_bytes = out_gdf.drop(columns="geometry").to_csv(index=False).encode("utf-8")
        st.session_state.geojson_bytes = out_gdf.to_json().encode("utf-8")

        # ✅ cache NDVI overlay
        st.session_state.ndvi_png = ndvi_to_rgba_png(index_arr, cmap_name="RdYlGn")
        st.session_state.ndvi_bounds_4326 = raster_bounds_to_wgs84(meta["bounds"], meta["crs"])

        st.session_state.results_ready = True

if st.session_state.results_ready and st.session_state.results_gdf is not None:
    st.subheader("Results (per feature)")
    st.dataframe(
        st.session_state.results_gdf.drop(columns="geometry"),
        use_container_width=True
    )

    st.download_button(
        "Download CSV",
        data=st.session_state.csv_bytes,
        file_name="zonal_stats_ndvi.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download GeoJSON (with stats)",
        data=st.session_state.geojson_bytes,
        file_name="zonal_stats_ndvi.geojson",
        mime="application/geo+json",
    )

    st.subheader("NDVI on map (OpenStreetMap + overlay)")

    if st.session_state.ndvi_png is None or st.session_state.ndvi_bounds_4326 is None:
        st.warning("Map overlay not cached yet. Click 'Compute' once.")
        st.stop()

    bounds = st.session_state.ndvi_bounds_4326
    (s, w), (n, e) = bounds
    center = [(s + n) / 2, (w + e) / 2]

    m = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

    b64 = base64.b64encode(st.session_state.ndvi_png).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    folium.raster_layers.ImageOverlay(
        image=data_url,
        bounds=bounds,
        opacity=0.75,
        interactive=True,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    folium.LayerControl().add_to(m)

    st_folium(m, width="100%", height=600)



else:
    st.info("Upload a GeoJSON and two GeoTIFFs: Sentinel-2 B04 (Red) and B08 (NIR).")
