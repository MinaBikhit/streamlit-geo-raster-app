from __future__ import annotations

import io
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import streamlit as st
from shapely.geometry import box

from core.indices import ndvi
from core.zonal import zonal_stats_from_index_raster
from core.utils import ensure_crs

st.set_page_config(page_title="Geo+Raster NDVI Zonal Stats", layout="wide")

st.title("One-click geospatial demo: NDVI + per-polygon mean")
st.caption("Upload GeoJSON polygons + a multi-band GeoTIFF, compute NDVI, then compute per-polygon statistics.")

with st.sidebar:
    st.header("Inputs")

    geojson_file = st.file_uploader("Upload GeoJSON (polygons preferred)", type=["geojson", "json"])
    raster_file = st.file_uploader("Upload Raster (GeoTIFF)", type=["tif", "tiff"])

    st.divider()
    st.subheader("NDVI band mapping (1-indexed)")
    preset = st.selectbox("Preset", ["Custom", "Sentinel-2 (B4=Red, B8=NIR)"], index=1)
    if preset == "Sentinel-2 (B4=Red, B8=NIR)":
        red_band = 4
        nir_band = 8
    else:
        red_band = st.number_input("Red band", min_value=1, value=1, step=1)
        nir_band = st.number_input("NIR band", min_value=1, value=2, step=1)
    st.caption(f"Using bands: Red={red_band}, NIR={nir_band}")
    compute_btn = st.button("Compute NDVI + zonal stats", type="primary", use_container_width=True)

@st.cache_data(show_spinner=False)
def load_geojson(file_bytes: bytes) -> gpd.GeoDataFrame:
    return gpd.read_file(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def load_raster(file_bytes: bytes):
    # Save to a temp-like in-memory object for rasterio; easiest is to write bytes to NamedTemporaryFile.
    # Streamlit caching requires deterministic outputs; we return the bytes and open later.
    return file_bytes

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

def compute_ndvi_array(raster_bytes: bytes, red_b: int, nir_b: int):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmp:
        tmp.write(raster_bytes)
        tmp.flush()
        with rasterio.open(tmp.name) as src:
            red = src.read(red_b).astype(np.float32)
            nir = src.read(nir_b).astype(np.float32)

            # Handle nodata if present
            nodata = src.nodata
            if nodata is not None:
                red = np.where(red == nodata, np.nan, red)
                nir = np.where(nir == nodata, np.nan, nir)

            index = ndvi(nir=nir, red=red)
            profile = src.profile.copy()
            profile.update(dtype="float32", count=1)

            return index, profile

col1, col2 = st.columns([1, 1])

if geojson_file and raster_file:
    gdf = load_geojson(geojson_file.getvalue())
    gdf = ensure_crs(gdf)

    meta = raster_meta(raster_file.getvalue())

    with col1:
        st.subheader("Vector (GeoJSON)")
        st.write(f"Features: **{len(gdf)}**")
        st.write(f"CRS: **{gdf.crs}**")
        st.dataframe(gdf.drop(columns="geometry").head(10), use_container_width=True)

    with col2:
        st.subheader("Raster (GeoTIFF)")
        st.write(f"CRS: **{meta['crs']}**")
        st.write(f"Bands: **{meta['count']}**")
        st.write(f"Size: **{meta['width']} × {meta['height']}**")
        st.write(f"Data type: **{meta['dtype']}** | nodata: **{meta['nodata']}**")

    if compute_btn:
        if meta["crs"] is None:
            st.error("Raster CRS is missing. Cannot align vector to raster CRS safely.")
            st.stop()

        # Reproject vector to raster CRS
        gdf_proj = gdf.to_crs(meta["crs"])

        # --- Overlap check: vector vs raster extent ---
        raster_bbox = box(*meta["bounds"])  # (left, bottom, right, top)

        # robust: handle empty geometries too
        geom_ok = gdf_proj.geometry.notna() & ~gdf_proj.geometry.is_empty
        intersects = geom_ok & gdf_proj.geometry.intersects(raster_bbox)

        if not bool(intersects.any()):
            st.error(
                "No overlap detected: your GeoJSON geometries do not intersect the raster extent.\n\n"
                "Fix: use a GeoJSON for the same area as the GeoTIFF, or choose the correct raster tile."
            )
            st.stop()

        # Compute only for intersecting features
        dropped = int((~intersects).sum())
        kept = int(intersects.sum())
        if dropped > 0:
            st.warning(
                f"{dropped} feature(s) do not overlap the raster extent and will be ignored. Using {kept} feature(s).")

        gdf_proj_use = gdf_proj.loc[intersects].copy()

        with st.spinner("Computing NDVI..."):
            index_arr, index_profile = compute_ndvi_array(
                raster_file.getvalue(),
                red_band,
                nir_band
            )

        with st.spinner("Computing zonal statistics (per polygon)..."):
            stats = zonal_stats_from_index_raster(
                gdf=gdf_proj_use,
                raster_path="in-memory",
                index_array=index_arr,
                index_profile=index_profile,
            )

        stats_df = pd.DataFrame(stats).set_index("feature_index")

        out_gdf = gdf.loc[intersects.values].copy()

        for c in ["mean", "min", "max", "std", "count"]:
            out_gdf[c] = stats_df[c].values

        st.success("Done.")

        st.subheader("Results (per feature)")
        st.dataframe(out_gdf.drop(columns="geometry"), use_container_width=True)

        csv = out_gdf.drop(columns="geometry").to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="zonal_stats_ndvi.csv",
            mime="text/csv",
        )

        # GeoJSON (geometry + attributes + stats)
        geojson_out = out_gdf.to_json().encode("utf-8")
        st.download_button(
            "Download GeoJSON (with stats)",
            data=geojson_out,
            file_name="zonal_stats_ndvi.geojson",
            mime="application/geo+json",
        )

        # Quick preview of NDVI as an image (downsampled)
        st.subheader("NDVI preview (downsampled)")
        preview = np.clip(index_arr.copy(), -1, 1)
        preview = (preview + 1) / 2.0  # map -1..1 to 0..1
        st.image(preview, caption="NDVI scaled to 0..1 for display", use_container_width=True)

else:
    st.info("Upload a GeoJSON and a GeoTIFF to begin.")
