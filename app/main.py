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
st.caption("Upload GeoJSON polygons + Sentinel-2 B04 (Red) and B08 (NIR) GeoTIFFs, compute NDVI, then compute per-polygon statistics.")


with st.sidebar:
    st.header("Inputs")

    geojson_file = st.file_uploader("Upload GeoJSON (polygons preferred)", type=["geojson", "json"])

    st.divider()
    st.subheader("Sentinel-2 bands for NDVI")
    red_file = st.file_uploader("Upload Red band (Sentinel-2 B04) [GeoTIFF]", type=["tif", "tiff"])
    nir_file = st.file_uploader("Upload NIR band (Sentinel-2 B08) [GeoTIFF]", type=["tif", "tiff"])

    compute_btn = st.button("Compute NDVI + zonal stats", type="primary", use_container_width=True)


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
        preview = (preview + 1) / 2.0  # -1..1 -> 0..1
        preview = np.nan_to_num(preview, nan=0.0)

        # Convert to 8-bit so Streamlit displays nicely
        preview_u8 = (preview * 255).astype(np.uint8)
        st.image(preview_u8, caption="NDVI scaled to 0..255 for display", use_container_width=True)


else:
    st.info("Upload a GeoJSON and two GeoTIFFs: Sentinel-2 B04 (Red) and B08 (NIR).")
