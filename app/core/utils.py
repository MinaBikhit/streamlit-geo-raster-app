from __future__ import annotations
import geopandas as gpd

def ensure_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # GeoJSON often comes as EPSG:4326 but may be missing CRS
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    return gdf
