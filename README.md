# streamlit-geo-raster-app

One-click geospatial demo: NDVI from Sentinel/Copernicus + per-polygon mean via GeoPandas/Rasterio, served in Streamlit.

## Features
- Upload GeoJSON vector data (GeoPandas)
- Upload raster GeoTIFF imagery (Rasterio)
- Compute NDVI from two selected bands
- Compute zonal statistics (mean/min/max/std/count) per polygon (vector + raster combined)
- Download results as CSV

## Run (Docker Compose)
```bash
docker compose up --build
