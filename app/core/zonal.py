from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping


@dataclass
class ZonalStats:
    mean: float | None
    min: float | None
    max: float | None
    std: float | None
    count: int

def _stats(arr: np.ndarray) -> ZonalStats:
    # arr is float with nans
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return ZonalStats(mean=None, min=None, max=None, std=None, count=0)
    return ZonalStats(
        mean=float(valid.mean()),
        min=float(valid.min()),
        max=float(valid.max()),
        std=float(valid.std(ddof=0)),
        count=int(valid.size),
    )

def zonal_stats_from_index_raster(
    gdf: gpd.GeoDataFrame,
    raster_path: str,
    index_array: np.ndarray,
    index_profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Compute per-feature stats by masking the computed index raster with each geometry.
    We write nothing to disk: we reuse the raster georeferencing from index_profile.
    """

    # Create an in-memory dataset-like view via rasterio.io.MemoryFile
    from rasterio.io import MemoryFile

    profile = index_profile.copy()
    profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
    )

    results: List[Dict[str, Any]] = []

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(index_array.astype("float32"), 1)

            for i, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    zs = ZonalStats(mean=None, min=None, max=None, std=None, count=0)
                else:
                    out, _ = mask(dataset, [mapping(geom)], crop=True, filled=True)
                    arr = out[0].astype("float32")

                    # rasterio.mask with filled=True may use dataset.nodata; we set it to nan
                    zs = _stats(arr)

                results.append(
                    {
                        "feature_index": i,
                        "mean": zs.mean,
                        "min": zs.min,
                        "max": zs.max,
                        "std": zs.std,
                        "count": zs.count,
                    }
                )

    return results
