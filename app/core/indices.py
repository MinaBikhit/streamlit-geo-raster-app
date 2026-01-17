import numpy as np

def safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.full_like(numer, np.nan, dtype=np.float32)
    mask = denom != 0
    out[mask] = (numer[mask] / denom[mask]).astype(np.float32)
    return out

def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return safe_divide(nir - red, nir + red)
