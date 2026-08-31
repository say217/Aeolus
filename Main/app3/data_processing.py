"""
Ocean data processing utilities.

Reads the CMEMS GLORYS12V1 reanalysis NetCDF file and produces two cached
artifacts per variable, written into ./formatting/ (relative to this file):

  1. <var>.npy   Raw RGBA uint8 array, already colour-mapped and alpha-
                 masked. Saved with numpy's binary format so loading it
                 back on a later request takes milliseconds instead of
                 re-reading the NetCDF and re-running the colormap.
  2. <var>.png   The same array encoded as a PNG, so it can be streamed
                 straight to the browser / used as a globe texture with
                 no per-request processing at all.

Call flow:
    process_variable("thetao")   -> slow the first time (opens the .nc,
                                     slices, colour-maps, writes cache)
    get_cached_png_bytes("thetao") -> fast every time after that (disk
                                     read only, no xarray/matplotlib work)

Land / no-data cells are written with alpha = 0 (fully transparent) so
that, once this PNG is composited over a solid black globe on the
frontend, land shows through as pure black and only the ocean data is
visible as a coloured heatmap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.colors import Normalize
from PIL import Image

try:
    # matplotlib >= 3.9
    from matplotlib import colormaps as _mpl_colormaps

    def _get_cmap(name: str):
        return _mpl_colormaps[name]
except ImportError:  # matplotlib < 3.9
    from matplotlib import cm as _mpl_cm

    def _get_cmap(name: str):
        return _mpl_cm.get_cmap(name)



DATA_DIR = Path(__file__).resolve().parent / "formatting"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NC_FILE = Path(__file__).resolve().parent.parent.parent / ".Data" / "cmems_mod_glo_phy_my_0.083deg_P1D-m_1787418903291.nc"

# variable code -> (colormap, display name, units)
VARIABLE_META: dict[str, tuple[str, str, str]] = {
    "thetao":  ("inferno", "Temperature",            "degrees_C"),
    "so":      ("viridis", "Salinity",                "1e-3"),
    "bottomT": ("inferno", "Sea floor temperature",   "degrees_C"),
    "zos":     ("RdBu_r",  "Sea surface height",      "m"),
    "mlotst":  ("viridis", "Mixed layer thickness",   "m"),
    "uo":      ("RdBu_r",  "Eastward velocity",       "m s-1"),
    "vo":      ("RdBu_r",  "Northward velocity",      "m s-1"),
    "siconc":  ("Blues_r", "Ice concentration",       "1"),
    "sithick": ("Blues_r", "Sea ice thickness",       "m"),
    "usi":     ("RdBu_r",  "Ice eastward velocity",   "m s-1"),
    "vsi":     ("RdBu_r",  "Ice northward velocity",  "m s-1"),
}



def _npy_path(var: str) -> Path:
    return DATA_DIR / f"{var}.npy"


def _png_path(var: str) -> Path:
    return DATA_DIR / f"{var}.png"



def _select_2d(da: xr.DataArray) -> np.ndarray:
    """Collapse any extra dims (time, depth) down to a plain (lat, lon) array."""
    for dim in ("time", "depth"):
        if dim in da.dims:
            da = da.isel({dim: 0})
    return da.values


def _rasterize(values: np.ndarray, cmap_name: str) -> np.ndarray:
    """
    Turn a 2D float array into an RGBA uint8 image.

    Valid ocean cells get colour-mapped using the 2nd-98th percentile as
    the colour range (robust to outliers, mirrors `robust=True` in the
    matplotlib snippet this replaces). NaN / land cells get alpha = 0 so
    they render as transparent -> black once composited on the frontend.
    """
    values = np.flipud(values)  # netCDF lat runs south->north; images run top->down
    mask = np.isnan(values)

    if np.all(mask):
        return np.zeros((*values.shape, 4), dtype=np.uint8)

    vmin, vmax = np.nanpercentile(values, [2, 98])
    if vmin == vmax:
        vmax = vmin + 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = _get_cmap(cmap_name)

    rgba = cmap(norm(np.nan_to_num(values, nan=vmin)))
    rgba = (rgba * 255).astype(np.uint8)
    rgba[mask, 3] = 0
    rgba[~mask, 3] = 255
    return rgba

def process_variable(var: str, force: bool = False) -> Path:
    """
    Ensure ./formatting/<var>.npy and .png exist, generating them from the
    source NetCDF only if missing (or force=True). Returns the PNG path.
    """
    if var not in VARIABLE_META:
        raise ValueError(f"Unknown variable '{var}'")

    npy_path, png_path = _npy_path(var), _png_path(var)

    if not force and npy_path.exists() and png_path.exists():
        return png_path

    if not NC_FILE.exists():
        raise FileNotFoundError(f"Source NetCDF not found at {NC_FILE}")

    cmap_name, _, _ = VARIABLE_META[var]

    try:
        # Explicit engine avoids xarray's auto-guessing step entirely —
        # if this raises ImportError, the message tells you exactly which
        # package to install rather than xarray's generic backend-guess error.
        with xr.open_dataset(NC_FILE, engine="netcdf4") as ds:
            raw = _select_2d(ds[var])
    except ImportError as e:
        raise RuntimeError(
            "netCDF4 backend not installed. Run: pip install netCDF4 h5netcdf"
        ) from e

    rgba = _rasterize(raw, cmap_name)

    np.save(npy_path, rgba)
    Image.fromarray(rgba, mode="RGBA").save(png_path, optimize=True)

    return png_path


def get_cached_png_bytes(var: str) -> bytes:
    """Fast path used by the API route: read the cached PNG straight off
    disk. Only falls back to full processing if the cache is cold."""
    if var not in VARIABLE_META:
        raise ValueError(f"Unknown variable '{var}'")
    png_path = _png_path(var)
    if not png_path.exists():
        png_path = process_variable(var)
    return png_path.read_bytes()


def precompute_all(force: bool = False) -> list[str]:
    """Warm the cache for every variable — call once at app startup so the
    very first dashboard click is already fast for every layer."""
    done = []
    for var in VARIABLE_META:
        process_variable(var, force=force)
        done.append(var)
    return done


def list_variables() -> list[dict]:
    out = []
    for code, (_, name, units) in VARIABLE_META.items():
        out.append({
            "code": code,
            "name": name,
            "units": units,
            "cached": _png_path(code).exists(),
        })
    return out