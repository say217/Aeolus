from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


NC_FILES = {
    "bay-of-bengal": Path(".Data/prediction_output/ocean_subsurface_temperature_prediction.nc"),
    "indian-ocean": Path(".Data/prediction_output/prediction_output_v2.nc"),
}

# Approximate prediction-coverage bounding boxes, used to draw highlight
# rectangles on the Leaflet map. Adjust these to match the actual grid
# extents baked into each .nc file if they differ.
REGION_BOUNDS = {
    "bay-of-bengal": {
        "label": "Bay of Bengal",
        "lat_min": 5.0,
        "lat_max": 23.0,
        "lon_min": 78.0,
        "lon_max": 100.0,
    },
    "indian-ocean": {
        "label": "Indian Ocean",
        "lat_min": -30.0,
        "lat_max": 30.0,
        "lon_min": 30.0,
        "lon_max": 120.0,
    },
}

# Candidate variable names for the predicted temperature field, tried in order.
TEMP_VAR_CANDIDATES = ["thetao_pred", "thetao", "temperature", "sea_water_potential_temperature"]

_dataset_cache: dict[str, xr.Dataset] = {}


def _load_dataset(region: str) -> xr.Dataset:
    if region in _dataset_cache:
        return _dataset_cache[region]

    path = NC_FILES.get(region)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown region '{region}'")
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Prediction file for '{region}' not found at {path}",
        )

    ds = xr.open_dataset(path)
    _dataset_cache[region] = ds
    return ds


def _find_temp_var(ds: xr.Dataset) -> str:
    for cand in TEMP_VAR_CANDIDATES:
        if cand in ds.data_vars:
            return cand
    # Fallback: first data var that carries a depth-like dimension.
    for name, da in ds.data_vars.items():
        if any(d in da.dims for d in ("depth", "z", "level")):
            return name
    raise HTTPException(status_code=500, detail="No temperature variable found in dataset")


def _dim_name(da_or_ds, candidates: list[str]) -> Optional[str]:
    dims = set(getattr(da_or_ds, "dims", []))
    coords = set(getattr(da_or_ds, "coords", {}).keys())
    for c in candidates:
        if c in dims or c in coords:
            return c
    return None


def _to_json_safe(arr: np.ndarray) -> list:
    """NaN-safe conversion of a numpy array to nested python lists (NaN -> None)."""
    obj = np.where(np.isfinite(arr), arr, None)
    return obj.tolist()


# ---------------------------------------------------------------
# 2. Page route
# ---------------------------------------------------------------
@router.get("/")
def home(request: Request):
    if not request.session.get("is_verified"):
        return RedirectResponse(url="/app2/login", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse("home4.html", {"request": request})


# ---------------------------------------------------------------
# 3. API: region bounding boxes (for the Leaflet highlight map)
# ---------------------------------------------------------------
@router.get("/api/regions")
def get_regions(request: Request):
    if not request.session.get("is_verified"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return JSONResponse(REGION_BOUNDS)


# ---------------------------------------------------------------
# 4. API: subsurface temperature cuboid for one region
# ---------------------------------------------------------------
@router.get("/api/subsurface/{region}")
def get_subsurface(
    region: str,
    request: Request,
    time_idx: int = Query(0, ge=0, description="Timestep index into the prediction file"),
    step: int = Query(3, ge=1, le=10, description="Spatial downsample stride for lat/lon"),
):
    if not request.session.get("is_verified"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    if region not in NC_FILES:
        raise HTTPException(status_code=404, detail=f"Unknown region '{region}'")

    ds = _load_dataset(region)
    var_name = _find_temp_var(ds)
    da = ds[var_name]

    lat_dim = _dim_name(da, ["lat", "latitude"])
    lon_dim = _dim_name(da, ["lon", "longitude"])
    depth_dim = _dim_name(da, ["depth", "z", "level"])
    time_dim = _dim_name(da, ["time"])

    if lat_dim is None or lon_dim is None:
        raise HTTPException(status_code=500, detail="Dataset is missing lat/lon dimensions")
    if depth_dim is None:
        raise HTTPException(status_code=500, detail="Dataset has no depth dimension")

    if time_dim and time_dim in da.dims:
        n_time = da.sizes[time_dim]
        time_idx = min(time_idx, n_time - 1)
        da = da.isel({time_dim: time_idx})

    # Spatial downsample so the payload stays small and Plotly stays smooth.
    da = da.isel({lat_dim: slice(None, None, step), lon_dim: slice(None, None, step)})

    lat_vals = ds[lat_dim].values[::step]
    lon_vals = ds[lon_dim].values[::step]
    depth_vals = ds[depth_dim].values

    arr = da.transpose(depth_dim, lat_dim, lon_dim).values.astype(float)

    vmin = float(np.nanmin(arr)) if np.isfinite(arr).any() else None
    vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else None

    payload = {
        "region": region,
        "label": REGION_BOUNDS.get(region, {}).get("label", region),
        "variable": var_name,
        "depths": [float(d) for d in depth_vals],
        "lat": [float(x) for x in lat_vals],
        "lon": [float(x) for x in lon_vals],
        "vmin": vmin,
        "vmax": vmax,
        "bounds": REGION_BOUNDS.get(region),
        "data": _to_json_safe(arr),  # shape: [depth][lat][lon], NaN -> null
    }
    return JSONResponse(payload)