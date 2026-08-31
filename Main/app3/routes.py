import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from . import data_processing as dp

logger = logging.getLogger(__name__)

router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@router.get("/")
def home(request: Request):
    if not request.session.get("is_verified"):
        return RedirectResponse(url="/app2/login", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse("home3.html", {"request": request})


@router.get("/api/layers")
def api_layers():
    """Metadata for every plottable variable — drives the layers panel list."""
    return dp.list_variables()


@router.get("/api/layer/{var_code}/image")
def api_layer_image(var_code: str):
    """
    Serves the cached heatmap texture for one variable.

    First request for a given variable: opens the source NetCDF, rasterizes
    it, and writes ./formatting/<var>.npy + .png (slow, one-time cost).
    Every request after that — including on server restart — just reads the
    cached PNG off disk, so clicking layers in the dashboard is instant.
    """
    # Validate the var code up front, before touching dp's internals, so a
    # genuinely bad code is the ONLY thing that produces a 404 here.
    if var_code not in dp.VARIABLE_META:
        raise HTTPException(status_code=404, detail=f"Unknown variable '{var_code}'")

    try:
        png_bytes = dp.get_cached_png_bytes(var_code)
    except FileNotFoundError as e:
        logger.exception("Source NetCDF file missing for variable '%s'", var_code)
        raise HTTPException(status_code=500, detail=f"Source NetCDF file not found on server: {e}")
    except Exception as e:
        # Catches xarray/matplotlib/PIL errors (bad engine, corrupt file,
        # colormap failure, etc.) and logs the real traceback instead of
        # silently mapping it to an unrelated status code.
        logger.exception("Failed to process variable '%s'", var_code)
        raise HTTPException(status_code=500, detail=f"Failed to process '{var_code}': {e}")

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.post("/api/layer/precompute")
def api_precompute():
    """
    Warms ./formatting/ for every variable in one call. Wire this into your
    app's startup event (see main.py note below) so the cache is already
    warm before the first user click, rather than paying the one-time cost
    on whichever variable happens to be clicked first.
    """
    try:
        done = dp.precompute_all()
    except Exception as e:
        logger.exception("precompute_all failed")
        raise HTTPException(status_code=500, detail=f"Precompute failed: {e}")
    return {"status": "ok", "variables": done}


