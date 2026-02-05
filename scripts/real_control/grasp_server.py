"""
FastAPI server for the allegro grasp pipeline.

Receives three files per request:
  - 0_color.png
  - 0_camerainfo.npy
  - 0_depth_aligned_rgb.npy

Overwrites the previous copies in kth_experiments/0/, then runs
0_img_pipeline_allegro.sh (with the interactive image-approval loop
auto-answered as "pass").

Once the pipeline finishes, loads the resulting grasp_in_base.npy
and returns the 23-element grasp vector as JSON:
  [qw, qx, qy, qz, x, y, z, joint_0 … joint_15]

Usage:
  conda run -n hamer uvicorn grasp_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import os
import shutil
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Paths – all absolute so the working directory doesn't matter
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent                          # /…/hamer
VIDEO_DIR = REPO_ROOT / "kth_experiments" / "0"
GRASP_OUT  = REPO_ROOT / "kth_experiments" / "grasp_in_base.npy"   # written by visualize_allegro_grasp.py
PIPELINE_SCRIPT = REPO_ROOT / "0_img_pipeline_allegro.sh"

# Expected file names on disk (keys match the client upload field names)
EXPECTED_FILES = {
    "color_png":              "0_color.png",
    "camerainfo_npy":         "0_camerainfo.npy",
    "depth_aligned_rgb_npy":  "0_depth_aligned_rgb.npy",
}

app = FastAPI(title="Allegro Grasp Pipeline Server")

# ---------------------------------------------------------------------------
# Lock: only one pipeline run at a time
# ---------------------------------------------------------------------------
_pipeline_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _save_upload(upload: UploadFile, dest: Path) -> None:
    """Stream an uploaded file to *dest*, overwriting if it exists."""
    content = await upload.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)


async def _run_pipeline() -> None:
    """
    Run 0_img_pipeline_allegro.sh.

    The script contains an interactive read loop that expects "pass" on stdin
    to accept the Gemini-generated grasp image.  We feed "pass\\n" immediately
    so the server can run non-interactively.

    The script also ends with visualize_allegro_grasp.py which calls
    fig.show() (Plotly).  We set DISPLAY= (empty) so Plotly's show() becomes
    a no-op instead of crashing; the .npy output is still written.
    """
    env = os.environ.copy()
    env["DISPLAY"] = ""                 # suppress Plotly fig.show()
    env["MPLBACKEND"] = "Agg"           # suppress any matplotlib display too

    proc = await asyncio.create_subprocess_exec(
        "bash", str(PIPELINE_SCRIPT),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
    )

    # Feed "pass\n" to satisfy the interactive loop, then close stdin.
    stdout, _ = await proc.communicate(input=b"pass\n")

    log_text = stdout.decode(errors="replace") if stdout else ""
    print(log_text)                     # mirror pipeline output to server log

    if proc.returncode != 0:
        raise RuntimeError(
            f"Pipeline exited with code {proc.returncode}.\n"
            f"--- pipeline log (last 3000 chars) ---\n"
            f"{log_text[-3000:]}"
        )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post(
    "/run_grasp",
    response_model=None,
    summary="Upload 3 files and run the allegro grasp pipeline",
)
async def run_grasp(
    color_png:             UploadFile = File(..., description="0_color.png"),
    camerainfo_npy:        UploadFile = File(..., description="0_camerainfo.npy"),
    depth_aligned_rgb_npy: UploadFile = File(..., description="0_depth_aligned_rgb.npy"),
):
    """
    1. Overwrite the three input files in kth_experiments/0/.
    2. Run 0_img_pipeline_allegro.sh end-to-end.
    3. Return the grasp pose in the robot base frame.
    """
    # --- save uploads ---------------------------------------------------
    uploads = {
        "color_png":              color_png,
        "camerainfo_npy":         camerainfo_npy,
        "depth_aligned_rgb_npy":  depth_aligned_rgb_npy,
    }
    for key, upload in uploads.items():
        dest = VIDEO_DIR / EXPECTED_FILES[key]
        await _save_upload(upload, dest)
        print(f"Saved {dest}")

    # --- run pipeline (serialised) --------------------------------------
    async with _pipeline_lock:
        try:
            await _run_pipeline()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # --- load & return result --------------------------------------------
    if not GRASP_OUT.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline finished but {GRASP_OUT} was not produced.",
        )

    grasp_in_base: np.ndarray = np.load(str(GRASP_OUT))  # shape (23,)
    # Return as a flat JSON list of floats
    return JSONResponse(
        content={
            "status": "ok",
            "grasp_in_base": grasp_in_base.tolist(),
            # Convenience: labelled breakdown
            "quat_wxyz":   grasp_in_base[0:4].tolist(),
            "position_xyz": grasp_in_base[4:7].tolist(),
            "joint_angles": grasp_in_base[7:23].tolist(),
        }
    )


# ---------------------------------------------------------------------------
# Health-check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}
