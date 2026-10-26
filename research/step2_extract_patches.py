# =============================================================================
# step2_extract_patches.py
#
# PURPOSE: Build 64×64 thermal image patches for every labelled FIRMS point
#          using the real VIIRS NetCDF granule files in data/nc/.
#
# REQUIRES: Real VNP02IMG + VNP03IMG .nc files in data/nc/
#           Download them with:  python get_viirs_granules.py download
#
# RUN: python research/step2_extract_patches.py
# =============================================================================

import os
import glob
import numpy as np
import pandas as pd

PATCH_SIZE   = 64
LABELLED_CSV = "data/firms_labelled.csv"
NC_DIR       = "data/nc"
OUT_PATCHES  = "data/patches.npy"
OUT_LABELS   = "data/patch_labels.npy"
OUT_META     = "data/patch_meta.csv"
RANDOM_SEED  = 42

np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Load labelled CSV
# ---------------------------------------------------------------------------
if not os.path.exists(LABELLED_CSV):
    raise FileNotFoundError(f"{LABELLED_CSV} not found. Run step1_label_dataset.py first.")

firms = pd.read_csv(LABELLED_CSV)
firms["acq_date"] = firms["acq_date"].astype(str)
firms["acq_time"] = pd.to_numeric(firms["acq_time"], errors="coerce").fillna(0).astype(int)

# Normalise brightness column name (FIRMS SNPP uses 'brightness')
if "brightness" in firms.columns and "bright_ti4" not in firms.columns:
    firms["bright_ti4"] = firms["brightness"]

for col in ["bright_ti4", "frp", "scan", "track"]:
    if col in firms.columns:
        firms[col] = pd.to_numeric(firms[col], errors="coerce")
        firms[col] = firms[col].fillna(firms[col].median())

print(f"Loaded {len(firms):,} labelled detections  "
      f"(wildfire: {(firms.label==0).sum():,}  "
      f"industrial: {(firms.label==1).sum():,})")


# ---------------------------------------------------------------------------
# 2. Build a granule key for each FIRMS row
# ---------------------------------------------------------------------------
def floor_to_6min_key(acq_date: str, acq_time: int) -> str:
    """Return 'YYYYDDD.HHMM' matching the VIIRS granule filename convention."""
    dt  = pd.Timestamp(acq_date)
    doy = dt.timetuple().tm_yday
    hh  = acq_time // 100
    mm  = (acq_time % 100) // 6 * 6
    return f"{dt.year}{doy:03d}.{hh:02d}{mm:02d}"


firms["granule_key"] = firms.apply(
    lambda r: floor_to_6min_key(r["acq_date"], r["acq_time"]), axis=1
)


# ---------------------------------------------------------------------------
# 3. Index available .nc files
# ---------------------------------------------------------------------------
rad_files = sorted(glob.glob(os.path.join(NC_DIR, "VNP02IMG.A*.nc")))
geo_files = sorted(glob.glob(os.path.join(NC_DIR, "VNP03IMG.A*.nc")))

print(f"\nFound {len(rad_files)} VNP02IMG and {len(geo_files)} VNP03IMG files in {NC_DIR}/")

if not rad_files:
    raise RuntimeError(
        f"No VNP02IMG files found in {NC_DIR}/\n"
        "Download them first:\n"
        "  python get_viirs_granules.py download"
    )


def key_from_filename(fname: str) -> str:
    """Extract 'YYYYDDD.HHMM' from VNP02IMG.A2012311.0812.002.*.nc"""
    base  = os.path.basename(fname)
    parts = base.split(".")
    return f"{parts[1][1:]}.{parts[2]}"


rad_index = {key_from_filename(f): f for f in rad_files}
geo_index = {key_from_filename(f): f for f in geo_files}

matched_keys = sorted(set(rad_index) & set(geo_index))
print(f"Matched granule pairs (VNP02+VNP03): {len(matched_keys)}")


# ---------------------------------------------------------------------------
# 4. Per-granule patch extraction
# ---------------------------------------------------------------------------
try:
    import netCDF4 as nc
except ImportError:
    raise RuntimeError("netCDF4 not installed.  pip install netCDF4")

try:
    from scipy.spatial import cKDTree
except ImportError:
    raise RuntimeError("scipy not installed.  pip install scipy")


def extract_patches_from_granule(rad_path, geo_path, subset_df, patch_size=64):
    half = patch_size // 2
    patches_out = {}

    geo_ds  = nc.Dataset(geo_path, "r")
    geo_grp = geo_ds["geolocation_data"] if "geolocation_data" in geo_ds.groups else geo_ds
    lat_var = next((v for v in geo_grp.variables if "lat" in v.lower()), None)
    lon_var = next((v for v in geo_grp.variables if "lon" in v.lower()), None)
    if lat_var is None or lon_var is None:
        geo_ds.close()
        return patches_out

    tile_lat = geo_grp[lat_var][:].data.astype(np.float32)
    tile_lon = geo_grp[lon_var][:].data.astype(np.float32)
    geo_ds.close()

    tile_lat[tile_lat >  90]  = np.nan
    tile_lat[tile_lat < -90]  = np.nan
    tile_lon[tile_lon >  180] = np.nan
    tile_lon[tile_lon < -180] = np.nan

    rad_ds  = nc.Dataset(rad_path, "r")
    rad_grp = rad_ds["observation_data"] if "observation_data" in rad_ds.groups else rad_ds
    band_var = None
    for candidate in ["I04", "I05", "I4", "I5", "M13", "M14"]:
        if candidate in rad_grp.variables:
            band_var = candidate
            break
    if band_var is None:
        for v in rad_grp.variables:
            if rad_grp[v].ndim == 2:
                band_var = v
                break
    if band_var is None:
        rad_ds.close()
        return patches_out

    radiance = rad_grp[band_var][:].data.astype(np.float32)
    try:
        fill = rad_grp[band_var]._FillValue
        radiance[radiance == fill] = np.nan
    except AttributeError:
        pass
    try:
        scale  = rad_grp[band_var].scale_factor
        offset = getattr(rad_grp[band_var], "add_offset", 0.0)
        radiance = radiance * scale + offset
    except AttributeError:
        pass
    rad_ds.close()

    n_rows, n_cols = radiance.shape
    grid_stride    = 8
    sample_rows    = np.unique(np.append(np.arange(0, n_rows, grid_stride), n_rows-1))
    sample_cols    = np.unique(np.append(np.arange(0, n_cols, grid_stride), n_cols-1))
    coarse_lat     = tile_lat[np.ix_(sample_rows, sample_cols)]
    coarse_lon     = tile_lon[np.ix_(sample_rows, sample_cols)]
    coarse_valid   = np.isfinite(coarse_lat) & np.isfinite(coarse_lon)
    if not coarse_valid.any():
        return patches_out

    crr, ccc       = np.meshgrid(sample_rows, sample_cols, indexing="ij")
    coarse_pixels  = np.column_stack((crr[coarse_valid], ccc[coarse_valid]))
    coarse_coords  = np.column_stack((coarse_lat[coarse_valid], coarse_lon[coarse_valid]))
    geo_tree       = cKDTree(coarse_coords)

    lat_min = np.nanmin(tile_lat); lat_max = np.nanmax(tile_lat)
    lon_min = np.nanmin(tile_lon); lon_max = np.nanmax(tile_lon)

    for orig_idx, row in subset_df.iterrows():
        f_lat = float(row["latitude"])
        f_lon = float(row["longitude"])
        if not (np.isfinite(f_lat) and np.isfinite(f_lon) and
                lat_min <= f_lat <= lat_max and
                lon_min <= f_lon <= lon_max):
            continue

        _, nn = geo_tree.query((f_lat, f_lon), k=min(4, len(coarse_coords)))
        nn = np.atleast_1d(nn)

        best_d2 = np.inf
        best_pix = None
        for si in nn:
            cr, cc = coarse_pixels[si]
            r0 = max(0, cr - grid_stride); r1 = min(n_rows, cr + grid_stride + 1)
            c0 = max(0, cc - grid_stride); c1 = min(n_cols, cc + grid_stride + 1)
            ll  = tile_lat[r0:r1, c0:c1]
            lo  = tile_lon[r0:r1, c0:c1]
            val = np.isfinite(ll) & np.isfinite(lo)
            if not val.any():
                continue
            d2 = (ll - f_lat)**2 + (lo - f_lon)**2
            d2[~val] = np.inf
            fi = int(np.argmin(d2))
            if d2.flat[fi] < best_d2:
                best_d2 = d2.flat[fi]
                lr, lc  = np.unravel_index(fi, d2.shape)
                best_pix = (r0 + lr, c0 + lc)

        if best_pix is None:
            continue
        pr, pc = best_pix
        r0, r1 = pr - half, pr + half
        c0, c1 = pc - half, pc + half
        if r0 < 0 or r1 > n_rows or c0 < 0 or c1 > n_cols:
            continue

        patch = radiance[r0:r1, c0:c1].copy()
        med   = np.nanmedian(patch)
        if not np.isfinite(med):
            continue
        patch = np.where(np.isnan(patch), med, patch)
        pmin, pmax = patch.min(), patch.max()
        if pmax > pmin:
            patch = (patch - pmin) / (pmax - pmin)
        else:
            patch = np.zeros_like(patch)
        patches_out[orig_idx] = patch.astype(np.float32)

    return patches_out


# ---------------------------------------------------------------------------
# 5. Run over all matched granules
# ---------------------------------------------------------------------------
groups   = firms.groupby("granule_key")
keys_csv = set(firms["granule_key"].unique())
overlap  = keys_csv & set(matched_keys)

print(f"\nCSV windows: {len(keys_csv)}   "
      f"Downloaded pairs: {len(matched_keys)}   "
      f"Overlap: {len(overlap)}")

if not overlap:
    raise RuntimeError(
        "None of the FIRMS windows match the downloaded granule files.\n"
        "Run:  python get_viirs_granules.py download"
    )

all_patches: dict = {}
for i, key in enumerate(sorted(overlap), 1):
    subset = groups.get_group(key)
    print(f"  [{i}/{len(overlap)}] {key}  ({len(subset)} rows) … ",
          end="", flush=True)
    extracted = extract_patches_from_granule(
        rad_index[key], geo_index[key], subset
    )
    all_patches.update(extracted)
    print(f"{len(extracted)} patches")

print(f"\nTotal real patches: {len(all_patches):,}")

if len(all_patches) < 10:
    raise RuntimeError(
        f"Only {len(all_patches)} patches extracted. "
        "The granule files may not spatially overlap the FIRMS detections."
    )

# ---------------------------------------------------------------------------
# 6. Assemble and save
# ---------------------------------------------------------------------------
valid_idx  = sorted(all_patches.keys())
firms_out  = firms.loc[valid_idx].copy().reset_index(drop=True)
patches_np = np.stack([all_patches[i] for i in valid_idx], axis=0)
labels_np  = firms_out["label"].values.astype(int)

print(f"\nFinal dataset: {len(firms_out):,} patches")
print(f"  Wildfire   (0): {(labels_np==0).sum():,}")
print(f"  Industrial (1): {(labels_np==1).sum():,}")
print(f"  Shape: {patches_np.shape}  Value range: [{patches_np.min():.3f}, {patches_np.max():.3f}]")

os.makedirs("data", exist_ok=True)
np.save(OUT_PATCHES, patches_np)
np.save(OUT_LABELS,  labels_np)
firms_out.to_csv(OUT_META, index=False)

print(f"\nSaved: {OUT_PATCHES}  {OUT_LABELS}  {OUT_META}")
print("Patch source: REAL VIIRS satellite data (VNP02IMG / VNP03IMG)")
