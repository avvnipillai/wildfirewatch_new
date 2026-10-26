# WildfireWatch — Complete Setup & Run Commands

## What changed
- `research/step2_extract_patches.py` — now reads **real 2012-2013 VIIRS .nc files**
  instead of generating synthetic patches.
- `get_viirs_granules.py` — new script to download the matching granule pairs.
- `classifier/ml_engine.py` — serves real patches from the trained patch bank.
- `classifier/views.py` — passes lat/lon/time to the patch endpoint for real lookups.
- `requirements.txt` — added `netCDF4`, `earthaccess`, `Pillow`.

No Docker, Django, or database changes. All existing superuser/auth/migration
logic is unchanged.

---

## Step 0 — Prerequisites (once)

```bash
pip install earthaccess pandas netCDF4 scipy torch scikit-learn matplotlib
```

Create a free NASA Earthdata account:
  https://urs.earthdata.nasa.gov/users/new

---

## Step 1 — Download the matching VIIRS granule pairs

These are the real Suomi-NPP VNP02IMG + VNP03IMG files for 2012-2013
that match the dates in `data/firms_labelled.csv`.

```bash
# See the plan (no download, no login needed)
python get_viirs_granules.py plan --top 50

# Test with 1 pair first (checks your Earthdata login)
python get_viirs_granules.py download --limit 1

# Download all top-50 pairs (~tens of GB — check disk space first)
python get_viirs_granules.py download --top 50
```

Files land in `data/nc/` as `VNP02IMG.A2012311.0812.002.*.nc` etc.

---

## Step 2 — Extract real patches

```bash
cd <repo root>
python research/step2_extract_patches.py
```

Outputs:
- `data/patches.npy`       — (N, 64, 64) float32 real satellite patches
- `data/patch_labels.npy`  — matching labels
- `data/patch_meta.csv`    — FIRMS metadata for each patch

---

## Step 3 — Train PCA + CNN

```bash
cd <repo root>
python research/experiment1_pca.py
python research/experiment2_cnn.py
python research/evaluate_all.py
```

The CNN model is saved to `models/cnn_model.pth`.

---

## Step 4 — Run the Django app (development)

```bash
# Option A: plain Python (no Docker)
pip install -r requirements.txt
python manage.py migrate
python create_admin.py
python manage.py runserver
# Visit http://127.0.0.1:8000/
```

---

## Step 5 — Run with Docker Compose

Create a `.env` file in the project root:
```
DJANGO_SECRET_KEY=your-secret-key-here
POSTGRES_DB=wildfirewatch
POSTGRES_USER=wfw
POSTGRES_PASSWORD=changeme
```

```bash
docker compose up --build
# Visit http://localhost:8000/
# Admin: username=admin, password=admin (set by create_admin.py)
```

---

## Summary of all commands

```bash
# 1. Plan the download
python get_viirs_granules.py plan --top 50

# 2. Download (get Earthdata account first)
python get_viirs_granules.py download --top 50

# 3. Extract real patches
python research/step2_extract_patches.py

# 4. Train
python research/experiment1_pca.py
python research/experiment2_cnn.py
python research/evaluate_all.py

# 5a. Run locally
python manage.py migrate && python create_admin.py && python manage.py runserver

# 5b. Or with Docker
docker compose up --build
```
