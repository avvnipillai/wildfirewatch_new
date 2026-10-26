"""
ml_engine.py — Core ML prediction logic for WildfireWatch.

Loads trained CNN model once at startup (cached in module scope).
Uses REAL satellite patches: reads data/patches.npy + data/patch_meta.csv
at startup, keyed by (acq_date, acq_time, latitude, longitude).

Classification logic:
  - Ground truth label from firms_labelled.csv (power plant proximity)
  - CNN probability used as confidence score (ROC-AUC 0.71)
  - PCA reconstruction error used as secondary signal
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ── CNN Architecture (must match experiment2_cnn.py) ─────────────────────────
class ThermalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(True), nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.AdaptiveAvgPool2d(4)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.head(x).squeeze(1)

    def extract_features(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        return self.head[:3](x)


# ── Module-level singletons ──────────────────────────────────────────────────
_cnn     = None
_device  = torch.device("cpu")
_patches = None   # (N, 64, 64) float32 – real satellite patches
_meta    = None   # DataFrame with lat, lon, acq_date, acq_time, granule_key


def _base_dir():
    return os.path.join(os.path.dirname(__file__), "..")


def get_cnn():
    global _cnn
    if _cnn is None:
        _cnn = ThermalCNN().to(_device)
        weights = os.path.join(_base_dir(), "models", "cnn_model.pth")
        if os.path.exists(weights):
            _cnn.load_state_dict(
                torch.load(weights, map_location=_device, weights_only=True)
            )
        _cnn.eval()
    return _cnn


def _load_patch_bank():
    """Load the pre-extracted patch bank once."""
    global _patches, _meta
    if _patches is not None:
        return

    patches_path = os.path.join(_base_dir(), "data", "patches.npy")
    meta_path    = os.path.join(_base_dir(), "data", "patch_meta.csv")

    if os.path.exists(patches_path) and os.path.exists(meta_path):
        _patches = np.load(patches_path, mmap_mode="r")   # (N,64,64) float32
        _meta    = pd.read_csv(meta_path)
        _meta["acq_date"] = _meta["acq_date"].astype(str)
        _meta["acq_time"] = pd.to_numeric(_meta["acq_time"], errors="coerce").fillna(0).astype(int)


def _floor_to_6min_key(acq_date: str, acq_time: int) -> str:
    dt  = pd.Timestamp(str(acq_date))
    doy = dt.timetuple().tm_yday
    hh  = int(acq_time) // 100
    mm  = (int(acq_time) % 100) // 6 * 6
    return f"{dt.year}{doy:03d}.{hh:02d}{mm:02d}"


def _lookup_patch(row_lat: float, row_lon: float,
                   acq_date: str, acq_time: int) -> np.ndarray | None:
    """
    Find the stored patch closest (lat/lon) within the same 6-min granule window.
    Returns a (64,64) float32 array, or None if not found.
    """
    if _patches is None or _meta is None:
        return None

    key = _floor_to_6min_key(acq_date, acq_time)

    # Filter meta to matching granule window
    if "granule_key" in _meta.columns:
        mask = _meta["granule_key"] == key
    else:
        mask = (
            (_meta["acq_date"] == str(acq_date)) &
            (_meta["acq_time"] == int(acq_time))
        )

    sub = _meta[mask]
    if len(sub) == 0:
        return None

    lats  = sub["latitude"].values
    lons  = sub["longitude"].values
    dists = (lats - row_lat) ** 2 + (lons - row_lon) ** 2
    best  = int(np.argmin(dists))

    if dists[best] > 0.01:   # > ~1 km: too far, don't use
        return None

    patch_idx = sub.index[best]
    return np.array(_patches[patch_idx], copy=True)


def get_real_patch(brightness: float, frp: float,
                   lat: float = None, lon: float = None,
                   acq_date: str = None, acq_time: int = None) -> np.ndarray:
    """
    Return the best available 64x64 patch for a detection.
    Tries real satellite bank first; otherwise makes a synthetic one.
    """
    _load_patch_bank()

    if lat is not None and acq_date is not None:
        p = _lookup_patch(lat, lon, acq_date, acq_time or 0)
        if p is not None:
            return p

    # Fallback: generate synthetic patch from FIRMS scalars
    return _make_synthetic_patch(brightness, frp)


def _make_synthetic_patch(brightness: float, frp: float, seed: int = 0) -> np.ndarray:
    """Generate a synthetic 64x64 patch from FIRMS scalar values (fallback only)."""
    SZ   = 64; half = SZ // 2
    ys, xs = np.mgrid[0:SZ, 0:SZ]
    dist   = np.sqrt((xs - half) ** 2 + (ys - half) ** 2).astype(np.float32)
    b = np.clip((float(brightness) - 200) / 200, 0, 1)
    f = np.clip(float(frp) / 500, 0, 1)
    rng = np.random.default_rng(seed=seed)
    if f > 0.3 and b > 0.6:
        sigma = rng.uniform(3, 6)
        bg    = rng.uniform(0.05, 0.15, (SZ, SZ))
        peak  = 0.6 + 0.4 * b
        patch = bg + peak * np.exp(-dist ** 2 / (2 * sigma ** 2))
        patch += rng.normal(0, 0.02, patch.shape)
    else:
        sigma = rng.uniform(10, 20)
        noise = np.kron(rng.uniform(0, 1, (SZ // 4, SZ // 4)), np.ones((4, 4)))[:SZ, :SZ]
        bg    = rng.uniform(0.1, 0.3, (SZ, SZ))
        peak  = 0.3 + 0.5 * b
        patch = bg + peak * np.exp(-dist ** 2 / (2 * sigma ** 2)) * (0.5 + 0.5 * noise)
        patch += rng.normal(0, 0.04, patch.shape)
    patch = np.clip(patch, 0, 1)
    mn, mx = patch.min(), patch.max()
    return ((patch - mn) / (mx - mn + 1e-8)).astype(np.float32)


# kept for backwards-compat with views.api_patch
def make_patch(brightness: float, frp: float, seed: int = 0,
               lat: float = None, lon: float = None,
               acq_date: str = None, acq_time: int = None) -> np.ndarray:
    return get_real_patch(brightness, frp, lat, lon, acq_date, acq_time)


# ── Main prediction function ─────────────────────────────────────────────────
def classify_detections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  DataFrame with FIRMS columns (including 'label' ground truth column)
    Output: DataFrame with prediction columns added

    Label source (in order of priority):
      1. Ground truth 'label' column from firms_labelled.csv
         (0 = wildfire, 1 = industrial, based on 5km proximity to power plants)
         This is the scientifically correct label.
      2. CNN model probability (ROC-AUC 0.71) as fallback if no label column.

    CNN probability is always shown as the confidence score in the map popup,
    since CNN (ROC 0.71) outperforms PCA (ROC 0.62).
    """
    _load_patch_bank()

    df = df.copy()
    if "bright_ti4" not in df.columns and "brightness" in df.columns:
        df["bright_ti4"] = df["brightness"]

    n   = len(df)
    cnn = get_cnn()

    # ── 1. Build patch array (real satellite patches where available) ──────────
    patches = []
    for _, r in df.iterrows():
        bright = r.get("bright_ti4", 300)
        frp_v  = r.get("frp", 10)
        lat_v  = r.get("latitude")
        lon_v  = r.get("longitude")
        date_v = str(r.get("acq_date", ""))
        time_v = int(r.get("acq_time", 0)) if "acq_time" in r.index else 0
        p = get_real_patch(bright, frp_v, lat_v, lon_v, date_v, time_v)
        patches.append(p)

    patches = np.array(patches)   # (N, 64, 64)

    # ── 2. CNN inference (primary model — ROC-AUC 0.71) ──────────────────────
    t = torch.tensor(patches[:, np.newaxis], dtype=torch.float32)
    with torch.no_grad():
        logits = cnn(t).numpy()
    cnn_prob = 1 / (1 + np.exp(-logits))   # sigmoid → probability of industrial

    # ── 3. PCA reconstruction error (secondary signal — ROC-AUC 0.62) ────────
    X_flat  = patches.reshape(n, -1)
    n_comp  = min(50, n - 1, X_flat.shape[1])
    pca     = PCA(n_components=n_comp)
    X_pca   = pca.fit_transform(X_flat)
    X_back  = pca.inverse_transform(X_pca)
    recon   = np.mean((X_flat - X_back) ** 2, axis=1)
    pca_prob = 1 / (1 + np.exp(-(recon - recon.mean()) * 10))

    # ── 4. Ensemble (CNN weighted higher as it has better ROC-AUC) ────────────
    ensemble = 0.4 * pca_prob + 0.6 * cnn_prob

    # ── 5. Assign labels ──────────────────────────────────────────────────────
    # PRIMARY: use ground truth label from firms_labelled.csv
    # This is the correct scientific label (power plant proximity check).
    # The CNN confidence is shown in the popup to indicate model certainty.
    if "label" in df.columns:
        raw_label = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        final_label = np.where(raw_label == 1, "industrial", "wildfire")
    else:
        # Fallback: use CNN-dominant ensemble if no ground truth available
        final_label = np.where(ensemble > 0.5, "industrial", "wildfire")

    # ── 6. Confidence score — based on CNN probability (best model) ───────────
    # Scale CNN output: 100% = CNN is very certain, 0% = uncertain (near 0.5)
    confidence = np.abs(cnn_prob - 0.5) * 200   # 0–100 range
    confidence = np.clip(np.round(confidence, 1), 0, 100)

    # ── 7. Build output ───────────────────────────────────────────────────────
    out = df.copy()
    out["pca_prob"]   = np.round(pca_prob, 4)
    out["cnn_prob"]   = np.round(cnn_prob, 4)
    out["ensemble"]   = np.round(ensemble, 4)
    out["label"]      = final_label
    out["confidence"] = confidence
    out["patch_seed"] = np.arange(n)
    return out