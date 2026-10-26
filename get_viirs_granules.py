#!/usr/bin/env python3
"""
get_viirs_granules.py
=====================
Download matching VNP02IMG + VNP03IMG granule pairs for every 6-minute
satellite window in data/firms_labelled.csv.

Uses the LAADS DAAC REST API (works for 2012-2013 Suomi-NPP data).

Usage
-----
  python get_viirs_granules.py plan   [--top N]
  python get_viirs_granules.py download [--top N] [--limit N]

Getting your LAADS App Key (free, 1 minute)
-------------------------------------------
  1. Go to https://ladsweb.modaps.eosdis.nasa.gov/
  2. Sign in (same Earthdata account) or register free.
  3. Click your name → Profile → "App Keys" tab → Generate Key.
  4. Copy the key and paste when the script asks, OR set:
       export LAADS_TOKEN=your_key_here

Requirements
------------
  pip install pandas requests tqdm
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

FIRMS_CSV = "data/firms_labelled.csv"
PLAN_CSV  = "granules_to_download.csv"
NC_DIR    = "data/nc"

# LAADS DAAC archive root for VNP02IMG/VNP03IMG Collection 2
# Archive set 5110 = VIIRS Level-1 Collection 2
LAADS_BASE = "https://ladsweb.modaps.eosdis.nasa.gov"
ARCHIVE    = "/archive/allData/5110"


# ── helpers ──────────────────────────────────────────────────────────────────

def floor_to_6min_key(acq_date: str, acq_time: int) -> str:
    dt  = pd.Timestamp(str(acq_date))
    doy = dt.timetuple().tm_yday
    hh  = int(acq_time) // 100
    mm  = (int(acq_time) % 100) // 6 * 6
    return f"{dt.year}{doy:03d}.{hh:02d}{mm:02d}"


def build_window_list(top_n: int = 50) -> pd.DataFrame:
    print(f"Reading {FIRMS_CSV} …")
    df = pd.read_csv(FIRMS_CSV)
    df["acq_date"] = df["acq_date"].astype(str)
    df["acq_time"] = pd.to_numeric(df["acq_time"], errors="coerce").fillna(0).astype(int)
    df["granule_key"] = df.apply(
        lambda r: floor_to_6min_key(r["acq_date"], r["acq_time"]), axis=1
    )

    grp = df.groupby("granule_key").agg(
        n_detections=("latitude", "count"),
        n_industrial=("label", "sum"),
    ).reset_index()
    grp = grp.sort_values("n_detections", ascending=False).reset_index(drop=True)
    top = grp.head(top_n).copy()

    def parse_key(k):
        # 'YYYYDDD.HHMM'
        yyyyddd, hhmm = k.split(".")
        year = int(yyyyddd[:4])
        doy  = int(yyyyddd[4:])
        hh   = int(hhmm[:2])
        mm   = int(hhmm[2:])
        return pd.Series({"year": year, "doy": doy, "hh": hh, "mm": mm})

    top[["year","doy","hh","mm"]] = top["granule_key"].apply(parse_key)
    top["hhmm"] = top.apply(lambda r: f"{r.hh:02d}{r.mm:02d}", axis=1)

    all_det = grp["n_detections"].sum()
    top_det = top["n_detections"].sum()
    top_ind = top["n_industrial"].sum()
    print(f"\nTop {top_n} windows cover {top_det:,}/{all_det:,} detections "
          f"({100*top_det/all_det:.0f}%)  |  industrial: {top_ind:,}")
    return top


# ── LAADS DAAC download ──────────────────────────────────────────────────────

def get_token() -> str:
    token = os.environ.get("LAADS_TOKEN", "").strip()
    if token:
        return token
    print("\nYou need a free LAADS App Key.")
    print("  1. Go to https://ladsweb.modaps.eosdis.nasa.gov/")
    print("  2. Sign in → Profile → App Keys → Generate Key.")
    print("  3. Paste it below (or set LAADS_TOKEN env var to skip this prompt).")
    token = input("Paste your LAADS App Key: ").strip()
    if not token:
        sys.exit("No token provided. Exiting.")
    return token


def list_granule_files(product: str, year: int, doy: int, token: str) -> list[str]:
    """Return list of .nc filenames for a given product/year/doy from LAADS."""
    url = f"{LAADS_BASE}{ARCHIVE}/{product}/{year}/{doy:03d}.json"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    # Response is a list of dicts with 'name' key
    return [item["name"] for item in r.json()]


def download_file(product: str, year: int, doy: int,
                   filename: str, dest_dir: str, token: str) -> bool:
    url      = f"{LAADS_BASE}{ARCHIVE}/{product}/{year}/{doy:03d}/{filename}"
    dest     = os.path.join(dest_dir, filename)
    headers  = {"Authorization": f"Bearer {token}"}

    print(f"    Downloading {filename} … ", end="", flush=True)
    try:
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            written = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    written += len(chunk)
            mb = written / 1e6
            print(f"OK  ({mb:.0f} MB)")
            return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def download_granules(plan: pd.DataFrame, limit=None, token: str = ""):
    os.makedirs(NC_DIR, exist_ok=True)
    if not token:
        token = get_token()

    rows = plan if limit is None else plan.head(limit)
    ok, fail = 0, 0

    for _, row in rows.iterrows():
        key  = row["granule_key"]
        year = int(row["year"])
        doy  = int(row["doy"])
        hhmm = str(row["hhmm"])

        for product in ("VNP02IMG", "VNP03IMG"):
            # Check if already downloaded
            existing = [f for f in os.listdir(NC_DIR)
                        if f.startswith(f"{product}.A{year}{doy:03d}.{hhmm}")]
            if existing:
                print(f"  Already have {existing[0]}, skipping.")
                ok += 1
                continue

            print(f"  [{product}] {key} — listing LAADS … ", end="", flush=True)
            try:
                names = list_granule_files(product, year, doy, token)
            except Exception as exc:
                print(f"LISTING FAILED: {exc}")
                fail += 1
                continue

            # Find the granule that matches our HHMM
            matches = [n for n in names if f".{hhmm}." in n and n.endswith(".nc")]
            if not matches:
                print(f"no file with HHMM={hhmm} found "
                      f"(available: {[n.split('.')[2] for n in names[:5]]})")
                fail += 1
                continue

            fname = matches[0]
            print(f"found {fname}")
            if download_file(product, year, doy, fname, NC_DIR, token):
                ok += 1
            else:
                fail += 1

        time.sleep(0.3)   # be polite to LAADS

    print(f"\nDone. Files downloaded: {ok}  failures: {fail}")
    print(f"Files are in: {NC_DIR}/")
    if fail > 0:
        print("\nFor any NOT FOUND files, try the web browser fallback:")
        print("  https://ladsweb.modaps.eosdis.nasa.gov/search/")
        print("  Search for VNP02IMG, Collection 2, enter the date and time.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plan / download VIIRS VNP02IMG+VNP03IMG pairs from LAADS DAAC"
    )
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("plan", help="Show what would be downloaded (no login)")
    p.add_argument("--top", type=int, default=50)

    p = sub.add_parser("download", help="Download granule pairs from LAADS DAAC")
    p.add_argument("--top",   type=int, default=50)
    p.add_argument("--limit", type=int, default=None,
                   help="Download only this many pairs (for testing)")

    args = parser.parse_args()

    if args.cmd == "plan":
        top = build_window_list(top_n=args.top)
        top.to_csv(PLAN_CSV, index=False)
        print(f"\nPlan saved to {PLAN_CSV}")
        print("\nNext step:")
        print("  python get_viirs_granules.py download --limit 1")

    elif args.cmd == "download":
        top = build_window_list(top_n=args.top)
        top.to_csv(PLAN_CSV, index=False)
        download_granules(top, limit=args.limit)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
