"""PERSON A - Data loading, cleaning, feature engineering and train/test split.

Run:  python -m src.preprocessing
"""
from __future__ import annotations
import argparse
import json
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

UNIT_MAP = {
    "marla": config.SQFT_PER_MARLA, "kanal": config.SQFT_PER_KANAL,
    "sqft": 1.0, "sqfeet": 1.0, "sqyd": 9.0, "sqyards": 9.0, "sqm": 10.7639,
}


# ----------------------------------------------------------------- helpers
def parse_area(value) -> float:
    """'5 Marla' -> 1361.25 sq ft ; '1,200 Sq. Ft.' -> 1200 ; 950 -> 950."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    match = re.match(r"\s*([\d,\.]+)\s*([A-Za-z\.\s]*)", str(value))
    if not match:
        return np.nan
    try:
        number = float(match.group(1).replace(",", ""))
    except ValueError:
        return np.nan
    unit = re.sub(r"[^a-z]", "", match.group(2).lower())
    return number * UNIT_MAP.get(unit, 1.0)


def haversine_km(lat1, lon1, lat2, lon2):
    """Distance in km between two lat/lon points (works on arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def add_features(df: pd.DataFrame, city_centers: dict) -> pd.DataFrame:
    """Feature engineering. Used for training AND for the web app (same logic)."""
    out = df.copy()
    out["log_area"] = np.log1p(out["area_sqft"])
    out["area_marla"] = out["area_sqft"] / config.SQFT_PER_MARLA
    out["total_rooms"] = out["bedrooms"] + out["baths"]
    out["bath_per_bed"] = out["baths"] / out["bedrooms"].clip(lower=1)
    center_lat = out["city"].map(lambda c: city_centers[c][0])
    center_lon = out["city"].map(lambda c: city_centers[c][1])
    out["dist_to_center_km"] = haversine_km(
        out["latitude"], out["longitude"], center_lat, center_lon)
    out["locality"] = out["location"].astype(str) + " | " + out["city"].astype(str)
    return out


# ---------------------------------------------------------------- cleaning
def load_raw(path=None) -> pd.DataFrame:
    path = path or (config.RAW_CSV if config.RAW_CSV.exists() else config.SAMPLE_CSV)
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def clean_data(raw: pd.DataFrame, purpose: str = "sale"):
    """Return (clean_df, city_centers, cleaning_log). purpose = "sale" or "rent"."""
    log = []

    def record(step, frame, note=""):
        log.append({"step": step, "rows": len(frame), "note": note})
        print(f"{step:<42} rows = {len(frame):>8,}  {note}")

    df = raw.copy()
    record("0. Raw data", df)

    # Zameen files mix "For Sale" and "For Rent": keep only the one we model
    if "purpose" in df.columns:
        word = "rent" if purpose == "rent" else "sale"
        df = df[df["purpose"].astype("string").str.contains(word, case=False, na=False)]
        record(f"0b. Keep only 'For {word.title()}' listings", df)
    elif purpose == "rent":
        raise ValueError("This file has no 'purpose' column, so rent listings cannot be found.")

    # 1. Remove duplicate listings
    if "property_id" in df.columns:
        df = df.drop_duplicates(subset="property_id")
    # 2. Area -> numeric square feet
    if "total_area" in df.columns:
        df["area_sqft"] = pd.to_numeric(df["total_area"], errors="coerce")
        if "area" in df.columns:
            df["area_sqft"] = df["area_sqft"].fillna(df["area"].map(parse_area))
    else:
        df["area_sqft"] = df["area"].map(parse_area)

    keep = ["property_type", "price", "location", "city", "latitude",
            "longitude", "baths", "bedrooms", "area_sqft"]
    df = df[keep].copy()
    for col in ["price", "latitude", "longitude", "baths", "bedrooms", "area_sqft"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["property_type", "location", "city"]:
        df[col] = df[col].astype("string").str.strip()

    df = df.drop_duplicates()
    record("1. Remove duplicates", df)

    # 3. Missing values
    missing = df.isna().sum()
    print("Missing values per column:\n", missing[missing > 0].to_string(), "\n")
    df = df.dropna(subset=["price", "area_sqft", "city", "location", "property_type"])
    record("2. Drop rows missing price/area/city/location", df)

    for col in ["bedrooms", "baths"]:      # fill with median of same property type
        df[col] = df[col].fillna(df.groupby("property_type")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())
    df["latitude"] = df["latitude"].fillna(
        df.groupby(["city", "location"])["latitude"].transform("median"))
    df["longitude"] = df["longitude"].fillna(
        df.groupby(["city", "location"])["longitude"].transform("median"))
    df["latitude"] = df["latitude"].fillna(df.groupby("city")["latitude"].transform("median"))
    df["longitude"] = df["longitude"].fillna(df.groupby("city")["longitude"].transform("median"))
    df = df.dropna(subset=["latitude", "longitude"])
    record("3. Fill bedrooms/baths/lat/lon (median)", df)

    # 4. Impossible values
    df = df[(df["price"] > 0) & (df["area_sqft"].between(50, 100_000))
            & (df["bedrooms"] <= 15) & (df["baths"] <= 15)]
    record("4. Remove impossible values", df)

    # 5. Keep cities with enough data
    min_city = config.MIN_CITY_LISTINGS_RENT if purpose == "rent" else config.MIN_CITY_LISTINGS
    counts = df["city"].value_counts()
    df = df[df["city"].isin(counts[counts >= min_city].index)]
    record("5. Keep cities with enough listings", df, f"cities: {df['city'].nunique()}")

    # 6. Outliers: price per sq ft outside 1st-99th percentile of its own city
    df["price_per_sqft"] = df["price"] / df["area_sqft"]
    grp = df.groupby("city")["price_per_sqft"]
    low, high = grp.transform(lambda s: s.quantile(0.01)), grp.transform(lambda s: s.quantile(0.99))
    df = df[(df["price_per_sqft"] >= low) & (df["price_per_sqft"] <= high)]
    record("6. Remove price-per-sqft outliers", df)

    # 7. Feature engineering
    centers = df.groupby("city")[["latitude", "longitude"]].median()
    city_centers = {c: [float(r.latitude), float(r.longitude)] for c, r in centers.iterrows()}
    df = add_features(df, city_centers)
    df["log_price"] = np.log1p(df["price"])
    record("7. Feature engineering done", df, f"columns: {df.shape[1]}")
    return df.reset_index(drop=True), city_centers, pd.DataFrame(log)


def split_data(df: pd.DataFrame):
    """80/20 train-test split (stratified by city, fixed seed)."""
    return train_test_split(
        df[config.FEATURES], df[config.TARGET],
        test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
        stratify=df["city"])


def run(input_path=None, purpose="sale"):
    out = config.paths(purpose)
    raw = load_raw(input_path)
    clean, centers, log = clean_data(raw, purpose)
    config.DATA_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    clean.to_csv(out["clean_csv"], index=False)
    out["centers"].write_text(json.dumps(centers, indent=2))
    log.to_csv(out["cleaning_log"], index=False)
    X_train, X_test, _, _ = split_data(clean)
    print(f"\nSaved {out['clean_csv'].name}. Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="path to raw Zameen csv")
    parser.add_argument("--purpose", choices=["sale", "rent"], default="sale")
    a = parser.parse_args()
    run(a.input, a.purpose)
