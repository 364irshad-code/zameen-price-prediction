"""PERSON B - Loads the saved model and turns user input into a price prediction.
Kept separate from the web app so it can be tested without Streamlit."""
from __future__ import annotations
import json

import joblib
import numpy as np
import pandas as pd

from src import config
from src.preprocessing import add_features


def artifacts_exist(purpose: str = "sale") -> bool:
    return config.paths(purpose)["model"].exists()


def load_artifacts(purpose: str = "sale"):
    """purpose = 'sale' (buy price) or 'rent' (monthly rent)."""
    p = config.paths(purpose)
    model = joblib.load(p["model"])
    meta = json.loads(p["meta"].read_text())
    lookup = pd.read_csv(p["lookup"])
    return model, meta, lookup


def predict_price(model, meta, lookup, city, location, property_type,
                  area_value, unit, bedrooms, baths) -> dict:
    """Return the predicted price (PKR) with an 80% likely range."""
    row = lookup[(lookup["city"] == city) & (lookup["location"] == location)].iloc[0]
    area_sqft = float(area_value) * config.UNIT_TO_SQFT[unit]
    raw = pd.DataFrame([{
        "property_type": property_type, "city": city, "location": location,
        "latitude": row["latitude"], "longitude": row["longitude"],
        "bedrooms": float(bedrooms), "baths": float(baths), "area_sqft": area_sqft,
    }])
    features = add_features(raw, meta["city_centers"])[config.FEATURES]
    log_pred = float(model.predict(features)[0])
    price = float(np.expm1(log_pred))
    return {
        "price": price,
        "low": float(np.expm1(log_pred + meta["residual_q10"])),
        "high": float(np.expm1(log_pred + meta["residual_q90"])),
        "area_sqft": area_sqft,
        "price_per_sqft": price / area_sqft,
        "locality_median_psf": float(row["median_psf"]),
    }


def format_pkr(value: float) -> str:
    """12,500,000 -> 'PKR 1.25 Crore' ; 8,500,000 -> 'PKR 85.00 Lakh'."""
    if value >= 1e7:
        return f"PKR {value / 1e7:.2f} Crore"
    if value >= 1e5:
        return f"PKR {value / 1e5:.2f} Lakh"
    return f"PKR {value:,.0f}"


def format_rent(value: float) -> str:
    """85000 -> 'PKR 85,000 / month' ; 150000 -> 'PKR 1.50 Lakh / month'."""
    if value >= 1e5:
        return f"PKR {value / 1e5:.2f} Lakh / month"
    return f"PKR {value:,.0f} / month"


def room_defaults(clean_df=None) -> dict:
    """Typical size of a 'Room' listing, so the user is not asked for area/bedrooms/baths."""
    fallback = {"area_sqft": 150.0, "bedrooms": 1.0, "baths": 1.0}
    if clean_df is None or "property_type" not in clean_df.columns:
        return fallback
    rooms = clean_df[clean_df["property_type"].astype(str).str.lower() == "room"]
    if len(rooms) < 20:
        return fallback
    return {
        "area_sqft": float(rooms["area_sqft"].median()),
        "bedrooms": float(max(rooms["bedrooms"].median(), 1)),
        "baths": float(max(rooms["baths"].median(), 1)),
    }
