"""Shared settings for the whole project (paths, feature lists, constants)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Put the real Kaggle file here. If it is missing, the small sample file is used.
RAW_CSV = DATA_DIR / "zameen-updated.csv"
SAMPLE_CSV = DATA_DIR / "sample_zameen.csv"
CLEAN_CSV = DATA_DIR / "zameen_clean.csv"
CITY_CENTERS_JSON = DATA_DIR / "city_centers.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_CITY_LISTINGS = 300          # cities with fewer listings are dropped (sale)
MIN_CITY_LISTINGS_RENT = 100     # rent has fewer listings, so a lower limit

# Area units -> square feet
SQFT_PER_MARLA = 272.25
SQFT_PER_KANAL = 5445.0
UNIT_TO_SQFT = {"Marla": SQFT_PER_MARLA, "Kanal": SQFT_PER_KANAL, "Sq Ft": 1.0}

# Model input features
NUM_FEATURES = [
    "log_area", "bedrooms", "baths", "latitude", "longitude",
    "dist_to_center_km", "bath_per_bed", "total_rooms",
]
CAT_FEATURES = ["property_type", "city"]
HIGH_CARD_FEATURES = ["locality"]          # target-encoded (hundreds of values)
FEATURES = NUM_FEATURES + CAT_FEATURES + HIGH_CARD_FEATURES
TARGET = "log_price"                       # model predicts log(1 + price in PKR)


def paths(purpose: str = "sale") -> dict:
    """All file paths for the 'sale' model or the 'rent' model.
    Sale keeps the original file names, rent files start with 'rent_'."""
    rent = purpose == "rent"
    tag = "rent_" if rent else ""
    return {
        "clean_csv": DATA_DIR / ("zameen_clean_rent.csv" if rent else "zameen_clean.csv"),
        "centers": DATA_DIR / ("city_centers_rent.json" if rent else "city_centers.json"),
        "cleaning_log": REPORTS_DIR / f"{tag}cleaning_log.csv",
        "model": MODELS_DIR / f"{tag}best_model.joblib",
        "meta": MODELS_DIR / f"{tag}meta.json",
        "lookup": MODELS_DIR / f"{tag}locality_lookup.csv",
        "tag": tag,
    }
