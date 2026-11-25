"""PERSON A - Exploratory Data Analysis. Saves graphs to reports/figures and
writes the key findings to reports/eda_insights.txt.

Run:  python -m src.eda
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.preprocessing import load_raw

sns.set_theme(style="whitegrid", context="notebook")
FIG = config.FIGURES_DIR


def save(name):
    plt.tight_layout()
    plt.savefig(FIG / f"{name}.png", dpi=130)
    plt.close()
    print("saved", name)


def run():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.CLEAN_CSV)
    notes = []

    # 0. Missing values in the RAW data (before cleaning)
    try:
        raw = load_raw()
        miss = raw.isna().mean().mul(100).sort_values(ascending=False)
        miss = miss[miss > 0].head(12)
        if len(miss):
            plt.figure(figsize=(8, 4.5))
            sns.barplot(x=miss.values, y=miss.index, color="#d9534f")
            plt.xlabel("% missing"); plt.title("Missing values in raw data")
            save("00_missing_values")
    except Exception as exc:            # raw file not essential
        print("skip missing-value plot:", exc)

    # 1. Target distribution
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["price"] / 1e6, bins=60, ax=ax[0], color="#4c72b0")
    ax[0].set(title="Price (million PKR) - heavily skewed", xlabel="Price (million PKR)")
    sns.histplot(df["log_price"], bins=60, ax=ax[1], color="#55a868")
    ax[1].set(title="log(Price) - close to normal", xlabel="log(1 + price)")
    save("01_price_distribution")
    notes.append(f"Price skewness = {df['price'].skew():.2f}; after log transform = {df['log_price'].skew():.2f}. "
                 "So the models predict log(price).")

    # 2. City level
    city = df.groupby("city").agg(listings=("price", "size"), median_price=("price", "median"),
                                   median_psf=("price_per_sqft", "median")).sort_values("median_price", ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(x=city.index, y=city["listings"], ax=ax[0], color="#4c72b0")
    ax[0].set(title="Number of listings per city", ylabel="listings", xlabel="")
    sns.barplot(x=city.index, y=city["median_price"] / 1e6, ax=ax[1], color="#dd8452")
    ax[1].set(title="Median price per city (million PKR)", ylabel="million PKR", xlabel="")
    save("02_city_listings_and_price")
    top = city["median_psf"].idxmax(); low = city["median_psf"].idxmin()
    notes.append(f"Most expensive city per sq ft: {top} ({city.loc[top, 'median_psf']:,.0f} PKR/sqft); "
                 f"cheapest: {low} ({city.loc[low, 'median_psf']:,.0f} PKR/sqft).")

    # 3. Area vs price
    s = df.sample(min(8000, len(df)), random_state=1)
    plt.figure(figsize=(8, 5.5))
    sns.scatterplot(data=s, x="area_sqft", y="price", hue="property_type", alpha=0.45, s=18)
    plt.xscale("log"); plt.yscale("log")
    plt.title("Area vs price (log-log): almost a straight line")
    save("03_area_vs_price")
    corr_area = np.corrcoef(np.log(df["area_sqft"]), df["log_price"])[0, 1]
    notes.append(f"Correlation between log(area) and log(price) = {corr_area:.2f} (area is the strongest numeric driver).")

    # 4. Price per sqft by city
    plt.figure(figsize=(9, 4.5))
    sns.boxplot(data=df, x="city", y="price_per_sqft", order=city.index, showfliers=False, color="#8da0cb")
    plt.title("Price per sq ft by city"); plt.ylabel("PKR per sq ft"); plt.xlabel("")
    save("04_price_per_sqft_by_city")

    # 5. Correlation heatmap
    cols = ["price", "area_sqft", "bedrooms", "baths", "total_rooms", "dist_to_center_km", "latitude", "longitude"]
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
    plt.title("Correlation heatmap")
    save("05_correlation_heatmap")
    notes.append("Correlation with price: " + ", ".join(
        f"{c}={df[c].corr(df['price']):.2f}" for c in ["area_sqft", "bedrooms", "baths", "dist_to_center_km"]) + ".")

    # 6. Property type
    order = df.groupby("property_type")["price_per_sqft"].median().sort_values(ascending=False).index
    plt.figure(figsize=(9, 4.5))
    sns.boxplot(data=df, x="property_type", y="price_per_sqft", order=order, showfliers=False, color="#66c2a5")
    plt.title("Price per sq ft by property type"); plt.xlabel(""); plt.xticks(rotation=20)
    save("06_property_type")

    # 7. Bedrooms
    b = df[df["bedrooms"].between(1, 8)].groupby("bedrooms")["price"].median() / 1e6
    plt.figure(figsize=(7, 4))
    sns.barplot(x=b.index.astype(int), y=b.values, color="#e78ac3")
    plt.title("Median price by number of bedrooms"); plt.ylabel("million PKR"); plt.xlabel("bedrooms")
    save("07_bedrooms_vs_price")

    # 8. Top localities
    loc = df.groupby("locality").agg(n=("price", "size"), psf=("price_per_sqft", "median"))
    loc = loc[loc["n"] >= 20].sort_values("psf", ascending=False).head(15)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=loc["psf"], y=loc.index, color="#fc8d62")
    plt.title("Top 15 localities by median price per sq ft (min 20 listings)"); plt.xlabel("PKR per sq ft"); plt.ylabel("")
    save("08_top_localities")
    notes.append(f"Most expensive locality: {loc.index[0]} ({loc['psf'].iloc[0]:,.0f} PKR/sqft). "
                 "Location explains a lot of price, which is why locality is target-encoded.")

    # 9. Distance to centre
    plt.figure(figsize=(7, 4.5))
    sns.scatterplot(data=s, x="dist_to_center_km", y="price_per_sqft", hue="city", alpha=0.4, s=14)
    plt.yscale("log"); plt.title("Distance from city centre vs price per sq ft")
    save("09_distance_vs_price")

    notes.append(f"Dataset after cleaning: {len(df):,} listings, {df['city'].nunique()} cities, "
                 f"{df['locality'].nunique()} localities, {df['property_type'].nunique()} property types.")
    (config.REPORTS_DIR / "eda_insights.txt").write_text("\n".join(f"- {n}" for n in notes))
    print("\nKEY INSIGHTS\n" + "\n".join(f"- {n}" for n in notes))


if __name__ == "__main__":
    run()
