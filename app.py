"""Web app (Streamlit).  Run:  streamlit run app.py

Mode switch: Buy (sale price)  /  Rent (monthly rent)
Tab 1 Predict           - user enters property details, gets price + likely range
Tab 2 Market insights   - interactive charts from the cleaned data
Tab 3 Model comparison  - metrics table, best-model explanation
"""
import pandas as pd
import streamlit as st

from src import config
from src.predict import (artifacts_exist, format_pkr, format_rent,
                         load_artifacts, predict_price, room_defaults)
from src.ui import page_css

st.set_page_config(page_title="Pakistan House Price Predictor", page_icon="🏠", layout="wide")
st.markdown(page_css(), unsafe_allow_html=True)


@st.cache_resource
def get_artifacts(purpose):
    return load_artifacts(purpose)


@st.cache_data
def get_clean_data(purpose):
    return pd.read_csv(config.paths(purpose)["clean_csv"])


st.title("🏠 Pakistan House Price & Rent Predictor")
st.caption("Machine-learning models trained on Zameen.com property listings.")

mode = st.radio("What do you want to know?", ["Buy price (sale)", "Monthly rent"], horizontal=True)
purpose = "rent" if mode.startswith("Monthly") else "sale"
fmt = format_rent if purpose == "rent" else format_pkr

if not artifacts_exist(purpose):
    if purpose == "rent":
        st.error("Rent model not found. Run these two commands, then refresh:\n\n"
                 "python -m src.preprocessing --purpose rent\n\n"
                 "python -m src.train_models --purpose rent")
    else:
        st.error("Model not found. Run:  python -m src.preprocessing  ->  python -m src.train_models")
    st.stop()

model, meta, lookup = get_artifacts(purpose)
tab_predict, tab_market, tab_models = st.tabs(["💰 Predict", "📊 Market insights", "🧠 Model comparison"])

# ------------------------------------------------------------ Tab 1: predict
with tab_predict:
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("Property details")
        city = st.selectbox("City", sorted(lookup["city"].unique()), key=f"city_{purpose}")
        locations = sorted(lookup.loc[lookup["city"] == city, "location"].unique())
        location = st.selectbox("Location / society", locations, key=f"loc_{purpose}")
        property_type = st.selectbox("Property type", meta["property_types"], key=f"type_{purpose}")
        if property_type.strip().lower() == "room":
            # A room has no meaningful area/bedrooms, so use typical values; the user only picks bathrooms
            clean_path = config.paths(purpose)["clean_csv"]
            ref = room_defaults(get_clean_data(purpose) if clean_path.exists() else None)
            area, unit, bedrooms = ref["area_sqft"], "Sq Ft", ref["bedrooms"]
            baths = st.number_input("Bathrooms", 0, 5, int(ref["baths"]), key=f"roombath_{purpose}")
            st.caption(f"Room selected: a typical room size (about {area:,.0f} sq ft) is used automatically.")
        else:
            c1, c2 = st.columns(2)
            unit = c2.selectbox("Unit", ["Marla", "Kanal", "Sq Ft"], key=f"unit_{purpose}")
            default_area = {"Marla": 5.0, "Kanal": 1.0, "Sq Ft": 1200.0}[unit]
            area = c1.number_input("Area", min_value=0.5, value=default_area, step=0.5, key=f"area_{purpose}")
            c3, c4 = st.columns(2)
            bedrooms = c3.number_input("Bedrooms", 0, 15, 3, key=f"bed_{purpose}")
            baths = c4.number_input("Bathrooms", 0, 15, 3, key=f"bath_{purpose}")
        go = st.button("Predict", type="primary", use_container_width=True, key=f"go_{purpose}")

    with right:
        st.subheader("Result")
        if go:
            res = predict_price(model, meta, lookup, city, location, property_type,
                                area, unit, bedrooms, baths)
            label = "Estimated monthly rent" if purpose == "rent" else "Estimated price"
            st.metric(label, fmt(res["price"]))
            st.write(f"**Likely range (80% confidence):** {fmt(res['low'])}  to  {fmt(res['high'])}")
            unit_txt = "/sqft/month" if purpose == "rent" else "/sqft"
            m1, m2 = st.columns(2)
            m1.metric("Predicted per sq ft", f"PKR {res['price_per_sqft']:,.1f}" if purpose == "rent"
                      else f"PKR {res['price_per_sqft']:,.0f}")
            m2.metric(f"Typical in {location}", f"PKR {res['locality_median_psf']:,.1f}{unit_txt}" if purpose == "rent"
                      else f"PKR {res['locality_median_psf']:,.0f}{unit_txt}")
            st.caption(f"Area used: {res['area_sqft']:,.0f} sq ft  |  Exact: PKR {res['price']:,.0f}")
            st.info("This is a statistical estimate from listing (asking) prices, not a valuation.")
        else:
            st.write("Fill in the details and press **Predict**.")

# ------------------------------------------------------- Tab 2: market insights
with tab_market:
    clean_path = config.paths(purpose)["clean_csv"]
    if not clean_path.exists():
        st.warning("Cleaned data not found. Run python -m src.preprocessing first.")
    else:
        df = get_clean_data(purpose)
        word = "rent per sq ft (PKR/month)" if purpose == "rent" else "price per sq ft (PKR)"
        st.subheader(f"Median {word} by city")
        st.bar_chart(df.groupby("city")["price_per_sqft"].median().sort_values(ascending=False))
        st.subheader("Explore one city")
        pick = st.selectbox("City", sorted(df["city"].unique()), key=f"explore_{purpose}")
        sub = df[df["city"] == pick]
        a, b = st.columns(2)
        a.markdown(f"**Top localities (median {word})**")
        a.bar_chart(sub.groupby("location")["price_per_sqft"].median().sort_values(ascending=False).head(12))
        b.markdown("**Median " + ("monthly rent (thousand PKR)" if purpose == "rent" else "price (million PKR)") + " by property type**")
        div = 1e3 if purpose == "rent" else 1e6
        b.bar_chart((sub.groupby("property_type")["price"].median() / div).sort_values(ascending=False))
        if purpose == "sale":
            st.subheader("EDA graphs")
            figs = sorted(config.FIGURES_DIR.glob("0*.png"))
            for i in range(0, len(figs), 2):
                cols = st.columns(2)
                for col, f in zip(cols, figs[i:i + 2]):
                    col.image(str(f))
            insights = config.REPORTS_DIR / "eda_insights.txt"
            if insights.exists():
                st.markdown("**Key insights**\n\n" + insights.read_text())

# --------------------------------------------------------- Tab 3: models
with tab_models:
    tag = config.paths(purpose)["tag"]
    st.subheader(f"Best model ({'rent' if purpose == 'rent' else 'sale'}): {meta['best_model']}")
    comp = config.REPORTS_DIR / f"{tag}model_comparison.csv"
    if comp.exists():
        table = pd.read_csv(comp).set_index("Model")
        st.dataframe(table.style.format("{:,.3f}").highlight_min(
            subset=["RMSE (log)", "MedAPE (%)", "MAE (PKR)"], color="#c7f0d2")
            .highlight_max(subset=["R2 (log)", "R2 (PKR)"], color="#c7f0d2"), use_container_width=True)
    m = meta["metrics"]
    k1, k2, k3 = st.columns(3)
    k1.metric("R² (log)", f"{m['R2 (log)']:.3f}")
    k2.metric("Median error", f"{m['MedAPE (%)']:.1f}%")
    k3.metric("MAE", fmt(m["MAE (PKR)"]))
    for name in ["10_model_comparison", "11_best_model_diagnostics", "12_feature_importance"]:
        f = config.FIGURES_DIR / f"{tag}{name}.png"
        if f.exists():
            st.image(str(f))
