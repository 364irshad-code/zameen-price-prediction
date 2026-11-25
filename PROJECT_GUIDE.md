# PROJECT GUIDE (Roman Urdu) - Pakistan House Price Prediction

## 1. Problem Statement
Pakistan mein ghar/plot ki qeemat aam tor par andaze ya dalal ki marzi se tay hoti hai, is liye khareedar
aur farokht karne wale dono ko sahi price ka pata nahi hota. Hum Zameen.com ke listings se ek ML model
banate hain jo city, location, property type, area, bedrooms aur bathrooms dekh kar **qeemat predict** karta hai
(Regression problem). Output: price (PKR) aur 80% likely range.

## 2. Dataset
- Source: Kaggle par "Zameen.com Pakistan property dataset" (`zameen-updated.csv`), Zameen.com ki listings. Link report mein lagana.
- Input features: property_type, city, location, latitude, longitude, area (marla/kanal/sqft), bedrooms, baths.
- Engineered features: log_area, total_rooms, bath_per_bed, dist_to_center_km (haversine), locality.
- Target: price (PKR). Model `log(1+price)` predict karta hai.

## 3. Preprocessing (kya aur kyun)
| Step | Kya kiya | Wajah |
|---|---|---|
| Duplicates | property_id aur poori row duplicates hataye | Same listing dobara na ginay |
| Area | "5 Marla" jaisa text sq ft mein badla (1 Marla = 272.25 sqft, 1 Kanal = 5445) | Sab area ek unit mein |
| Missing | price/area/city missing wali rows drop; bedrooms/baths ko same property type ki median se; lat/lon ko locality/city median se | Data zyada zaya na ho |
| Impossible values | price<=0, area bohat chhota/bara, 15+ bedrooms hataye | Data entry ghaltiyan |
| Outliers | Har city mein price-per-sqft ka 1%-99% se bahar hataya | 1 crore ki jagah 1 arab jaisi ghalat listings |
| Encoding | property_type, city -> One-Hot; locality (sainkron values) -> Target Encoding | One-Hot se hazaron columns ban jate |
| Scaling | StandardScaler (numeric) | Ridge aur Neural Network ko zaroori |
| Log transform | price ka log | Price bohat skewed hai (skew ~8 -> ~0) |
| Split | 80% train / 20% test, city ke hisab se stratified, seed 42 | Fair testing |

Zaroori: encoding aur scaling **Pipeline ke andar** hai, yaani sirf train data par fit hoti hai. Isay "data leakage se bachna" kehte hain.

## 4. Models (4)
1. **Ridge Regression** - baseline. Seedhi line + penalty (L2) jo coefficients ko bara hone se rokti hai.
2. **Random Forest** - bohat saare decision trees ka average. Non-linear patterns pakarta hai, overfit kam.
3. **XGBoost** (agar install na ho to sklearn ka HistGradientBoosting) - trees ek ke baad ek, har naya tree pichli ghalti theek karta hai.
4. **Neural Network (MLP)** - 2-3 hidden layers, early stopping ke saath.

Har model par RandomizedSearchCV (3-fold cross-validation) se hyperparameters tune hote hain, phir poore train data par refit.

**Metrics:** MAE, RMSE (PKR mein), R2, MedAPE (median % error). Best model ka intikhab test RMSE (log) se hota hai
kyunke is se sasti aur mehngi dono properties barabar tolti hain. R2 (PKR) mehngi properties se dominate hota hai, isliye usay akela na dekho.

## 5. Best model kyun? (viva ka jawab - results aane ke baad apne numbers se bharna)
Aam tor par real Zameen data par boosting (XGBoost) jeetta hai kyunke:
- Price aur area/location ka rishta non-linear hai (area double hone par price double nahi hota, location ka asar multiply hota hai); Ridge sirf linear pakar sakta hai.
- Boosting har step par pichli ghalti theek karta hai (bias kam), Random Forest sirf variance kam karta hai.
- Mixed data (categories + numbers) par trees bina zyada tayari ke achhe chalte hain; Neural Network ko zyada data aur tuning chahiye.
Yeh sirf umeed hai: jo model aap ke **test results** mein jeete, wahi best likho aur wajah samjhao.

## 6. Advanced cheezen jo is project mein hain (marks ke liye)
- Geo feature: shehar ke markaz se distance (haversine)
- Target Encoding with leakage-safe pipeline
- Log-target + outlier handling per city
- Hyperparameter tuning with cross-validation
- Permutation feature importance (kaun si input sab se zyada matter karti hai)
- Error analysis by city
- Prediction **range** (residual quantiles se), sirf ek number nahi
- Web app mein 3 tabs: Predict, Market insights, Model comparison

## 7. Kaam ki taqseem (2 log)
**Person A - Data & Insights**
- `src/preprocessing.py`, `src/eda.py`, app ka "Market insights" tab
- Report: Section 1 Problem, 2 Dataset, 3 Preprocessing, 4 EDA (graphs `reports/figures/00-09`, `cleaning_log.csv`, `eda_insights.txt`)

**Person B - Models & App**
- `src/train_models.py`, `src/predict.py`, `app.py`
- Report: Section 5 Models, 6 Comparison + best model ki wajah, 7 Web app (screenshots), 8 Submission (GitHub)
- Results: `model_comparison.csv`, graphs `10-12`

Viva se pehle dono ek doosre ka hissa zaroor samajh lein.

## 8. Report ka dhancha
1 Introduction & problem | 2 Dataset (source link, columns, size) | 3 Preprocessing (cleaning_log table) | 4 EDA (graphs + insights) |
5 Models (har ka short explanation + best params) | 6 Results table + best model discussion + limitations | 7 Web app (screenshots) | 8 Conclusion & future work | References + GitHub link

## 9. Limitations (report mein zaroor likho)
- Dataset **asking prices** hai, asal sale price nahi.
- Data purana ho sakta hai (qeemat badalti rehti hain).
- Naye ya kam listings wali locality par error zyada hoga.
- Sirf wahi shehar jo dataset mein hain.

## 10. Demo flow
1. App kholo -> city/location/type/area daalo -> Predict -> price + range dikhao.
2. Market insights tab: city comparison.
3. Model comparison tab: table aur "best model kyun".

## 11. Viva ke mumkina sawal
1. Log transform kyun? 2. Target Encoding kya hai aur One-Hot kyun nahi? 3. Data leakage kya hai, aap ne kaise roka?
4. R2 aur MedAPE mein farq? 5. Random Forest aur Boosting mein farq? 6. Ridge mein alpha kya karta hai?
7. Outliers kaise handle kiye? 8. Overfitting kaise check ki (CV vs test)? 9. Prediction range kaise banai?
10. Model ki limitations kya hain?
