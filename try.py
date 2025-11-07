import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

# Earth reference values (in Earth Units)
EARTH_MASS_REF = 1.0
EARTH_RADIUS_REF = 1.0
EARTH_TEMP_REF = 288.0 # Earth's mean surface temperature in Kelvin
EARTH_FLUX_REF = 1.0 # Earth's solar flux (insolation) in Earth units

def calculate_sephi_for_values(P_RADIUS, P_MASS, P_TEMP_EQUIL, P_FLUX):
    """
    Calculates an Enhanced Earth Similarity Index (EESI) as a SEPHI proxy 
    using the geometric mean of Interior, Surface Temperature, and Flux similarities.
    This will be the unbiased target (Habitability Score) for the ML model.
    """
    if any(pd.isna(v) for v in [P_RADIUS, P_MASS, P_TEMP_EQUIL, P_FLUX]):
        return 0.0

    # 1. Interior Similarity (Telluricity)
    # Exponent 0.57 is a standard ESI weight for Radius/Mass.
    I_radius = np.power(
        1 - np.abs(P_RADIUS - EARTH_RADIUS_REF) / (P_RADIUS + EARTH_RADIUS_REF),
        0.57
    )
    I_mass = np.power(
        1 - np.abs(P_MASS - EARTH_MASS_REF) / (P_MASS + EARTH_MASS_REF),
        0.57
    )
    I_interior = np.sqrt(I_radius * I_mass)
    
    # 2. Surface Temperature Similarity (Liquid Water Potential)
    # Exponent 5.58 is a high ESI weight for surface temperature.
    I_surface_temp = np.power(
        1 - np.abs(P_TEMP_EQUIL - EARTH_TEMP_REF) / (P_TEMP_EQUIL + EARTH_TEMP_REF),
        5.58
    )

    # 3. Stellar Flux Similarity (Energy Input)
    I_flux = np.power(
        1 - np.abs(P_FLUX - EARTH_FLUX_REF) / (P_FLUX + EARTH_FLUX_REF),
        1.0
    )
    
    # SEPHI Proxy (Geometric Mean)
    sephi_proxy = np.power(I_interior * I_surface_temp * I_flux, 1/3)
    return sephi_proxy


# -----------------------------
# 1. Load CSV
# -----------------------------
file = r"C:\Users\Acer\OneDrive\Desktop\SIHnew\exoplanets_complete.csv"
f_data = pd.read_csv(file)

# -----------------------------
# 2. Standardize column names
# -----------------------------
col_map = {
    "P_ORBPER": ["P_ORBPER", "P_PERIOD", "pl_orbper"],
    "P_RADIUS": ["P_RADIUS", "pl_rade", "pl_rad"],
    "P_MASS": ["P_MASS", "pl_bmasse", "pl_mass", "pl_masse", "pl_msinie"],
    "P_TEMP_EQUIL": ["P_TEMP_EQUIL", "pl_eqt"],
    "S_TEMPERATURE": ["S_TEMPERATURE", "st_teff"],
    "S_RADIUS": ["S_RADIUS", "st_rad"],
    "S_MASS": ["S_MASS", "st_mass"],
    "P_FLUX": ["P_FLUX", "pl_insol"],
    "P_SEMI_MAJOR_AXIS": ["P_SEMI_MAJOR_AXIS", "pl_orbsmax"],
    "P_PERIASTRON": ["P_PERIASTRON", "pl_periastron"],
    "P_APASTRON": ["P_APASTRON", "pl_apastron"],
    "P_OMEGA": ["P_OMEGA", "pl_omega"]
}

for standard, aliases in col_map.items():
    for alias in aliases:
        if alias in f_data.columns:
            f_data.rename(columns={alias: standard}, inplace=True)
            break
    if standard not in f_data.columns:
        f_data[standard] = np.nan

print("✅ Columns standardized")

# -----------------------------
# 3. Ensure Targets (Keeping original HZ_SCORE for classification label only)
# -----------------------------
if "habitability_label" not in f_data.columns:
    if "HZ_SCORE" in f_data.columns:
        f_data["habitability_label"] = np.where(f_data["HZ_SCORE"] > 0.5, "Habitable", "Non-Habitable")
    else:
        f_data["habitability_label"] = "Unknown"

if "habitability_score" not in f_data.columns:
    if "HZ_SCORE" in f_data.columns:
        f_data["habitability_score"] = f_data["HZ_SCORE"]
    else:
        f_data["habitability_score"] = np.nan

# -----------------------------
# 4.Feature Engineering kari using  SEPHI_calc as a new targett
# -----------------------------
EARTH_MASS = 5.972e24 # kg
EARTH_RADIUS = 6.371e6 # m

def density(row):
    if pd.notna(row["P_MASS"]) and pd.notna(row["P_RADIUS"]) and row["P_RADIUS"] > 0:
        mass = row["P_MASS"] * EARTH_MASS
        radius = row["P_RADIUS"] * EARTH_RADIUS
        volume = (4/3) * np.pi * (radius**3)
        return (mass / volume) / 1000 # g/cm³
    return np.nan

f_data["density"] = f_data.apply(density, axis=1)

#Peak wavelength(Wien's law, nm)
f_data["peak_wavelength_nm"] = np.where(
    f_data["S_TEMPERATURE"] > 0, 2.9e6 / f_data["S_TEMPERATURE"], np.nan
)

#ΔT with closest and farthest distance 
periastron = f_data["P_PERIASTRON"].fillna(f_data["P_SEMI_MAJOR_AXIS"])
apastron = f_data["P_APASTRON"].fillna(f_data["P_SEMI_MAJOR_AXIS"])

f_data["T_min"] = f_data["P_TEMP_EQUIL"] * np.sqrt(f_data["P_SEMI_MAJOR_AXIS"] / apastron)
f_data["T_max"] = f_data["P_TEMP_EQUIL"] * np.sqrt(f_data["P_SEMI_MAJOR_AXIS"] / periastron)
f_data["delta_T"] = f_data["T_max"] - f_data["T_min"]

#Orientation x and y 
omega = f_data["P_OMEGA"].fillna(0)
f_data["orientation_x"] = f_data["P_SEMI_MAJOR_AXIS"] * np.cos(np.radians(omega))
f_data["orientation_y"] = f_data["P_SEMI_MAJOR_AXIS"] * np.sin(np.radians(omega))

# Example TLP metric
f_data["TLP"] = (
    (1 / (f_data["delta_T"].abs() + 1)) +
    (1 / (abs(f_data["density"] - 5.5) + 1)) +
    (f_data["P_FLUX"] / 1.0)
)

# --- NEW: Calculate SEPHI Proxy for all data. This is our new, unbiased regression target. ---
f_data["SEPHI_CALC"] = f_data.apply(
    lambda row: calculate_sephi_for_values(row["P_RADIUS"], row["P_MASS"], row["P_TEMP_EQUIL"], row["P_FLUX"]),
    axis=1
)
print("✅ SEPHI Proxy calculated for training data")

# -----------------------------
# 5.featuresss
# -----------------------------
f_features = [
    "P_ORBPER", "P_RADIUS", "P_MASS", "P_TEMP_EQUIL",
    "S_TEMPERATURE", "S_RADIUS", "S_MASS", "P_FLUX", "P_SEMI_MAJOR_AXIS",
    "density", "peak_wavelength_nm", "delta_T", "orientation_x", "orientation_y", "TLP"
]

X = f_data[f_features].copy()
X = X.fillna(X.median())

# -----------------------------
# 6. Classification
# -----------------------------
y_class = f_data["habitability_label"]

train_Xc, val_Xc, train_yc, val_yc = train_test_split(X, y_class, random_state=0)

clf_model = RandomForestClassifier(n_estimators=200, random_state=0, class_weight="balanced")
clf_model.fit(train_Xc, train_yc)

val_preds_class = clf_model.predict(val_Xc)
print("\n🔹 Classification Results")
print("Accuracy:", accuracy_score(val_yc, val_preds_class))

# -----------------------------
# 7. Regression tuning/training on sephi_calc for unbiased habitability scoreee
# -----------------------------
# SEPHI_CALC ko use kia as a primary target
y_reg = f_data["SEPHI_CALC"]
mask = y_reg > 0.0 # Only use data where SEPHI could be calculated (i.e., not NaN)
X_reg = X.loc[mask]
y_reg = y_reg.loc[mask]

# The reg_model ko use kia to predict 
reg_model = None
if len(y_reg) > 0:
    train_Xr, val_Xr, train_yr, val_yr = train_test_split(X_reg, y_reg, random_state=0)

    reg_model = RandomForestRegressor(n_estimators=200, random_state=0)
    reg_model.fit(train_Xr, train_yr)

    val_preds_reg = reg_model.predict(val_Xr)
    print("\n🔹 Regression Results (Target: SEPHI_CALC / Habitability Score)")
    print("MAE:", mean_absolute_error(val_yr, val_preds_reg))
else:
    print("\n⚠️ No valid SEPHI regression target values found.")

# -----------------------------
# 8.Prediction
# -----------------------------
new_planet = pd.DataFrame({
    "P_ORBPER": [365], "P_RADIUS": [1.0], "P_MASS": [1.0], "P_TEMP_EQUIL": [288],
    "S_TEMPERATURE": [5772], "S_RADIUS": [1.0], "S_MASS": [1.0], "P_FLUX": [1.0],
    "P_SEMI_MAJOR_AXIS": [1.0], "density": [5.5], "peak_wavelength_nm": [500],
    "delta_T": [0], "orientation_x": [1.0], "orientation_y": [0.0], "TLP": [3.0]
})
kepler_186f = pd.DataFrame({
    "P_ORBPER": [130.0], "P_RADIUS": [1.11], "P_MASS": [1.4], "P_TEMP_EQUIL": [227],
    "S_TEMPERATURE": [3788], "S_RADIUS": [0.47], "S_MASS": [0.5], "P_FLUX": [0.32],
    "P_SEMI_MAJOR_AXIS": [0.35], "density": [5.1], "peak_wavelength_nm": [766],
    "delta_T": [0], "orientation_x": [0.35], "orientation_y": [0.0], "TLP": [1.38]
})

print("\n--- 🌍 Habitability Predictor Results ---")

if reg_model is not None:
    #1. Earth ki Prediction
    predicted_earth_score = reg_model.predict(new_planet)[0]
    print(f"\n🌟 Earth Analog Habitability Score: {predicted_earth_score:.4f}")
    
    #2. Kepler-186f Prediction
    predicted_k186f_score = reg_model.predict(kepler_186f)[0]
    print(f"🌟 Kepler-186f Habitability Score: {predicted_k186f_score:.4f}")

    print("\n--- Kepler-186f Classification ---")
    score = predicted_k186f_score
    
    #Habitability classification 
    if 0.8 <= score <= 1.0:
        print("Habitability: Highly Habitable")
        print("🌱 Vegetation type: Dense Vegetation \nPlanet Type: Amazonian")
    elif 0.6 <= score < 0.8:
        print("Habitability: Potentially Habitable")
        print("🌱 Vegetation type: Mixed Vegetation \nPlanet Type: Serengetian")
    elif 0.4 <= score < 0.6:
        print("Habitability: Weakly Habitable")
        print("🌱 Vegetation type: Shrublands \nPlanet Type: Mediterranean")
    elif 0.0 < score < 0.4:
        print("Habitability: Non-Habitable")
        print("🌱 Vegetation type: Sparse Vegetation \nPlanet Type: Saharan")
    elif score <= 0:
        print("Habitability: Non-Habitable")
        print("🌱 Vegetation type: None \nPlanet Type: Dead")
