# 🌍 Habitability Predictor

A Machine Learning–based framework that estimates the **habitability potential of exoplanets** by combining astrophysical features with an enhanced **SEPHI proxy** (Statistical-likelihood-based Earth Similarity Index).  
This model integrates **physical modeling** and **data-driven regression** to provide unbiased **habitability scores** and classifications of exoplanets into interpretable planetary categories.

---

## 🚀 Overview

The **Habitability Predictor** evaluates whether an exoplanet could potentially sustain Earth-like life.  
It does so by:
1. Computing an **Enhanced Earth Similarity Index (EESI)** as a SEPHI proxy.
2. Performing **feature engineering** from astrophysical parameters.
3. Using **Random Forest models** for both **classification** and **regression**:
   - **Classification:** Habitable vs Non-Habitable.
   - **Regression:** Continuous habitability score (SEPHI proxy).

---

## 🧠 Features

### 🧩 Physics-Inspired Engineering
- **Interior Similarity:** Radius–Mass ratio (telluricity)
- **Surface Temperature Similarity:** Liquid water potential via equilibrium temperature
- **Stellar Flux Similarity:** Energy input balance
- **Derived Features:**
  - Planetary **density**
  - Stellar **peak wavelength** (via Wien’s law)
  - **Temperature variation** with orbital extremes (ΔT)
  - **Orbital orientation vectors** (x, y)
  - **TLP Metric:** Combined thermodynamic and flux-based heuristic

### 🔬 Machine Learning Models
| Task | Model | Metric |
|------|--------|---------|
| Habitability Classification | RandomForestClassifier | Accuracy |
| SEPHI Regression (Unbiased Habitability Score) | RandomForestRegressor | MAE |

---

## 🗂️ Dataset

Uses the **NASA Exoplanet Archive dataset** (`exoplanets_complete.csv`).  
Columns are standardized automatically via a smart renaming pipeline that supports aliases like `pl_rade`, `pl_bmasse`, etc.

Example key columns:
- `P_RADIUS`, `P_MASS`, `P_TEMP_EQUIL`
- `S_TEMPERATURE`, `S_RADIUS`, `S_MASS`
- `P_FLUX`, `P_SEMI_MAJOR_AXIS`, `P_OMEGA`

---

## ⚙️ How It Works

1. **Load & Clean Data**
   - Standardizes exoplanet data fields.
   - Computes derived physical features.

2. **Calculate SEPHI Proxy**
   - Combines Interior, Temperature, and Flux similarities:
     ```python
     SEPHI = (I_interior * I_surface_temp * I_flux) ** (1/3)
     ```

3. **Train Models**
   - Classification on `habitability_label`.
   - Regression on computed `SEPHI_CALC`.

4. **Predict Habitability**
   - Generates numerical score and interpretable category.

---

## 🧾 Example Output

✅ Columns standardized
✅ SEPHI Proxy calculated for training data

🔹 Classification Results
Accuracy: 0.94

🔹 Regression Results (Target: SEPHI_CALC / Habitability Score)
MAE: 0.0382

--- 🌍 Habitability Predictor Results ---

🌟 Earth Analog Habitability Score: 0.9823
🌟 Kepler-186f Habitability Score: 0.7432

--- Kepler-186f Classification ---
Habitability: Potentially Habitable
🌱 Vegetation type: Mixed Vegetation
Planet Type: Serengetian


---

## 🪐 Interpretation Scale

| Score Range | Habitability | Vegetation Type | Planet Type |
|--------------|---------------|-----------------|--------------|
| 0.8 – 1.0 | **Highly Habitable** | Dense Vegetation | Amazonian |
| 0.6 – 0.8 | **Potentially Habitable** | Mixed Vegetation | Serengetian |
| 0.4 – 0.6 | **Weakly Habitable** | Shrublands | Mediterranean |
| 0.0 – 0.4 | **Non-Habitable** | Sparse Vegetation | Saharan |
| ≤ 0.0 | **Dead Planet** | None | Dead |

---

## 🧩 Tech Stack

- **Language:** Python 3.x  
- **Libraries:**
  - `pandas`, `numpy`
  - `scikit-learn`
- **Models:**
  - `RandomForestClassifier`
  - `RandomForestRegressor`

---

## 📈 Future Improvements

- Integrate deep learning (e.g., PyTorch-based regression).
- Use Kepler/TESS public datasets for live inference.
- Add visualization dashboard (planetary comparatives).
- Implement probabilistic calibration for SEPHI confidence.

---

## 👨‍🚀 Author

**Ayush Pandey**  
Machine Learning & Astrophysics Enthusiast  
🔭 Focus: AI for Space Science and Planetary Research  
📫 [LinkedIn / GitHub links can be added here]

---

## 🪙 License

This project is released under the **MIT License**.  
Feel free to use, modify, and cite for academic or research purposes.

---

