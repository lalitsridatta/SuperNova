# Measuring Cosmological Parameters Using Type Ia Supernovae

## 📌 Project Overview

Type Ia Supernovae are powerful stellar explosions that act as **standard candles** in cosmology. 
They enable accurate measurement of cosmic distances and provide strong evidence for the expansion of the universe.

In this project, real-world **Pantheon+SH0ES supernova data** is used to analyze the **redshift–distance relationship**, estimate the **Hubble Constant (H₀)**, and validate modern cosmological models such as **ΛCDM**.

## ❓ Problem Statement

- Can Type Ia Supernovae data be used to estimate the **Hubble Constant (H₀)**?
- Can the **expansion of the universe** be validated using the redshift–distance relation?
- Does the observed supernova data align with theoretical cosmological predictions?

## ⚙️ Methodology

1. Loaded the **Pantheon+SH0ES supernova dataset**
2. Extracted **Redshift (z)** and **Distance Modulus (μ)** values
3. Plotted **μ vs z** to study the expansion behavior
4. Observed linearity at low redshift confirming **Hubble’s Law**
5. Computed **theoretical distance modulus (μ_model)** using cosmological equations
6. Plotted **residuals (μ_obs − μ_model) vs redshift**
7. Analyzed residual behavior to validate model accuracy

## 📊 Observations & Results

### μ vs z Plot
- Linear trend for **z < 0.1**, confirming **Hubble’s Law**
- Upward curvature at higher redshifts
- Matches predictions of the **ΛCDM cosmological model**
- Indicates the influence of **dark energy**

### Residual Plot (μ_obs − μ_model vs z)
- Residuals are centered around **zero**
- No systematic deviations observed
- Low scatter indicates **high-quality data and accurate modeling**
- Confirms the reliability of Type Ia Supernovae as standard candles

## 🌍 Applications

- Measuring large-scale cosmic distances  
- Estimating the expansion rate of the universe  
- Understanding the age and structure of the universe  
- Testing and validating cosmological models  

## ✅ Conclusion

- Successfully validated **Hubble’s Law** using real supernova data
- Estimated the **Hubble Constant (H₀)**
- Observed strong agreement between theoretical and observed values
- Confirmed that the universe is **expanding**
- Residual analysis proves high model accuracy and data reliability

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- 
## ▶️ How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/lalitsridatta/SuperNova.git
