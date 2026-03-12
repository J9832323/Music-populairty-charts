# 🎵 Predictive Analytics: Music Popularity & Billboard Hot 100

**Tools:** R, Machine Learning, Ensemble Modeling, Neural Networks  
**Skills:** Predictive Modeling, Cross-Validation, Data Preprocessing, Data Visualization

## Overview

Can data predict a hit song? This project builds a predictive framework for Billboard Hot 100 success, giving record labels data-backed insights to optimize promotional spend and reduce market risk.
Using R and a dataset of song attributes — including TikTok virality metrics, streaming trends, genre, and audio features — I developed and compared multiple machine learning models to forecast chart performance.

---
## Models Built

| Model | Approach |
|-------|----------|
| Linear Regression | Baseline predictor |
| k-Nearest Neighbors (k-NN) | Tuned via 5-fold cross-validation |
| Neural Network | Min-max scaled inputs, inverse-transformed outputs |
| **Ensemble** | Average of all three — best overall performance |

The ensemble model outperformed every individual model on RMSE and R-squared, demonstrating that combining predictions reduces error better than any single approach.

---

## Key Technical Details

- **Cross-validation:** 5-fold CV used to tune k in k-NN (k = 1 to 25)
- **Scaling:** Min-max normalization applied for neural network; inverse-transformed for interpretable output
- **Evaluation metrics:** RMSE and R-squared across all models
- **Feature engineering:** Dummy coding for Genre variable using `fastDummies`; dropped non-predictive identifiers (Song/Artist)
- **Libraries:** `caret`, `neuralnet`, `tidyverse`, `fastDummies`

---

## What I Found

The ensemble model delivered the strongest predictions. Genre, TikTok engagement, and streaming volume were among the most influential features. The project also surfaced what additional data (e.g., radio airplay, label marketing spend) could improve accuracy in future iterations.
---
## Files

- `Music_Popularity.Rmd` — Full analysis with code and commentary
- `data/` — Dataset used for modeling
---
