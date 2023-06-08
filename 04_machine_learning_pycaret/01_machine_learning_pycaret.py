# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING & APIS
# MODULE 4: MACHINE LEARNING | PYCARET
# ----

# Core
import os
import pandas as pd
import numpy as np
import pycaret.classification as clf

# Lead Scoring
import email_lead_scoring as els

# RECAP ----

leads_df = els.db_read_and_process_els_data() 



# 1.0 PREPROCESSING (SETUP) ---- 
# - Infers data types (requires user to say yes)
# - Returns a preprocessing pipeline






# 2.0 GET CONFIGURATION ----
# - Understanding what Pycaret is doing underneath
# - Can extract pre/post transformed data
# - Get the Scikit Learn Pipeline


# Transformed Dataset



# Extract Scikit learn Pipeline



# Check difference in columns



# 3.0 MACHINE LEARNING (COMPARE MODELS) ----

# Available Models


# Running All Available Models



# Get the grid



# Top 3 Models



# Make predictions



# Refits on Full Dataset



# Save / load model



# 4.0 PLOTTING MODEL PERFORMANCE -----



# Get all plots 
# - Note that this can take a long time for certain plots
# - May want to just plot individual (see that next)


# - ROC Curves & PR Curves


# Confusion Matrix / Error


# Gain/Lift Plots


# Feature Importance


# Shows the Precision/Recall/F1


# Get model parameters used





# 5.0 CREATING & TUNING INDIVIDUAL MODELS ----



# Create more models




# Tuning Models



# Save xgb tuned


# 6.0 INTERPRETING MODELS ----
# - SHAP Package Integration



# 1. Summary Plot: Overall top features


# 2. Analyze Specific Features ----

# Our Exploratory Function
els.explore_sales_by_category(
    leads_df, 
    'member_rating', 
    sort_by='prop_in_group'
)

# Correlation Plot


# Partial Dependence Plot


# 3. Analyze Individual Observations


# Shap Force Plot



# 7.0 BLENDING MODELS (ENSEMBLES) -----



# 8.0 CALIBRATION ----
# - Improves the probability scoring (makes the probability realistic)




# 9.0 FINALIZE MODEL ----
# - Equivalent of refitting on full dataset


# 10.0 MAKING PREDICTIONS & RANKING LEADS ----

# Prediction

# Scoring


# SAVING / LOADING PRODUCTION MODELS -----



# CONCLUSIONS ----

# - We now have an email lead scoring model
# - Pycaret simplifies the process of building, selecting, improving machine learning models
# - Scikit Learn would take 1000's of lines of code to do all of this


