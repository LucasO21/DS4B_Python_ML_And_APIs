# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 5: ADVANCED MACHINE LEARNING 
# PART 1: SCIKIT LEARN PIPELINES
# ----

# Core
import pandas as pd
import numpy as np

# Pycaret
import pycaret.classification as clf

# Sklearn
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import SCORERS
from sklearn.metrics import roc_auc_score, confusion_matrix

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Lead Scoring
import email_lead_scoring as els

# Leads Data

leads_df = els.db_read_and_process_els_data()


# 1.0 LOADING A PYCARET MODEL
mod_1 = clf.load_model("models/best_model_0")

clf.predict_model(mod_1, data = leads_df, raw_score=True)


# 2.0 WHAT IS A SCIKIT LEARN PIPELINE?
type(mod_1)

mod_1[0]
mod_1[1]
mod_1[2]
mod_1[len(mod_1) - 1]

# Keys
mod_1.__dict__.keys()

# Steps
mod_1.__dict__["steps"]


# 3.0 HOW TO SETUP A SKLEARN MACHINE LEARNING PIPELINE 

# 3.1 DATA PREPARATION




# 3.2 CREATING A SKLEARN PIPELINE ----

# Instantiate an Encoder & Connect to a Column



# Make a Pipeline




# Fit & Predict with a Pipeline




# Metrics




# 4.0 GRIDSEARCH -----

# Grid Search CV 



# Make A Pipeline With GridSearch




# Metrics




# 5.0 PCYARET COMPARISON ----- 




# CONCLUSIONS ----

# 1. See the benefit of using Pycaret (or AutoML)
# 2. Faster to get results than Scikit Learn
# 3. Handles a lot of the cumbersome preprocessing
# 4. The result is a Scikit Learn Pipeline

