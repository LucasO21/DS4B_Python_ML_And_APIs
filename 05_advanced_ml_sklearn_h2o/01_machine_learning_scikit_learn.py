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
from sklearn.ensemble import GradientBoostingClassifier

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

# Get X and y
X = leads_df\
    .drop(
        ["mailchimp_id", "user_full_name", "user_email", 
         "email_provider", "made_purchase", "optin_time"], 
        axis = 1
    )

y = leads_df["made_purchase"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 123)

# ========================================================================
# 3.2 CREATING A SKLEARN PIPELINE ----
# ========================================================================

# Instantiate an Encoder & Connect to a Column

ohe = OneHotEncoder(handle_unknown = "ignore")

transformer = make_column_transformer( (ohe, ["country_code"]) )


# Make a Pipeline
pipeline_rf = make_pipeline( transformer, RandomForestClassifier() )


# Fit & Predict with a Pipeline
pipeline_rf.fit(X_train, y_train)

pipeline_rf.predict(X_test)

pipeline_rf.predict_proba(X_test)


# ========================================================================
# Metrics
# ========================================================================

# - Accuracy
pipeline_rf.score(X_test, y_test)

# - AUC ROC
predicted_class_rf = pipeline_rf.predict_proba(X_test)[:, 1] > 0.035

roc_auc_score(y_true = y_test, y_score = predicted_class_rf)

confusion_matrix(y_true = y_test, y_pred = predicted_class_rf)


# ========================================================================
# 4.0 GRIDSEARCH -----
# ========================================================================

# - Grid Search CV 
grid_xgb = GridSearchCV(
    estimator  = XGBClassifier(),
    param_grid =  {"learning_rate": [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]},
    cv         = 5,
    refit      = True,
    scoring    = "roc_auc"
)


# - Make A Pipeline With GridSearch
pipeline_xgb = make_pipeline(transformer, grid_xgb)

pipeline_xgb.fit(X_train, y_train)

pipeline_xgb[1].best_params_


# - Metrics
predicted_class_xgb = pipeline_xgb.predict_proba(X_test)[:, 1] > 0.035

roc_auc_score(y_true = y_test, y_score = predicted_class_xgb)

confusion_matrix(y_true = y_test, y_pred = predicted_class_xgb)



# 5.0 PCYARET COMPARISON ----- 
mod_1 = clf.load_model("models/blended_tuned_models_final")
predicted_class_pycaret = mod_1.predict_proba(leads_df.iloc[X_test.index])[:,1] > 0.035

roc_auc_score(y_true = y_test, y_score = predicted_class_pycaret)


# ========================================================================
# SAVE / LOAD ----
# ========================================================================
import joblib

# - Save
joblib.dump(pipeline_xgb, "models/scikit_learn/pipeline_xgb.pkl")

# - Load
joblib.load("models/scikit_learn/pipeline_xgb.pkl")


# ========================================================================
# CONCLUSIONS ----
# ========================================================================
# 1. See the benefit of using Pycaret (or AutoML)
# 2. Faster to get results than Scikit Learn
# 3. Handles a lot of the cumbersome preprocessing
# 4. The result is a Scikit Learn Pipeline

# ========================================================================
# ========================================================================
# ========================================================================
# ========================================================================


