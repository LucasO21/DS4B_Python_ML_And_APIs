# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING & APIS
# MODULE 4: MACHINE LEARNING | PYCARET
# =====

# ========================================================================
# LIBRARIES
# ========================================================================

# - Core
import os
import pandas as pd
import numpy as np
import pycaret.classification as clf

# - Lead Scoring
import email_lead_scoring as els

# RECAP ----

leads_df = els.db_read_and_process_els_data() 


# ========================================================================
# 1.0 PREPROCESSING (SETUP) ---- 
# ========================================================================
# - Infers data types (requires user to say yes)
# - Returns a preprocessing pipeline
# ?clf.setup

leads_df.info()

# - Removing Unnecessary Columns
df = leads_df \
    .drop(
        [
            "mailchimp_id", 
            "user_full_name", 
            "user_email", 
            "optin_time", 
            "email_provider"
        ], axis = 1
    )


# - Specify Feature Datatypes

# -- Numeric Features
_tag_mask = df.columns.str.match("^tag_")

numeric_features = df.columns[_tag_mask].to_list()

numeric_features.append("optin_days")

# -- Categorical Features
categorical_features = ["country_code"]

# -- Ordinal Features
ordinal_features = {"member_rating": ["1", "2", "3", "4", "5"]}


# - Classifier Setup
clf_1 = clf.setup(
    
    # main
    data                       = df, 
    target                     = "made_purchase",
    train_size                 = 0.8,
    preprocess                 = True,
    imputation_type            = "simple",
    
    # categorical
    categorical_features       = categorical_features,
    handle_unknown_categorical = True,
    combine_rare_levels        = True,
    rare_level_threshold       = 0.005,
    
    # ordinal
    ordinal_features           = ordinal_features,
    
    # numeric
    numeric_features           = numeric_features,
    
    # k-fold
    fold_strategy              = "stratifiedkfold",
    fold                       = 5,
    
    # experiment logging
    n_jobs                     = 1,
    session_id                 = 123,
    log_experiment             = True,
    experiment_name            = "email_lead_scoring_0",
    
    # silent: turns off asking if data types infered correctly
    silent                     = False
)


# ========================================================================
# 2.0 GET CONFIGURATION ----
# ========================================================================
# - Understanding what Pycaret is doing underneath
# - Can extract pre/post transformed data
# - Get the Scikit Learn Pipeline
# - Specify ordinal, categorical, and numeric features

# - Transformed Dataset
# ?clf.get_config

clf.get_config("data_before_preprocess")

clf.get_config("X")

clf.get_config("y_test")

# - Extract Scikit learn Pipeline
pipeline = clf.get_config("prep_pipe")


# - Check difference in columns
pipeline.fit(df)


# ========================================================================
# 3.0 MACHINE LEARNING (COMPARE MODELS) ----
# ========================================================================

# - Available Models
clf.models()


# - Running All Available Models
best_models = clf.compare_models(
    sort        = "AUC",
    n_select    = 3,
    budget_time = 3
)


# - Get the grid (metrics)
clf.pull()


# - Top 3 Models
best_models

best_models[0]


# - Make predictions
clf.predict_model(best_models[0])

clf.predict_model(
    estimator = best_models[1],
    data      = df.iloc[[1]]
)

# - Refits on Full Dataset
best_model_0_finalized = clf.finalize_model(best_models[0])

best_model_1_finalized = clf.finalize_model(best_models[1])

best_model_2_finalized = clf.finalize_model(best_models[2])



# - Save / load model
os.mkdir("models")

clf.save_model(
    model      = best_model_0_finalized,
    model_name = "models/best_model_0"
)

clf.save_model(
    model      = best_model_1_finalized,
    model_name = "models/best_model_1"
)

clf.save_model(
    model      = best_model_2_finalized,
    model_name = "models/best_model_2"
)

clf.load_model("models/best_model_0")

# ========================================================================
# 4.0 PLOTTING MODEL PERFORMANCE -----
# ========================================================================

# Get all plots 
# - Note that this can take a long time for certain plots
# - May want to just plot individual (see that next)
clf.evaluate_model(best_model_0_finalized)

# - ROC Curves & PR Curves
clf.plot_model(best_models[1], plot =  "auc")

clf.plot_model(best_models[1], plot =  "pr")

clf.plot_model(best_models[6], plot =  "tree")

# Confusion Matrix / Error
clf.plot_model(best_models[1], plot =  "confusion_matrix")

# Gain/Lift Plots
clf.plot_model(best_models[1], plot =  "gain")

clf.plot_model(best_models[1], plot =  "lift")


# Feature Importance
clf.plot_model(best_models[1], plot =  "feature")


# Shows the Precision/Recall/F1
clf.plot_model(best_models[1], plot =  "class_report")


# Get model parameters used
clf.plot_model(best_models[0], plot =  "parameter")



# ========================================================================
# 5.0 CREATING & TUNING INDIVIDUAL MODELS ----
# ========================================================================

clf.models()

# - Create Models
gbc_model = clf.create_model(estimator = "gbc")

cat_model = clf.create_model(estimator = "catboost")

ada_model = clf.create_model(estimator = "ada")

lgbm_model = clf.create_model(estimator = "lightgbm")

xgb_model = clf.create_model(estimator = "xgboost")


# - Tuning Models
gbc_model_tuned = clf.tune_model(
    estimator = gbc_model,
    n_iter    = 5,
    optimize  = "AUC"
)

catboost_model_tuned = clf.tune_model(
    estimator = cat_model,
    n_iter    = 5,
    optimize  = "AUC"
)

adaboost_model_tuned = clf.tune_model(
    estimator = ada_model,
    n_iter    = 5,
    optimize  = "AUC"
)

lgbm_model_tuned = clf.tune_model(
    estimator = lgbm_model,
    n_iter    = 5,
    optimize  = "AUC"
)

xgb_model_tuned = clf.tune_model(
    estimator = xgb_model,
    n_iter    = 5,
    optimize  = "AUC"
)

# - Finalize tuned model 
gbc_model_model_tuned_finalized = clf.finalize_model(gbc_model_tuned)

catboost_model_tuned_finalized = clf.finalize_model(catboost_model_tuned)

adaboost_model_tuned_finalized = clf.finalize_model(adaboost_model_tuned)

lgbm_model_model_tuned_finalized = clf.finalize_model(lgbm_model_tuned)

xgb_model_tuned_finalized = clf.finalize_model(xgb_model_tuned)


# - Save tunded models
clf.save_model(
    model      = gbc_model_model_tuned_finalized,
    model_name = "models/gbc_model_model_tuned_finalized"
)

clf.save_model(
    model      = catboost_model_tuned_finalized,
    model_name = "models/catboost_model_tuned_finalized"
)

clf.save_model(
    model      = adaboost_model_tuned_finalized,
    model_name = "models/adaboost_model_tuned_finalized"
)

clf.save_model(
    model      = lgbm_model_model_tuned_finalized,
    model_name = "models/lgbm_model_model_tuned_finalized"
)

clf.save_model(
    model      = xgb_model_tuned_finalized,
    model_name = "models/xgb_model_tuned_finalized"
)



# ========================================================================
# 6.0 INTERPRETING MODELS ----
# ========================================================================

# - SHAP Package Integration
# ?clf.interpret_model


# - 1. Summary Plot: Overall top features
clf.interpret_model(catboost_model_tuned, plot = "summary")

clf.interpret_model(lgbm_model_tuned, plot = "summary")


# - 2. Analyze Specific Features 

# -- Our Exploratory Function
els.explore_sales_by_category(
    leads_df, 
    'member_rating', 
    sort_by='prop_in_group'
)

# -- Correlation Plot
clf.interpret_model(
    estimator = catboost_model_tuned, 
    plot      = "correlation",
    feature   = "optin_days"    
)

# -- Partial Dependence Plot
clf.interpret_model(
    estimator = best_models[1], 
    plot      = "pdp",
    feature   = "member_rating",
    ice       = True    
)


# - 3. Analyze Individual Observations
leads_df.iloc[[0]]

# Shap Force Plot
clf.interpret_model(
    estimator    = catboost_model_tuned, 
    plot         = "reason",
    X_new_sample = leads_df.iloc[[0]]
)

clf.interpret_model(
    estimator    = catboost_model_tuned, 
    plot         = "reason",
    X_new_sample = leads_df.iloc[[11]]
)

clf.interpret_model(
    estimator    = xgb_model_tuned, 
    plot         = "reason",
    X_new_sample = leads_df.iloc[[0]]
)

clf.interpret_model(
    estimator    = xgb_model_tuned, 
    plot         = "reason",
    X_new_sample = leads_df.iloc[[11]]
)

# - 4. Feature importance
clf.plot_model(catboost_model_tuned, plot =  "feature")

clf.plot_model(xgb_model_tuned, plot =  "feature")

# 5. Gain 
clf.plot_model(catboost_model_tuned, plot =  "gain")

clf.plot_model(catboost_model_tuned, plot =  "lift")

# 6. Confusion Matrix
clf.plot_model(catboost_model_tuned, plot = "confusion_matrix")

clf.plot_model(xgb_model_tuned, plot = "confusion_matrix")

# ========================================================================
# 7.0 BLENDING MODELS (ENSEMBLES)
# ========================================================================

blend_models = clf.blend_models(best_models, optimize = "AUC")

blend_models_tuned = clf.blend_models(
    estimator_list = [catboost_model_tuned, lgbm_model_tuned, xgb_model_tuned],
    optimize       = "AUC"
)


# ========================================================================
# 8.0 CALIBRATION
# ========================================================================
# - Improves the probability scoring (makes the probability realistic)

blended_models_calibrated = clf.calibrate_model(blend_models)

blended_tuned_models_calibrated = clf.calibrate_model(blend_models_tuned)

clf.plot_model(blend_models, plot = "calibration")

clf.plot_model(blended_models_calibrated, plot = "calibration")

# - Blended Model Metrics Plots
# clf.plot_model(blended_models_calibrated, plot =  "auc") # auc-roc plot

# clf.plot_model(blended_models_calibrated, plot =  "confusion_matrix") # confusion matrix

# clf.plot_model(best_models[1], plot =  "pr")

# clf.plot_model(xgb_model_tuned, plot =  "tree")

# ========================================================================
# 9.0 FINALIZE MODEL
# ========================================================================
# - Equivalent of refitting on full dataset

blended_models_final = clf.finalize_model(blended_models_calibrated)

blended_tuned_models_final = clf.finalize_model(blended_tuned_models_calibrated)


# ========================================================================
# 10.0 MAKING PREDICTIONS & RANKING LEADS
# ========================================================================

# - Prediction
predictions_df = clf.predict_model(
    estimator = blended_models_final,
    data      = leads_df
)

predictions_df.query("Label = 1").sort_values("Score", ascending = False)

# - Scoring
leads_scored_df = pd.concat([1 - predictions_df["Score"], leads_df], axis = 1)

leads_scored_df.sort_values("Score", ascending = False)

# ========================================================================
# SAVING / LOADING PRODUCTION MODELS
# ========================================================================

clf.save_model(
    model      = blended_models_final,
    model_name = "models/blended_model_final"
)

clf.save_model(
    model      = blended_tuned_models_final,
    model_name = "models/blended_tuned_models_final"
)

# ========================================================================
# CONCLUSIONS
# ========================================================================

# - We now have an email lead scoring model
# - Pycaret simplifies the process of building, selecting, improving machine learning models
# - Scikit Learn would take 1000's of lines of code to do all of this

# ========================================================================
# ========================================================================
# ========================================================================
# ========================================================================